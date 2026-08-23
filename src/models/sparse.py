"""Sparse feature extractor — il metodo di Kavaki & Mandel.

L'idea: invece di dare al transformer tutti i frame temporali, gli si danno
N TOKEN LATENTI, ciascuno accoppiato a una REGIONE del piano tempo-frequenza
da cui estrae informazione. Token e regioni iniziali sono PARAMETRI APPRESI:
il modello impara dove guardare, non solo cosa farne.

Configurazione dichiarata: N=4 token, P=36 campioni, d_token=64,
d_encoder=128, L_rep=3 ripetizioni, L_enc=8 encoder.

FORMULE — trascritte dal paper (eq. 7.1-7.8 del manuale), con i dettagli
non dichiarati presi dal codice ufficiale di SparseFormer
(github.com/showlab/sparseformer -> imagenet/models/sparseformer.py).
Nessuna riga copiata.

CONVENZIONE SUGLI ASSI. La feature map e' [B, C, F, T]: F (frequenza) e'
l'altezza, T (tempo) la larghezza. Le regioni vivono in coordinate
NORMALIZZATE [0, 1] su entrambi gli assi, e sono memorizzate come angoli
opposti (x1, y1, x2, y2) come in SparseFormer — centro e dimensione si
ricavano al volo. `grid_sample` vuole invece coordinate in [-1, 1], da cui
la conversione 2*c-1 al momento del campionamento.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# "none" e' la lettura letterale del paper: nulla impedisce alle regioni di
# uscire dal piano. "clip" e' una NOSTRA variante, introdotta dopo aver
# misurato che dalla seconda ripetizione meta' dei punti campionati cade
# fuori (manuale, 12.12). Il default resta "none": le nostre aggiunte sono
# ablation dichiarate, non il comportamento di riferimento.
RegionConstraint = Literal["none", "clip"]


def boxes_to_cwh(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(x1,y1,x2,y2) -> (centro, dimensione), entrambi [..., 2]."""
    center = (boxes[..., 2:] + boxes[..., :2]) * 0.5
    size = (boxes[..., 2:] - boxes[..., :2]).abs()
    return center, size


def cwh_to_boxes(center: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    return torch.cat([center - 0.5 * size, center + 0.5 * size], dim=-1)


def init_boxes_on_grid(num_tokens: int, unit: float = 0.5) -> torch.Tensor:
    """Regioni iniziali su una griglia regolare in [0, 1].

    Il paper: «we initialize the center of the regions to a grid and the
    width and height of them to half of the feature dimension» — da cui
    `unit = 0.5`.

    La griglia e' root x root con root = sqrt(N), e da questo discende un
    fatto che spiega un dettaglio del paper: N DEVE ESSERE UN QUADRATO
    PERFETTO. Non a caso l'ablation della Tabella 3 usa N in {4, 9, 16,
    25, 36}. Non era una scelta estetica degli autori ma un vincolo
    strutturale ereditato da SparseFormer.
    """
    root = round(math.sqrt(num_tokens))
    if root * root != num_tokens:
        raise ValueError(
            f"num_tokens={num_tokens} non e' un quadrato perfetto: la griglia "
            f"di inizializzazione e' {root}x{root}. Valori ammessi: 4, 9, 16, 25, 36..."
        )
    # posizioni equispaziate in [0, 1-unit]; con root=1 si evita la
    # divisione per zero mettendo l'unica regione all'origine
    steps = torch.arange(root, dtype=torch.float32)
    steps = steps / (root - 1) if root > 1 else steps
    gx = steps.view(root, 1).expand(root, root)
    gy = steps.view(1, root).expand(root, root)
    grid = torch.stack([gx, gy], dim=-1).reshape(num_tokens, 2)
    corner = grid * (1.0 - unit)
    return torch.cat([corner, corner + unit], dim=-1)


class RegionAdjust(nn.Module):
    """Passo 1 — il token sposta e ridimensiona la propria regione.

        (tx, ty, tw, th) = Linear(t)
        x' = x + tx*w      w' = w * exp(tw)

    PERCHE' L'ESPONENZIALE, tre ragioni tutte necessarie (Faster R-CNN):

    - POSITIVITA': larghezza e altezza devono restare positive, e
      w*exp(t) lo garantisce per costruzione qualunque cosa produca la
      rete. Una forma additiva richiederebbe un clamp, che azzera il
      gradiente proprio quando e' attivo.
    - INVARIANZA DI SCALA: l'aggiornamento e' moltiplicativo, quindi lo
      stesso t raddoppia una regione piccola e una grande. La rete impara
      RAPPORTI, non incrementi assoluti.
    - SIMMETRIA: t e -t danno fattori reciproci, quindi allargare e
      restringere costano lo stesso nello spazio dei parametri.

    Anche gli spostamenti sono relativi alla dimensione della regione
    (tx*w): una regione grande si sposta di piu' a parita' di segnale.
    """

    def __init__(self, token_dim: int, constraint: RegionConstraint = "none",
                 min_size: float = 1.0 / 64) -> None:
        super().__init__()
        self.constraint = constraint
        self.min_size = min_size
        self.to_delta = nn.Linear(token_dim, 4)
        # partenza dall'identita': senza questo, alla prima iterazione le
        # regioni schizzerebbero via da una griglia scelta con cura
        nn.init.zeros_(self.to_delta.weight)
        nn.init.zeros_(self.to_delta.bias)

    # Limite sul delta logaritmico prima dell'esponenziale.  [NOSTRA AGGIUNTA]
    #
    # Ne' il paper ne' SparseFormer lo prevedono, e nel regime normale NON
    # SI ATTIVA MAI: `to_delta` parte da zero e i delta restano dell'ordine
    # dell'unita'. Ma exp() in float32 va a infinito oltre ~88, e un solo
    # passo anomalo produrrebbe regioni di dimensione inf o 0, quindi NaN
    # nel gradiente e un run di ore perso.
    #
    # exp(4) ~ 55x di crescita per ripetizione, cioe' ~163.000x sulle tre:
    # enormemente piu' di qualunque variazione sensata per regioni che
    # vivono in [0,1]. E' una rete di sicurezza, non un vincolo sul modello.
    MAX_LOG_SCALE = 4.0

    def forward(self, tokens: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        delta = self.to_delta(tokens)                      # [B, N, 4]
        center, size = boxes_to_cwh(boxes)
        center = center + delta[..., :2] * size
        log_scale = delta[..., 2:].clamp(-self.MAX_LOG_SCALE, self.MAX_LOG_SCALE)
        size = size * log_scale.exp()

        if self.constraint == "clip":
            # La regione viene riportata DENTRO il piano [0,1]x[0,1].
            #
            # Prima la dimensione, in [min_size, 1]: sopra 1 la regione non
            # entra piu' nel piano, sotto min_size e' piu' piccola di una
            # cella della feature map (16x51) e il campionamento legge quattro
            # volte lo stesso valore. Poi il centro, ristretto a
            # [size/2, 1 - size/2], che e' l'intervallo in cui la regione sta
            # tutta dentro. L'ordine conta: vincolare il centro prima della
            # dimensione lascerebbe passare regioni grandi centrate al bordo.
            #
            # NOTA SUL GRADIENTE, da dichiarare: dove il clamp morde il
            # gradiente rispetto a `delta` e' nullo, quindi quel passo non
            # corregge la regione. Non e' un blocco permanente — il token
            # cambia a ogni ripetizione e a ogni batch — ma e' una differenza
            # rispetto a una penalita' morbida, che manterrebbe segnale
            # ovunque al prezzo di un iperparametro in piu'.
            size = size.clamp(self.min_size, 1.0)
            half = 0.5 * size
            center = torch.min(torch.max(center, half), 1.0 - half)

        return cwh_to_boxes(center, size)


class SparseSampler(nn.Module):
    """Passo 2 — P punti campionati dentro la regione, per interpolazione bilineare.

        {(dx_i, dy_i)}_P = Linear(LayerNorm(t))
        offset standardizzati sull'asse dei P punti, poi divisi per 3
        x~_i = x + 0.5 * dx_i * w

    LA NORMALIZZAZIONE «A TRE DEVIAZIONI STANDARD». Il paper la nomina
    senza darne la formula; SparseFormer la fornisce:

        offset = (offset - mean(-2)) / (3 * (std(-2) + 1e-7))

    Media e deviazione sono calcolate SULL'ASSE DEI P PUNTI, cioe' fra i
    campioni di uno stesso token, non sul batch. Dopo, la deviazione vale
    1/3, quindi per una distribuzione quasi normale il 99.7% della massa
    cade in [-1, 1] — cioe', dopo il riscalamento, DENTRO la regione. E'
    la regola dei tre sigma applicata per costruzione invece che sperando
    che accada.

    Il meccanismo e' disaccoppiato: il modello controlla la FORMA della
    nuvola di campionamento, mentre posizione e dimensione restano
    governate dalla regione. Per questo servono entrambi.

    IL GRADIENTE. L'interpolazione bilineare rende differenziabile la
    SELEZIONE: dF/dx~ e' una differenza finita fra celle adiacenti
    (manuale eq. 7.6). Ne segue che il gradiente vede solo le 4 celle
    vicine, quindi la regione si muove per piccoli passi seguendo la
    pendenza locale — e che su una superficie ruvida quella differenza e'
    rumore. E' esattamente il motivo per cui si campiona sull'uscita
    della early convolution e non sullo spettrogramma grezzo.
    """

    def __init__(self, token_dim: int, num_points: int) -> None:
        super().__init__()
        self.num_points = num_points
        self.norm = nn.LayerNorm(token_dim)
        self.to_offsets = nn.Linear(token_dim, num_points * 2)

    def forward(self, tokens: torch.Tensor, boxes: torch.Tensor,
                features: torch.Tensor, *, return_coords: bool = False):
        """tokens [B,N,d] · boxes [B,N,4] · features [B,C,F,T] -> [B,N,P,C].

        `return_coords=True` restituisce anche le coordinate normalizzate dei
        P punti, [B,N,P,2], in [0,1] sugli assi (tempo, frequenza) nell'ordine
        che vuole `grid_sample`. Servono a disegnare la Figura 2 del paper —
        i punti campionati sovrapposti allo spettrogramma — e senza di esse la
        traccia mostrerebbe le regioni ma non ciò che il modello legge davvero
        al loro interno. Il percorso di training non le chiede e resta
        identico.
        """
        b, n, _ = tokens.shape
        offsets = self.to_offsets(self.norm(tokens))
        offsets = offsets.view(b, n, self.num_points, 2)

        # standardizzazione sull'asse dei punti, poi /3 (regola dei tre sigma)
        mean = offsets.mean(dim=-2, keepdim=True)
        std = offsets.std(dim=-2, keepdim=True) + 1e-7
        offsets = (offsets - mean) / (3.0 * std)

        center, size = boxes_to_cwh(boxes)                 # [B,N,2]
        coords = center.unsqueeze(-2) + 0.5 * offsets * size.unsqueeze(-2)

        # [0,1] -> [-1,1], la convenzione di grid_sample
        grid = 2.0 * coords - 1.0                          # [B,N,P,2]
        sampled = F.grid_sample(features, grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
        out = rearrange(sampled, "b c n p -> b n p c")
        return (out, coords) if return_coords else out


class AdaptiveDecoder(nn.Module):
    """Passo 3 — i P campioni tornano nel token, con pesi generati dal token.

        [Mc | Ms] = F(t),  Mc in R^{CxC},  Ms in R^{PxP}
        x1 = GELU(x0 Mc)        mixing sui canali
        x2 = GELU(Ms x1)        mixing spaziale
        t' = t + Linear(x2)     aggiornamento residuo

    PESI FISSI CONTRO PESI CONDIZIONATI. Uno strato lineare ordinario
    applica gli stessi pesi a tutti gli input: impara una trasformazione
    buona in media. Qui Mc e Ms sono FUNZIONI DEL TOKEN, quindi ogni token
    decide come mescolare i propri campioni — uno che segue una regione di
    bassa frequenza puo' pesare i canali diversamente da uno su un
    transiente. E' lo stesso principio delle hypernetwork.

    Il paper e' esplicito sul fatto che serva: «simply using a linear
    layer for this encoding is not effective».

    Il costo si vede nei parametri: F deve produrre C^2 + P^2 valori, che
    con C=96 e P=36 sono 10.512 uscite da una dimensione nascosta di
    d/4 = 16. E' il blocco piu' pesante dell'estrattore.

    L'aggiornamento e' RESIDUO: il token accumula informazione a ogni
    ripetizione invece di essere riscritto, e il gradiente ha un percorso
    diretto verso i parametri iniziali.

    L'ultimo Linear opera sul tensore APPIATTITO P*C -> d. Non e' dedotto
    dal testo — che dice solo «applied to a linear layer with dimension d»
    — ma dal BUDGET DI PARAMETRI: la lettura appiattita da' 2,85 M contro
    i 2,87 M dichiarati (-0,8%), quella ridotta sui punti 2,20 M (-23%).
    """

    def __init__(self, token_dim: int, channels: int, num_points: int,
                 hidden_div: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_points = num_points
        hidden = max(1, token_dim // hidden_div)

        self.to_weights = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.Linear(hidden, channels * channels + num_points * num_points),
        )
        self.out = nn.Linear(num_points * channels, token_dim)

    def forward(self, tokens: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
        """tokens [B,N,d] · sampled [B,N,P,C] -> tokens aggiornati [B,N,d]."""
        b, n, p, c = sampled.shape
        weights = self.to_weights(tokens)
        mc, ms = weights.split([c * c, p * p], dim=-1)
        mc = mc.reshape(b, n, c, c)
        ms = ms.reshape(b, n, p, p)

        x = F.gelu(torch.einsum("bnpc,bncd->bnpd", sampled, mc))   # canali
        x = F.gelu(torch.einsum("bnqp,bnpc->bnqc", ms, x))         # spaziale
        return tokens + self.out(x.reshape(b, n, p * c))


class SparseFeatureExtractor(nn.Module):
    """L_rep ripetizioni di {aggiusta regione, campiona, decodifica}.

    I moduli NON sono condivisi fra ripetizioni: ognuna ha i propri pesi.
    Lo conferma il budget di parametri, che con moduli condivisi darebbe
    un totale molto piu' basso di quello dichiarato.
    """

    def __init__(self, num_tokens: int = 4, num_points: int = 36,
                 token_dim: int = 64, channels: int = 96, repeats: int = 3,
                 hidden_div: int = 4, unit: float = 0.5,
                 region_constraint: RegionConstraint = "none") -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_points = num_points

        # token e regioni iniziali: PARAMETRI APPRESI, non costanti
        self.token_init = nn.Parameter(torch.empty(num_tokens, token_dim))
        nn.init.trunc_normal_(self.token_init, std=1.0)
        self.box_init = nn.Parameter(init_boxes_on_grid(num_tokens, unit))

        self.stages = nn.ModuleList(
            nn.ModuleDict({
                "adjust": RegionAdjust(token_dim, constraint=region_constraint),
                "sample": SparseSampler(token_dim, num_points),
                "decode": AdaptiveDecoder(token_dim, channels, num_points, hidden_div),
            })
            for _ in range(repeats)
        )

    def forward(self, features: torch.Tensor,
                return_trace: bool = False):
        """features [B,C,F,T] -> token [B,N,d].

        `return_trace=True` restituisce anche, per ogni stadio, le regioni
        (`boxes`), i token e le coordinate dei P punti campionati
        (`points`, [B,N,P,2] in [0,1]): servono a riprodurre la Figura 2 del
        paper, in cui si vede il campionamento passare da uniforme a
        concentrato sulle zone informative.
        """
        b = features.shape[0]
        tokens = self.token_init.unsqueeze(0).expand(b, -1, -1)
        boxes = self.box_init.unsqueeze(0).expand(b, -1, -1)
        trace = []

        for stage in self.stages:
            boxes = stage["adjust"](tokens, boxes)
            if return_trace:
                sampled, coords = stage["sample"](tokens, boxes, features,
                                                  return_coords=True)
            else:
                sampled = stage["sample"](tokens, boxes, features)
            tokens = stage["decode"](tokens, sampled)
            if return_trace:
                trace.append({"boxes": boxes.detach(), "tokens": tokens.detach(),
                              "points": coords.detach()})

        return (tokens, trace) if return_trace else tokens
