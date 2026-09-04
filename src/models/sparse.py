"""Sparse feature extractor — il metodo di Kavaki & Mandel.

Invece di tutti i frame temporali, il transformer riceve N TOKEN LATENTI,
ciascuno accoppiato a una REGIONE del piano tempo-frequenza da cui estrae
informazione. Token e regioni iniziali sono parametri appresi: il modello
impara dove guardare. Configurazione dichiarata: N=4, P=36, d_token=64,
L_rep=3.

Formule dal paper, dettagli non dichiarati dal codice ufficiale di
SparseFormer (github.com/showlab/sparseformer). Nessuna riga copiata.

ASSI. La feature map e' [B, C, F, T]: F (frequenza) e' l'altezza, T (tempo)
la larghezza. Le regioni stanno in coordinate normalizzate [0, 1] come
angoli opposti (x1, y1, x2, y2); `grid_sample` vuole [-1, 1], da cui la
conversione 2*c-1 al campionamento.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# "none" e' la lettura letterale del paper: nulla impedisce alle regioni di
# uscire dal piano, e nemmeno SparseFormer lo impedisce. "clip" e' una NOSTRA
# ablation: risponde alla domanda se quel vincolo mancante conti.
RegionConstraint = Literal["none", "clip"]

# Come si determinano le regioni. Serve a rispondere alla domanda che il paper
# pone implicitamente e non misura: la saliency appresa serve davvero?
#
#   "learned"  il metodo del paper: regioni apprese piu' aggiustamento per
#              campione a ogni ripetizione.
#   "grid"     `to_delta` e `box_init` congelati: nessuna saliency.
#
# Alla PRIMA ripetizione le regioni sono identiche per ogni ingresso anche in
# "learned", perche' i token entrano come `token_init`: l'adattivita' esiste
# solo dalla seconda in poi.
RegionMode = Literal["learned", "grid"]


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

    La griglia e' root x root con root = sqrt(N), quindi N DEVE ESSERE UN
    QUADRATO PERFETTO: e' il vincolo strutturale, ereditato da SparseFormer,
    che spiega perche' l'ablation della Tabella 3 usa N in {4, 9, 16, 25, 36}.
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

    L'esponenziale (come in Faster R-CNN) garantisce dimensioni positive per
    costruzione, rende l'aggiornamento invariante di scala — la rete impara
    rapporti, non incrementi — e simmetrico fra allargare e restringere.
    Anche gli spostamenti sono relativi alla dimensione (tx*w).
    """

    def __init__(self, token_dim: int, constraint: RegionConstraint = "none",
                 min_size: float = 1.0 / 64) -> None:
        super().__init__()
        self.constraint = constraint
        self.min_size = min_size
        self.to_delta = nn.Linear(token_dim, 4)
        # partenza dall'identita': senza, alla prima iterazione le regioni
        # schizzerebbero via dalla griglia iniziale
        nn.init.zeros_(self.to_delta.weight)
        nn.init.zeros_(self.to_delta.bias)

    # Limite sul delta logaritmico prima dell'esponenziale.  [NOSTRA AGGIUNTA]
    # Serve perche' exp() in float32 va a infinito oltre ~88: un solo passo
    # anomalo darebbe regioni inf o 0, quindi NaN nel gradiente.
    #
    # Il valore 4 lascia a una regione di moltiplicare il proprio lato per
    # e^4 ~ 55 a ogni ripetizione: molto oltre qualunque regione che stia
    # ancora dentro il piano, e molto sotto la soglia di overflow. Ne' il
    # paper ne' SparseFormer prevedono questo limite.
    MAX_LOG_SCALE = 4.0

    def forward(self, tokens: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        delta = self.to_delta(tokens)                      # [B, N, 4]
        center, size = boxes_to_cwh(boxes)
        center = center + delta[..., :2] * size
        log_scale = delta[..., 2:].clamp(-self.MAX_LOG_SCALE, self.MAX_LOG_SCALE)
        size = size * log_scale.exp()

        if self.constraint == "clip":
            # Regione riportata dentro [0,1]x[0,1]: prima la dimensione, poi
            # il centro nell'intervallo in cui la regione sta tutta dentro.
            # L'ordine conta — vincolare il centro per primo lascerebbe
            # passare regioni grandi centrate al bordo. Dove il clamp morde
            # il gradiente su `delta` e' nullo.
            size = size.clamp(self.min_size, 1.0)
            half = 0.5 * size
            center = torch.min(torch.max(center, half), 1.0 - half)

        return cwh_to_boxes(center, size)


class SparseSampler(nn.Module):
    """Passo 2 — P punti campionati nella regione, per interpolazione bilineare.

        {(dx_i, dy_i)}_P = Linear(LayerNorm(t))
        x~_i = x + 0.5 * dx_i * w

    La normalizzazione «a tre deviazioni standard», che il paper nomina senza
    formula, viene da SparseFormer:

        offset = (offset - mean(-2)) / (3 * (std(-2) + 1e-7))

    Media e deviazione sono SULL'ASSE DEI P PUNTI, non sul batch: dopo, la
    deviazione vale 1/3, quindi il 99.7% della massa cade dentro la regione
    per costruzione. Il modello controlla cosi' la FORMA della nuvola,
    mentre posizione e dimensione restano governate dalla regione.

    L'interpolazione bilineare rende differenziabile la SELEZIONE: dF/dx~ e'
    una differenza finita fra celle adiacenti, ed e' il motivo per cui si
    campiona sull'uscita della early convolution e non sullo spettrogramma.
    """

    def __init__(self, token_dim: int, num_points: int) -> None:
        super().__init__()
        self.num_points = num_points
        self.norm = nn.LayerNorm(token_dim)
        self.to_offsets = nn.Linear(token_dim, num_points * 2)

    def forward(self, tokens: torch.Tensor, boxes: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        """tokens [B,N,d] · boxes [B,N,4] · features [B,C,F,T] -> [B,N,P,C]."""
        b, n, _ = tokens.shape
        offsets = self.to_offsets(self.norm(tokens))
        offsets = offsets.view(b, n, self.num_points, 2)

        # standardizzazione sull'asse dei punti, poi /3 (tre sigma)
        mean = offsets.mean(dim=-2, keepdim=True)
        std = offsets.std(dim=-2, keepdim=True) + 1e-7
        offsets = (offsets - mean) / (3.0 * std)

        center, size = boxes_to_cwh(boxes)                 # [B,N,2]
        coords = center.unsqueeze(-2) + 0.5 * offsets * size.unsqueeze(-2)

        # [0,1] -> [-1,1], la convenzione di grid_sample
        grid = 2.0 * coords - 1.0                          # [B,N,P,2]
        sampled = F.grid_sample(features, grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
        return rearrange(sampled, "b c n p -> b n p c")


class AdaptiveDecoder(nn.Module):
    """Passo 3 — i P campioni tornano nel token, con pesi generati dal token.

        [Mc | Ms] = F(t),  Mc in R^{CxC},  Ms in R^{PxP}
        x1 = GELU(x0 Mc)        mixing sui canali
        x2 = GELU(Ms x1)        mixing spaziale
        t' = t + Linear(x2)     aggiornamento residuo

    Mc e Ms sono FUNZIONI DEL TOKEN — stesso principio delle hypernetwork —
    quindi ogni token decide come mescolare i propri campioni invece di
    subire pesi buoni in media. Il paper e' esplicito: «simply using a linear
    layer for this encoding is not effective».

    E' il blocco piu' pesante dell'estrattore: F produce C^2 + P^2 valori,
    10.512 uscite con C=96 e P=36.

    L'ultimo Linear opera sul tensore APPIATTITO P*C -> d. Il paper non lo
    dice; lo dice AdaMixer (Gao et al., CVPR 2022), che Kavaki & Mandel
    citano a fianco dell'equazione (9): "the final output ... is flattened
    and transformed to the d_q dimension by a linear layer to add back to
    the content vector".
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

    I moduli NON sono condivisi fra ripetizioni, e qui ci si discosta da
    SparseFormer, che invece li condivide ("lightweight, repeated & weight
    sharing", sua fig. 2). A imporlo e' il budget di Kavaki & Mandel: con
    pesi condivisi il modello avrebbe 2.010.591 parametri contro i 2,87 M
    dichiarati, cioe' -30 %; senza condivisione 2.822.711, cioe' -1,6 %.
    """

    def __init__(self, num_tokens: int = 4, num_points: int = 36,
                 token_dim: int = 64, channels: int = 96, repeats: int = 3,
                 hidden_div: int = 4, unit: float = 0.5,
                 region_constraint: RegionConstraint = "none",
                 region_mode: RegionMode = "learned") -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_points = num_points

        # token e regioni iniziali sono parametri appresi, non costanti
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

        # Congelamento DOPO la costruzione, senza sostituire i moduli: forma,
        # parametri e FLOPs restano identici a "learned", cosi' l'ablation e'
        # un confronto controllato e non un modello piu' piccolo.
        self.region_mode = region_mode
        if region_mode == "grid":
            for stage in self.stages:
                for param in stage["adjust"].parameters():
                    param.requires_grad_(False)
            self.box_init.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features [B,C,F,T] -> token [B,N,d]."""
        b = features.shape[0]
        tokens = self.token_init.unsqueeze(0).expand(b, -1, -1)
        boxes = self.box_init.unsqueeze(0).expand(b, -1, -1)

        for stage in self.stages:
            boxes = stage["adjust"](tokens, boxes)
            sampled = stage["sample"](tokens, boxes, features)
            tokens = stage["decode"](tokens, sampled)

        return tokens
