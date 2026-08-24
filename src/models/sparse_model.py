"""Audio Sparse-Transformer completo — il modello di Kavaki & Mandel.

    spettrogramma  [B, 1, 64, 101]
      -> early convolution        [B, 96, 16, 51]
      -> sparse feature extractor [B, 4, 64]      <- da 816 celle a 4 token
      -> ponte lineare 64 -> 128  [B, 4, 128]
      -> 8 transformer encoder    [B, 4, 128]     senza codifica posizionale
      -> media + classificatore   [B, 35]

Condivide con il modello denso il front-end, l'encoder e l'intero training
loop: l'unica differenza e' COME si producono i token. E' il motivo per cui
il denso e' stato costruito per primo — quando si arriva qui, tutto il resto
e' gia' validato e l'unica variabile nuova e' l'estrattore.

NIENTE CODIFICA POSIZIONALE, ed e' dichiarato dal paper: i token latenti non
hanno un ordine intrinseco, sono un insieme e non una sequenza. Nel modello
denso la questione era aperta — i frame temporali un ordine ce l'hanno — e
infatti la testiamo in entrambe le varianti. Qui no: aggiungerla sarebbe un
errore concettuale.
"""

from __future__ import annotations

import torch
from torch import nn

from .frontend import ChannelReading, EarlyConv, NormKind, StemKind
from .sparse import RegionConstraint, RegionMode, SparseFeatureExtractor
from .transformer import TransformerEncoder


class SparseAudioTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 35,
        n_mels: int = 64,
        n_frames: int = 101,
        *,
        # front-end: identico al denso, cosi' il confronto e' controllato
        channel_reading: ChannelReading = "c196_proj96",
        stem: StemKind = "paper",
        num_conv_layers: int = 1,
        norm: NormKind = "layernorm",
        pool_stride: tuple[int, int] = (2, 1),
        # estrattore sparso
        num_tokens: int = 4,
        num_points: int = 36,
        token_dim: int = 64,
        repeats: int = 3,
        hidden_div: int = 4,
        unit: float = 0.5,
        region_constraint: RegionConstraint = "none",
        region_mode: RegionMode = "learned",
        # transformer di classificazione
        dim: int = 128,
        depth: int = 8,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.frontend = EarlyConv(
            channel_reading=channel_reading, stem=stem, pool_stride=pool_stride,
            num_conv_layers=num_conv_layers, norm=norm,
        )
        shape = self.frontend.output_shape(n_mels, n_frames)
        self.feature_shape = shape
        self.seq_len = num_tokens          # stessa interfaccia del modello denso

        self.extractor = SparseFeatureExtractor(
            num_tokens=num_tokens, num_points=num_points, token_dim=token_dim,
            channels=shape.channels, repeats=repeats, hidden_div=hidden_div,
            unit=unit, region_constraint=region_constraint,
            region_mode=region_mode,
        )
        # «two models are bridged by a linear layer to change the dimension
        # of token from 64 to 128»
        self.bridge = nn.Linear(token_dim, dim)
        self.encoder = TransformerEncoder(dim, depth, num_heads, ffn_ratio, dropout)
        self.head = nn.Linear(dim, num_classes)

        # copia immutabile della griglia iniziale, per misurare quanto le
        # regioni si sono spostate durante il training (vedi `diagnostics`).
        # E' un buffer non persistente: non finisce nei checkpoint e non
        # sporca lo state_dict.
        self.register_buffer("_box_init_0", self.extractor.box_init.detach().clone(),
                             persistent=False)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec [B, 1, n_mels, n_frames] -> logits [B, num_classes]."""
        features = self.frontend(spec)
        tokens = self.extractor(features)
        tokens = self.encoder(self.bridge(tokens))
        return self.head(tokens.mean(dim=1))

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        """Il meccanismo centrale del paper sta funzionando?

        Il rischio silenzioso di questo metodo e' che le regioni NON si
        muovano: il gradiente rispetto alle coordinate e' una differenza
        finita fra celle adiacenti, e se non porta segnale utile le regioni
        restano dove la griglia le ha messe. Il modello si addestrerebbe
        comunque — degradando a un campionamento fisso — e produrrebbe un
        numero plausibile senza che nulla nei log lo segnali.

        Due misure, entrambe gratuite (nessun forward):

        `region_init_drift` — quanto le regioni APPRESE si sono spostate
            dalla griglia iniziale. Zero significa che il modello non ha
            imparato dove guardare.

        `region_adjust_norm` — norma dei pesi che generano l'aggiustamento
            per-campione. Partono da ESATTAMENTE zero (inizializzazione
            all'identita'), quindi qualunque valore > 0 dice che il ramo di
            aggiustamento sta ricevendo gradiente e imparando.

        Entrambe a zero dopo qualche epoca = il meccanismo e' morto.
        """
        drift = (self.extractor.box_init - self._box_init_0).norm(dim=-1).mean()
        adjust = sum(float(s["adjust"].to_delta.weight.abs().sum())
                     for s in self.extractor.stages)
        return {"region_init_drift": float(drift), "region_adjust_norm": adjust}

    @torch.no_grad()
    def sampling_trace(self, spec: torch.Tensor) -> list[dict]:
        """Regioni e token a ogni stadio — per riprodurre la Figura 2 del paper.

        Il paper mostra i punti campionati sovrapposti allo spettrogramma,
        un colore per token, ai tre stadi: si vede il campionamento passare
        da uniforme a concentrato sulle regioni informative. E' la verifica
        qualitativa che il meccanismo faccia quello che dichiara, e la
        diapositiva piu' convincente della presentazione.
        """
        _, trace = self.extractor(self.frontend(spec), return_trace=True)
        return trace
