"""Test dell'infrastruttura di misura e dei modelli.

Nessuno di questi test richiede il dataset scaricato: verificano forme,
invarianti numeriche e il contatore di FLOPs su tensori sintetici. Servono
a separare i bug dell'infrastruttura da quelli del modello, PRIMA di
spendere ore di training.

    pytest -q
"""

from __future__ import annotations

import pytest
import torch

from src.data.augment import apply_augmentations, mixing, one_hot, phasemix
from src.data.features import LogMelSpectrogram
from src.data.speech_commands import (
    EXPECTED_COUNTS,
    EXPECTED_TOTAL,
    LABEL_TO_INDEX,
    LABELS,
)
from src.flops import analyze, count_parameters
from src.utils import ModelEMA

BATCH = 8
CLIP = 16_000
NUM_CLASSES = 35


# --------------------------------------------------------------------------
# Costanti del protocollo
# --------------------------------------------------------------------------

def test_label_set_matches_paper():
    assert len(LABELS) == 35
    assert len(set(LABELS)) == 35
    assert LABEL_TO_INDEX["backward"] == 0
    assert LABEL_TO_INDEX["zero"] == 34


def test_expected_counts_match_paper():
    # I tre numeri dichiarati in Kavaki & Mandel, sez. 3.1
    assert EXPECTED_COUNTS == {"train": 84_843, "validation": 9_981, "test": 11_005}
    assert EXPECTED_TOTAL == 105_829


# --------------------------------------------------------------------------
# Front-end tempo-frequenza
# --------------------------------------------------------------------------

def test_logmel_shape_is_64x101():
    """64 bin mel x 101 frame: e' la forma che il resto del modello assume."""
    front = LogMelSpectrogram()
    wave = torch.randn(BATCH, CLIP)
    spec = front(wave)
    assert spec.shape == (BATCH, 1, 64, 101)
    assert front.n_frames(CLIP) == 101


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [mixing, phasemix])
def test_augmentation_produces_multi_hot_targets(fn):
    """I target sono MULTI-HOT, non distribuzioni pesate per lam.

    E' la semantica del ramo 'bce' di EAT: se la miscela contiene
    udibilmente entrambe le parole, entrambe le classi valgono 1. La BCE
    tratta ogni classe come una domanda binaria indipendente.
    """
    wave = torch.randn(BATCH, CLIP)
    targets = one_hot(torch.randint(NUM_CLASSES, (BATCH,)), NUM_CLASSES)
    mixed, mixed_t = fn(wave, targets)
    assert mixed.shape == wave.shape
    assert mixed_t.shape == targets.shape
    assert torch.isfinite(mixed).all()
    # solo valori 0 o 1, mai frazioni
    assert torch.all((mixed_t == 0) | (mixed_t == 1))
    # una o due classi attive per campione
    active = mixed_t.sum(dim=1)
    assert torch.all((active == 1) | (active == 2))


def test_lam_acts_as_threshold_not_as_weight():
    """lam >= 0.9: il secondo suono e' troppo debole e perde l'etichetta."""
    from src.data.augment import LAM_LABEL_THRESHOLD, mix_targets

    a = one_hot(torch.zeros(4, dtype=torch.long), NUM_CLASSES)
    b = one_hot(torch.ones(4, dtype=torch.long), NUM_CLASSES)

    below = mix_targets(a, b, torch.full((4,), LAM_LABEL_THRESHOLD - 0.05))
    assert torch.all(below.sum(dim=1) == 2)

    above = mix_targets(a, b, torch.full((4,), LAM_LABEL_THRESHOLD + 0.05))
    assert torch.all(above.sum(dim=1) == 1)
    assert torch.equal(above, a)


def test_phasemix_with_lam_one_is_the_identity():
    """Con lam=1 la fase mixata coincide con la propria: deve tornare x.

    Verifica in un colpo solo la correttezza del percorso STFT -> modifica
    -> iSTFT e la fedelta' della ricostruzione.
    """
    torch.manual_seed(0)
    wave = torch.randn(BATCH, CLIP)
    targets = one_hot(torch.randint(NUM_CLASSES, (BATCH,)), NUM_CLASSES)
    mixed, mixed_t = phasemix(wave, targets, lam=1.0)
    inner = slice(400, CLIP - 400)          # esclude i bordi del padding
    err = (mixed[:, inner] - wave[:, inner]).abs().max()
    assert float(err) < 1e-4
    assert torch.allclose(mixed_t, targets, atol=1e-5)


def test_mixing_renormalises_energy():
    """La divisione per sqrt(p^2+(1-p)^2) tiene costante l'energia attesa.

    Senza, il volume della miscela diventerebbe un indizio dell'avvenuta
    augmentation e il modello potrebbe sfruttarlo.
    """
    torch.manual_seed(0)
    wave = torch.randn(512, 4000)
    targets = one_hot(torch.randint(NUM_CLASSES, (512,)), NUM_CLASSES)
    mixed, _ = mixing(wave, targets)
    ratio = mixed.pow(2).mean().sqrt() / wave.pow(2).mean().sqrt()
    assert 0.8 < float(ratio) < 1.25


def test_stem_follows_sparseformer_sequence():
    """conv(bias) -> ReLU -> maxpool -> LayerNorm sui canali, nessuna BatchNorm.

    E' la sequenza di SparseFormer, che scioglie l'ambiguita' del paper sui
    canali: "a 7x7 stride-2 convolution, a ReLU, and a 3x3 stride-2 max
    pooling to extract initial 96-d image features". Nessuna BatchNorm: la
    normalizzazione e' sui canali e viene dopo il pooling.
    """
    from src.models.frontend import ChannelLayerNorm, EarlyConv

    stem = EarlyConv(channel_reading="c96")
    kinds = [type(m).__name__ for m in stem.body]
    assert kinds == ["Conv2d", "ReLU", "MaxPool2d", "ChannelLayerNorm"], kinds
    assert stem.body[0].bias is not None
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in stem.modules())
    assert isinstance(stem.body[-1], ChannelLayerNorm)


def test_channel_layernorm_normalises_over_channels():
    from src.models.frontend import ChannelLayerNorm

    x = torch.randn(2, 32, 5, 7) * 3.0 + 1.0
    y = ChannelLayerNorm(32)(x)
    assert y.shape == x.shape
    # media e varianza vanno a 0 e 1 lungo l'asse dei canali, per ogni pixel
    assert torch.allclose(y.mean(dim=1), torch.zeros(2, 5, 7), atol=1e-5)
    assert torch.allclose(y.std(dim=1, unbiased=False), torch.ones(2, 5, 7), atol=1e-2)


def test_epoch_mix_disables_augmentation_before_the_threshold():
    """`epoch_mix` fa partire il training su dati puliti, come in EAT."""
    wave = torch.randn(BATCH, CLIP)
    labels = torch.randint(NUM_CLASSES, (BATCH,))
    out, targets = apply_augmentations(
        wave, labels, NUM_CLASSES, epoch=3, epoch_mix=5
    )
    assert torch.equal(out, wave)
    assert torch.equal(targets, one_hot(labels, NUM_CLASSES))


# --------------------------------------------------------------------------
# Contatore di FLOPs
# --------------------------------------------------------------------------

class _TinyNet(torch.nn.Module):
    """Rete di calibrazione con un costo calcolabile a mano."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(100, 200, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_flops_counter_counts_flops_not_macs():
    """Il contatore conta FLOPs, non MAC.

    E' l'avvertenza centrale di src/flops.py: un Linear 100->200 senza bias
    costa 100*200 = 20_000 MAC, cioe' 40_000 FLOPs, e il test fissa che il
    numero riportato sia il secondo. Il paper non dichiara la propria
    convenzione, quindi la nostra va dichiarata e sorvegliata.
    """
    report = analyze(_TinyNet(), (100,))
    assert report.params == 100 * 200
    if report.flops is not None:
        assert report.flops == pytest.approx(40_000, rel=1e-6)


# --------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------


# La configurazione ADOTTATA, cioe' `configs/dense_flatten.yaml`: e' quella
# da cui vengono i numeri riportati nel README.
ADOPTED_DENSE = {
    "channel_reading": "c96",
    "pool_stride": (2, 1),
    "seq_mode": "flatten",
    "dim": 224,
    "depth": 8,
    "num_heads": 7,
}


def test_adopted_dense_still_costs_what_we_reported():
    """Guardia di regressione sul modello denso: sono ancora i NOSTRI numeri?

    Il test fissa cio' che abbiamo riportato, non cio' che dichiara il paper.
    La distinzione non e' pedanteria: una soglia costruita attorno ai valori
    del paper si rompe quando ci allontaniamo da loro, che e' un fatto da
    discutere e non un difetto da segnalare; una soglia costruita attorno ai
    nostri si rompe quando cambia il MODELLO, che e' l'unica cosa che un test
    di regressione deve sorvegliare.

    Per riferimento, non come vincolo: il paper dichiara 4.80 M parametri e
    0.645 G FLOPs, quindi siamo a +8.3% e -13.1%. Lo scarto e' dichiarato nel
    README e discusso nella presentazione.
    """
    from src.flops import analyze
    from src.models.dense import DenseAudioTransformer

    model = DenseAudioTransformer(**ADOPTED_DENSE)
    assert model.seq_len == 51, model.seq_len
    assert (model.feature_shape.channels, model.feature_shape.freq) == (96, 16)

    rep = analyze(model, (1, 64, 101))
    assert rep.params == 5_197_795, f"parametri cambiati: {rep.params:,}"
    assert rep.flops == 560_431_424, f"FLOPs cambiati: {rep.flops:,}"


def test_ema_moves_towards_model_and_lags_it():
    torch.manual_seed(0)
    model = _TinyNet()
    ema = ModelEMA(model, decay=0.9)

    start = ema.module.fc.weight.clone()
    with torch.no_grad():
        model.fc.weight.add_(1.0)
    ema.update(model)
    after = ema.module.fc.weight

    # si e' mosso verso il modello...
    assert not torch.allclose(after, start)
    # ...ma di 1 - decay, non fino in fondo: e' il senso della media mobile
    assert torch.allclose(after, start + 0.1, atol=1e-6)


# ==========================================================================
# Estrattore sparso — il metodo di Kavaki & Mandel
# ==========================================================================

ADOPTED_SPARSE = {"channel_reading": "c96", "pool_stride": (2, 1)}


def test_grid_init_covers_the_plane_and_rejects_non_squares():
    """Le regioni iniziali coprono [0,1]^2 su una griglia sqrt(N) x sqrt(N).

    Il vincolo sul quadrato perfetto non e' nostro: discende dalla griglia,
    ed e' il motivo per cui l'ablation del paper usa 4, 9, 16, 25, 36.
    """
    from src.models.sparse import init_boxes_on_grid

    boxes = init_boxes_on_grid(4, unit=0.5)
    assert boxes.shape == (4, 4)
    assert float(boxes.min()) == 0.0 and float(boxes.max()) == 1.0
    # quattro quadranti, ciascuno di lato 0.5
    _, size = __import__("src.models.sparse", fromlist=["boxes_to_cwh"]).boxes_to_cwh(boxes)
    assert torch.allclose(size, torch.full((4, 2), 0.5), atol=1e-6)

    for bad in (2, 5, 10):
        with pytest.raises(ValueError, match="quadrato perfetto"):
            init_boxes_on_grid(bad)


def test_region_adjust_starts_from_the_identity():
    """A pesi inizializzati a zero le regioni non si muovono.

    Serve: le regioni iniziali sono su una griglia scelta con cura, e farle
    schizzare via alla prima iterazione sprecherebbe quell'inizializzazione.
    """
    from src.models.sparse import RegionAdjust, init_boxes_on_grid

    adj = RegionAdjust(64)
    boxes = init_boxes_on_grid(4).unsqueeze(0)
    out = adj(torch.randn(1, 4, 64), boxes)
    assert torch.allclose(out, boxes, atol=1e-6)


def test_region_adjust_survives_pathological_deltas():
    """Positivita' E finitezza anche con delta assurdi.

    La forma w*exp(t) garantisce w > 0 per costruzione — e' la ragione per
    cui il paper la preferisce a w+t, che richiederebbe un clamp capace di
    azzerare il gradiente. Ma exp() in float32 esplode oltre ~88, e senza
    limite sul delta logaritmico si otterrebbero regioni di dimensione inf
    o 0, quindi NaN nel gradiente.

    Il limite MAX_LOG_SCALE e' una NOSTRA aggiunta, non prevista dal paper
    ne' da SparseFormer, ed e' dichiarata nel README.
    """
    from src.models.sparse import RegionAdjust, boxes_to_cwh, init_boxes_on_grid

    torch.manual_seed(0)
    adj = RegionAdjust(64)
    torch.nn.init.normal_(adj.to_delta.weight, std=3.0)   # delta patologici
    torch.nn.init.normal_(adj.to_delta.bias, std=3.0)
    boxes = init_boxes_on_grid(4).unsqueeze(0).expand(16, -1, -1)
    out = adj(torch.randn(16, 4, 64) * 3.0, boxes)
    _, size = boxes_to_cwh(out)
    assert (size > 0).all(), "regioni degenerate: exp e' andato a zero"
    assert torch.isfinite(out).all(), "regioni infinite: exp e' esploso"


def test_region_mode_freezes_the_right_parameters():
    """`grid` congela, senza cambiare la forma del modello.

    Il punto dell'ablation e' che sia un confronto CONTROLLATO: congelare non
    deve togliere parametri ne' FLOPs, altrimenti si starebbe confrontando un
    modello piu' piccolo invece dello stesso modello senza saliency appresa.
    """
    from src.models.sparse import SparseFeatureExtractor

    tot = None
    for mode in ("learned", "grid"):
        ex = SparseFeatureExtractor(region_mode=mode)
        n = sum(p.numel() for p in ex.parameters())
        tot = n if tot is None else tot
        assert n == tot, "il congelamento non deve cambiare il numero di parametri"

        adjust_frozen = all(not p.requires_grad
                            for st in ex.stages for p in st["adjust"].parameters())
        assert adjust_frozen == (mode != "learned")
        assert ex.box_init.requires_grad == (mode != "grid")

        # i moduli restano al loro posto: si congela, non si sostituisce
        assert len(ex.stages) == 3


def test_frozen_regions_do_not_move():
    """Con `grid` le regioni restano ESATTAMENTE sulla griglia iniziale.

    `to_delta` parte da zero, quindi congelarlo rende l'aggiustamento
    l'identita' a ogni ripetizione: e' il modo in cui l'ablation ottiene
    regioni fisse senza toccare la struttura del modello.
    """
    import torch

    from src.models.sparse import SparseFeatureExtractor, init_boxes_on_grid

    torch.manual_seed(0)
    ex = SparseFeatureExtractor(region_mode="grid").eval()
    _, trace = ex(torch.randn(4, 96, 16, 51), return_trace=True)
    atteso = init_boxes_on_grid(4)
    for stage in trace:
        assert torch.allclose(stage["boxes"][0], atteso, atol=1e-6)


def test_region_constraint_keeps_boxes_inside_the_plane():
    """Con `constraint="clip"` le regioni restano dentro [0,1], sempre.

    E' la variante opzionale descritta nel README: la lettura letterale del
    paper non pone vincoli sulle regioni, questa li impone. Il test usa delta
    deliberatamente patologici e verifica che il vincolo li assorba.
    """
    from src.models.sparse import RegionAdjust, boxes_to_cwh, init_boxes_on_grid

    adj = RegionAdjust(64, constraint="clip")
    torch.nn.init.normal_(adj.to_delta.weight, std=20.0)   # deliberatamente assurdi
    torch.nn.init.normal_(adj.to_delta.bias, std=20.0)

    boxes = init_boxes_on_grid(4).unsqueeze(0).expand(32, -1, -1)
    for _ in range(3):                                     # tre ripetizioni, come L_rep
        boxes = adj(torch.randn(32, 4, 64), boxes)

    assert torch.isfinite(boxes).all()
    assert float(boxes.detach().min()) >= -1e-6
    assert float(boxes.detach().max()) <= 1 + 1e-6
    _, size = boxes_to_cwh(boxes)
    assert float(size.detach().min()) >= adj.min_size - 1e-6        # niente regioni sotto-cella


def test_region_constraint_is_off_by_default():
    """Il default e' la lettura letterale del paper: nessun vincolo.

    Serve a garantire che la nostra variante non diventi il comportamento di
    riferimento per distrazione. Con gli stessi delta, "none" produce regioni
    fuori dal piano e "clip" no: se un giorno il default cambiasse, questo
    test lo direbbe.
    """
    from src.models.sparse import RegionAdjust, SparseFeatureExtractor

    assert RegionAdjust(64).constraint == "none"
    assert SparseFeatureExtractor().stages[0]["adjust"].constraint == "none"

    torch.manual_seed(0)
    libero = RegionAdjust(64, constraint="none")
    torch.nn.init.normal_(libero.to_delta.weight, std=20.0)
    boxes = libero(torch.randn(16, 4, 64),
                   torch.tensor([[0.25, 0.25, 0.75, 0.75]]).expand(16, 4, 4).clone())
    assert float(boxes.detach().max()) > 1.0        # senza vincolo si esce, ed e' voluto


def test_sampler_shapes_and_three_sigma_normalisation():
    """[B,N,P,C] in uscita, e i punti cadono quasi tutti dentro la regione.

    Gli offset vengono standardizzati sull'asse dei P punti e divisi per 3:
    dopo, la deviazione vale 1/3, quindi per una distribuzione quasi normale
    circa il 99,7% della massa sta in [-1,1], cioe' dentro la regione.
    """
    from src.models.sparse import SparseSampler, init_boxes_on_grid

    torch.manual_seed(0)
    sampler = SparseSampler(64, 36)
    boxes = init_boxes_on_grid(4).unsqueeze(0).expand(8, -1, -1)
    feats = torch.randn(8, 96, 16, 51)
    out = sampler(torch.randn(8, 4, 64), boxes, feats)
    assert out.shape == (8, 4, 36, 96)
    assert torch.isfinite(out).all()

    # gli offset normalizzati devono avere deviazione ~1/3 sull'asse dei punti
    off = sampler.to_offsets(sampler.norm(torch.randn(64, 4, 64)))
    off = off.view(64, 4, 36, 2)
    off = (off - off.mean(-2, keepdim=True)) / (3 * (off.std(-2, keepdim=True) + 1e-7))
    assert abs(float(off.detach().std(dim=-2).mean()) - 1 / 3) < 0.02
    assert float((off.detach().abs() <= 1.0).float().mean()) > 0.95


def test_decoder_update_is_residual():
    """t' = t + Linear(...): il token accumula, non viene riscritto."""
    from src.models.sparse import AdaptiveDecoder

    dec = AdaptiveDecoder(64, 96, 36)
    torch.nn.init.zeros_(dec.out.weight)
    torch.nn.init.zeros_(dec.out.bias)
    tokens = torch.randn(4, 4, 64)
    out = dec(tokens, torch.randn(4, 4, 36, 96))
    assert torch.allclose(out, tokens, atol=1e-6)


def test_extractor_learns_where_to_look():
    """Il gradiente raggiunge token E regioni iniziali.

    Se `box_init` non ricevesse gradiente, il modello non imparerebbe DOVE
    guardare — cioe' verrebbe a mancare il meccanismo centrale del paper.
    """
    from src.models.sparse import SparseFeatureExtractor

    ex = SparseFeatureExtractor(num_tokens=4, num_points=36, token_dim=64,
                                channels=96, repeats=3)
    out = ex(torch.randn(2, 96, 16, 51))
    assert out.shape == (2, 4, 64)
    out.sum().backward()
    assert ex.token_init.grad is not None and ex.token_init.grad.abs().sum() > 0
    assert ex.box_init.grad is not None and ex.box_init.grad.abs().sum() > 0


def test_sparse_model_matches_the_declared_budget():
    """Guardia di regressione sui due vincoli dichiarati dal paper.

    2,87 M parametri e 0,055 G FLOPs, i due soli numeri che il paper
    dichiara per questo modello.
    """
    from src.flops import analyze
    from src.models.sparse_model import SparseAudioTransformer

    model = SparseAudioTransformer(**ADOPTED_SPARSE)
    assert model.seq_len == 4
    assert model.feature_shape.channels == 96

    rep = analyze(model, (1, 64, 101))
    assert abs(rep.params - 2.87e6) / 2.87e6 < 0.05
    assert abs(rep.flops - 0.055e9) / 0.055e9 < 0.20


def test_params_match_the_paper_on_both_ablation_axes():
    """Nove configurazioni dichiarate, nove verifiche senza addestrare nulla.

    E' la verifica piu' forte che la lettura del metodo sia corretta, perche'
    i due assi si controllano a vicenda: lungo N i parametri dichiarati sono
    COSTANTI (2.87 M), lungo P piu' che raddoppiano (2.44 -> 5.35 M) per via
    di `Ms` in R^{PxP}. Riprodurre entrambi gli andamenti per caso e'
    improbabile; riprodurne uno solo no.
    """
    from src.models.sparse_model import SparseAudioTransformer
    from src.paper import POINTS, TOKENS

    for key, series in (("num_tokens", TOKENS), ("num_points", POINTS)):
        for value, paper_params, _gflop, _acc in series:
            model = SparseAudioTransformer(**{**ADOPTED_SPARSE, key: value})
            got = count_parameters(model)
            err = abs(got - paper_params * 1e6) / (paper_params * 1e6)
            assert err < 0.03, f"{key}={value}: {got:,} contro {paper_params} M"


def test_geometry_counts_the_points_outside_the_plane():
    """I conteggi di `geometry_from_trace`, sul modello appena inizializzato.

    Con `to_delta` inizializzato a zero le regioni restano sulla griglia e il
    lato medio vale `unit`. I punti dentro il piano non sono il 100% ma il
    ~99.7%: la normalizzazione a tre sigma lascia fuori la coda, e le regioni
    della griglia toccano il bordo. E' il caso di riferimento contro cui
    `scripts/region_geometry.py` legge i modelli addestrati.
    """
    from src.models.sparse import geometry_from_trace
    from src.models.sparse_model import SparseAudioTransformer

    model = SparseAudioTransformer(**ADOPTED_SPARSE).eval()
    spec = torch.randn(2, 1, 64, 101)
    stages = geometry_from_trace(model.sampling_trace(spec))

    assert len(stages) == 3
    for stage in stages:
        assert stage["points"] == 2 * 4 * 36
        assert stage["inside"] / stage["points"] > 0.99
        assert stage["size_sum"] / stage["boxes"] == pytest.approx(0.5)

    # il metodo del modello e' la stessa cosa, cosi' non divergono
    assert model.sampling_geometry(spec) == stages


def test_sparse_model_is_batch_independent():
    from src.models.sparse_model import SparseAudioTransformer

    torch.manual_seed(0)
    model = SparseAudioTransformer(**ADOPTED_SPARSE)
    spec = torch.randn(4, 1, 64, 101)
    with torch.no_grad():
        model.train()
        a = model(spec)
        model.eval()
        b = model(spec)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# Override della configurazione da riga di comando
# --------------------------------------------------------------------------


def test_override_refuses_an_unknown_key():
    """Un refuso creerebbe una chiave che nessuno legge: l'ablation girerebbe
    con il valore di default fingendo di essere un'ablation."""
    from src.utils import apply_overrides

    cfg = {"model": {"num_tokens": 4}}
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["model.num_token=9"])
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["modello.num_tokens=9"])
    assert cfg == {"model": {"num_tokens": 4}}


