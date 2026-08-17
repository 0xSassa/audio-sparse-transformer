"""Test dell'infrastruttura della Fase 2.

Nessuno di questi test richiede il dataset scaricato: verificano forme,
invarianti numeriche e il contatore di FLOPs su tensori sintetici. Servono
a separare i bug dell'infrastruttura da quelli del modello, PRIMA di
spendere ore di training.

    pytest -q
"""

from __future__ import annotations

import numpy as np
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
from src.flops import analyze, count_parameters, grid_sample_flops
from src.utils import ModelEMA, seed_everything

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


def test_logmel_normalisation_is_per_sample():
    front = LogMelSpectrogram(normalize=True)
    # Due campioni con guadagni molto diversi devono normalizzare allo stesso
    # regime: e' il punto della normalizzazione per-campione.
    wave = torch.randn(2, CLIP)
    wave[1] *= 100.0
    spec = front(wave)
    per_sample_mean = spec.mean(dim=(1, 2, 3))
    per_sample_std = spec.std(dim=(1, 2, 3))
    assert torch.allclose(per_sample_mean, torch.zeros(2), atol=1e-4)
    assert torch.allclose(per_sample_std, torch.ones(2), atol=1e-2)


def test_logmel_rejects_wrong_rank():
    front = LogMelSpectrogram()
    with pytest.raises(ValueError):
        front(torch.randn(BATCH, 1, CLIP))


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------

def test_one_hot_is_proper_distribution():
    labels = torch.randint(NUM_CLASSES, (BATCH,))
    t = one_hot(labels, NUM_CLASSES)
    assert t.shape == (BATCH, NUM_CLASSES)
    assert torch.allclose(t.sum(dim=1), torch.ones(BATCH))


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


def test_mix_targets_clamps_when_classes_coincide():
    """Se i due campioni hanno la stessa classe, il clamp evita il target 2."""
    from src.data.augment import mix_targets

    a = one_hot(torch.full((4,), 7, dtype=torch.long), NUM_CLASSES)
    out = mix_targets(a, a, torch.zeros(4))
    assert float(out.max()) == 1.0
    assert torch.all(out.sum(dim=1) == 1)


def test_phasemix_returns_real_signal():
    """irfft deve restituire un segnale reale della lunghezza originale."""
    wave = torch.randn(BATCH, CLIP)
    targets = one_hot(torch.randint(NUM_CLASSES, (BATCH,)), NUM_CLASSES)
    mixed, _ = phasemix(wave, targets)
    assert mixed.dtype == torch.float32
    assert mixed.shape[-1] == CLIP


def test_phasemix_preserves_energy_scale():
    """Il mixing non deve far esplodere l'ampiezza del segnale."""
    wave = torch.randn(BATCH, CLIP)
    targets = one_hot(torch.randint(NUM_CLASSES, (BATCH,)), NUM_CLASSES)
    mixed, _ = phasemix(wave, targets)
    ratio = mixed.std() / wave.std()
    assert 0.2 < float(ratio) < 3.0


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


def test_phasemix_amplitude_is_preserved_in_the_stft_but_not_after_istft():
    """Documenta la CONSISTENZA della STFT, che non e' un dettaglio minore.

    "Preserving amplitude" vale sulla matrice STFT modificata, dove per
    costruzione il modulo e' quello dell'originale. Ma una matrice complessa
    arbitraria non e' la STFT di alcun segnale reale: la iSTFT proietta sul
    sottospazio delle STFT consistenti, e la proiezione altera anche i
    moduli. Il test fissa entrambi i fatti.
    """
    from src.data.augment import STFT_HOP, STFT_N_FFT

    torch.manual_seed(0)
    wave = torch.randn(BATCH, CLIP)
    targets = one_hot(torch.randint(NUM_CLASSES, (BATCH,)), NUM_CLASSES)
    win = torch.hann_window(STFT_N_FFT)

    before = torch.stft(wave, STFT_N_FFT, STFT_HOP, window=win,
                        center=True, return_complex=True).abs()
    mixed, _ = phasemix(wave, targets, lam=0.0)      # fase interamente dell'altro
    after = torch.stft(mixed, STFT_N_FFT, STFT_HOP, window=win,
                       center=True, return_complex=True).abs()

    rel = float((after - before).abs().mean() / before.mean())
    # non e' preservata dopo la ricostruzione...
    assert rel > 0.05
    # ...ma la deviazione resta contenuta: non e' rumore scorrelato
    assert rel < 0.8


def test_phasemix_label_lam_is_remapped_to_upper_half():
    """Il lam dell'etichetta e' lam*0.5+0.5 in [0.5, 1].

    Con la soglia a 0.9 questo significa che la seconda etichetta
    sopravvive quando lam < 0.8, cioe' nell'80% dei casi. Il test verifica
    i due estremi.
    """
    torch.manual_seed(0)
    wave = torch.randn(64, 2000)
    targets = one_hot(torch.arange(64) % NUM_CLASSES, NUM_CLASSES)

    # lam=0 -> lam_label=0.5 < 0.9 -> due etichette, salvo i punti fissi
    # della permutazione (un campione mixato con se stesso resta a una)
    _, t_low = phasemix(wave, targets, lam=0.0)
    active = t_low.sum(dim=1)
    assert torch.all((active == 1) | (active == 2))
    assert float((active == 2).float().mean()) > 0.8

    # lam=1 -> lam_label=1.0 >= 0.9 -> una sola etichetta, l'originale
    _, t_high = phasemix(wave, targets, lam=1.0)
    assert torch.equal(t_high, targets)


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


def test_mixing_keeps_second_label_when_lam_is_small():
    torch.manual_seed(0)
    wave = torch.randn(64, 2000)
    targets = one_hot(torch.arange(64) % NUM_CLASSES, NUM_CLASSES)

    _, t_low = mixing(wave, targets, lam=0.3)
    active = t_low.sum(dim=1)
    # i punti fissi di randperm mixano un campione con se stesso: il clamp
    # riporta quel target a una sola classe. E' il comportamento del
    # riferimento, non un difetto: circa 1/B dei campioni per batch.
    assert torch.all((active == 1) | (active == 2))
    assert float((active == 2).float().mean()) > 0.8

    _, t_high = mixing(wave, targets, lam=0.95)
    assert torch.equal(t_high, targets)


def test_stem_follows_sparseformer_sequence():
    """conv(bias) -> ReLU -> maxpool -> LayerNorm sui canali, nessuna BatchNorm.

    Regressione sulla deviazione corretta il 18/08: una versione precedente
    inseriva una BatchNorm fra conv e ReLU, che ne' il paper ne' il codice
    di SparseFormer prevedono.
    """
    from src.models.frontend import ChannelLayerNorm, EarlyConv

    stem = EarlyConv(channel_reading="c96", stem="paper")
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


def test_apply_augmentations_without_augs_is_identity():
    wave = torch.randn(BATCH, CLIP)
    labels = torch.randint(NUM_CLASSES, (BATCH,))
    out, targets = apply_augmentations(wave, labels, NUM_CLASSES, names=())
    assert torch.equal(out, wave)
    assert torch.allclose(targets, one_hot(labels, NUM_CLASSES))


def test_apply_augmentations_always_returns_targets_of_fixed_shape():
    """Il percorso di loss deve essere unico: sempre [B, C] float."""
    wave = torch.randn(BATCH, CLIP)
    labels = torch.randint(NUM_CLASSES, (BATCH,))
    for _ in range(10):
        out, targets = apply_augmentations(wave, labels, NUM_CLASSES)
        assert out.shape == wave.shape
        assert targets.shape == (BATCH, NUM_CLASSES)


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


def test_flops_counters_agree_up_to_factor_two():
    """fvcore conta MAC, il contatore nativo conta FLOPs: rapporto ~2x.

    E' l'avvertenza centrale di src/flops.py. Un Linear 100->200 senza bias
    costa 100*200 = 20_000 MAC, cioe' 40_000 FLOPs.
    """
    report = analyze(_TinyNet(), (100,))
    assert report.params == 100 * 200

    if report.fvcore_macs is not None:
        assert report.fvcore_macs == pytest.approx(20_000, rel=1e-6)
    if report.native_flops is not None:
        assert report.native_flops == pytest.approx(40_000, rel=1e-6)
    if report.native_flops is not None and report.fvcore_macs is not None:
        assert report.native_flops / report.fvcore_macs == pytest.approx(2.0, rel=1e-6)


def test_flops_report_adds_manual_extras():
    extra = grid_sample_flops(num_points=36, channels=96)
    report = analyze(_TinyNet(), (100,), manual_extra_flops=extra)
    assert report.manual_extra_flops == extra
    if report.native_flops is not None:
        assert report.native_total == report.native_flops + extra


def test_grid_sample_flops_scales_with_points_and_channels():
    assert grid_sample_flops(72, 96) > grid_sample_flops(36, 96)
    assert grid_sample_flops(36, 192) > grid_sample_flops(36, 96)


def test_count_parameters_respects_trainable_flag():
    net = _TinyNet()
    assert count_parameters(net) == 20_000
    net.fc.weight.requires_grad_(False)
    assert count_parameters(net, trainable_only=True) == 0


# --------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------

def test_dataset_is_picklable_without_the_memmap(tmp_path):
    """Regressione: su Windows il DataLoader usa 'spawn' e PICKLA il Dataset.

    Se la memmap restasse un attributo dell'istanza, pickle proverebbe a
    serializzare 2.7 GB e fallirebbe con OSError [Errno 22]. Il test
    costruisce una cache finta minuscola e verifica che il round-trip
    funzioni e che la memmap non finisca nello stato serializzato.
    """
    import pickle

    from src.data.dataset import SpeechCommandsCached

    n, clip = 8, 16_000
    np.lib.format.open_memmap(
        tmp_path / "train_wave.npy", mode="w+", dtype=np.int16, shape=(n, clip)
    ).flush()
    np.save(tmp_path / "train_label.npy", np.zeros(n, dtype=np.int64))

    ds = SpeechCommandsCached(tmp_path, "train")
    _ = ds[0]                      # forza l'apertura della memmap
    assert ds._waves is not None

    blob = pickle.dumps(ds)
    assert len(blob) < 100_000     # non deve contenere l'array

    revived = pickle.loads(blob)
    assert revived._waves is None  # riaperta pigramente nel worker
    assert len(revived) == n
    assert revived[0][0].shape == (clip,)


ADOPTED_DENSE = {
    "channel_reading": "c196_proj96",
    "stem": "paper",
    "pool_stride": (2, 1),
    "seq_mode": "pool_freq",
    "dim": 224,
    "depth": 8,
    "num_heads": 7,
}


def test_adopted_dense_configuration_stays_within_the_declared_budget():
    """Guardia di regressione sulla ricostruzione del modello denso.

    Il paper dichiara 4.80 M parametri e 0.645 G FLOPs. La nostra
    configurazione sta a +2.1% e -11.0%. Se una modifica al codice sposta
    questi numeri, il confronto col paper smette di essere legittimo e
    vogliamo accorgercene subito, non dopo un training.
    """
    from src.flops import analyze
    from src.models.dense import DenseAudioTransformer

    model = DenseAudioTransformer(**ADOPTED_DENSE)
    assert model.seq_len == 51, model.seq_len
    assert (model.feature_shape.channels, model.feature_shape.freq) == (96, 16)

    rep = analyze(model, (1, 64, 101))
    param_err = abs(rep.params - 4.80e6) / 4.80e6
    flop_err = abs(rep.native_flops - 0.645e9) / 0.645e9
    assert param_err < 0.05, f"parametri fuori budget: {rep.params:,}"
    assert flop_err < 0.15, f"FLOPs fuori budget: {rep.native_flops/1e9:.3f} G"


def test_dense_is_batch_independent():
    """Con LayerNorm al posto della BatchNorm, train ed eval devono coincidere.

    Verifica indiretta che nessuna normalizzazione dipendente dal batch sia
    rientrata nel modello.
    """
    from src.models.dense import DenseAudioTransformer

    torch.manual_seed(0)
    model = DenseAudioTransformer(**ADOPTED_DENSE)
    spec = torch.randn(4, 1, 64, 101)
    with torch.no_grad():
        model.train()
        a = model(spec)
        model.eval()
        b = model(spec)
    assert torch.equal(a, b)


def test_seed_everything_is_reproducible():
    seed_everything(1234)
    a = torch.randn(50), np.random.rand(50)
    seed_everything(1234)
    b = torch.randn(50), np.random.rand(50)
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


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
