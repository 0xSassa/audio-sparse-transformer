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
def test_augmentation_preserves_shapes_and_mass(fn):
    """I target restano distribuzioni: e' cio' che rende lecita la BCE."""
    wave = torch.randn(BATCH, CLIP)
    targets = one_hot(torch.randint(NUM_CLASSES, (BATCH,)), NUM_CLASSES)
    mixed, mixed_t = fn(wave, targets)
    assert mixed.shape == wave.shape
    assert mixed_t.shape == targets.shape
    assert torch.isfinite(mixed).all()
    assert torch.allclose(mixed_t.sum(dim=1), torch.ones(BATCH), atol=1e-5)
    assert (mixed_t >= 0).all()


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


def test_apply_augmentations_without_augs_is_identity():
    wave = torch.randn(BATCH, CLIP)
    labels = torch.randint(NUM_CLASSES, (BATCH,))
    out, targets = apply_augmentations(wave, labels, NUM_CLASSES, names=())
    assert torch.equal(out, wave)
    assert torch.allclose(targets, one_hot(labels, NUM_CLASSES))


def test_apply_augmentations_always_returns_soft_targets():
    """Il percorso di loss deve essere unico: sempre [B, C] float."""
    wave = torch.randn(BATCH, CLIP)
    labels = torch.randint(NUM_CLASSES, (BATCH,))
    for _ in range(10):
        out, targets = apply_augmentations(wave, labels, NUM_CLASSES, prob=1.0)
        assert out.shape == wave.shape
        assert targets.shape == (BATCH, NUM_CLASSES)


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
