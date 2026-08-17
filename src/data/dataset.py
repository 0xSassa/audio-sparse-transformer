"""Dataset e DataLoader su cache memory-mapped.

Il Dataset restituisce forme d'onda grezze float32 in [-1, 1]. Tutto il
resto — spettrogramma e augmentation — avviene sul batch in GPU: e' la
scelta che tiene il DataLoader fuori dal percorso critico.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .speech_commands import CLIP_SAMPLES, LABELS, load_cache

INT16_SCALE = 32_768.0


class SpeechCommandsCached(Dataset):
    def __init__(self, cache_dir: Path, split: str) -> None:
        self.split = split
        self.waves, self.labels = load_cache(cache_dir, split)
        self.num_classes = len(LABELS)

    def __len__(self) -> int:
        return self.waves.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # np.asarray forza la copia dalla memmap: senza, i worker
        # condividerebbero pagine e il pin_memory fallirebbe.
        wave = np.asarray(self.waves[idx], dtype=np.float32) / INT16_SCALE
        return torch.from_numpy(wave), int(self.labels[idx])


def make_loader(
    cache_dir: Path,
    split: str,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
) -> DataLoader:
    """DataLoader con i default giusti per train vs eval."""
    is_train = split == "train"
    dataset = SpeechCommandsCached(cache_dir, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        drop_last=is_train if drop_last is None else drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def sanity_check(cache_dir: Path) -> None:
    """Verifica rapida di shape, range e bilanciamento delle classi."""
    for split in ("train", "validation", "test"):
        ds = SpeechCommandsCached(cache_dir, split)
        wave, label = ds[0]
        assert wave.shape == (CLIP_SAMPLES,), wave.shape
        assert wave.dtype == torch.float32
        assert -1.0 <= float(wave.min()) and float(wave.max()) <= 1.0
        counts = np.bincount(ds.labels, minlength=len(LABELS))
        print(
            f"[check] {split:<10} n={len(ds):>6}  classi={int((counts > 0).sum()):>2}"
            f"  min/classe={counts.min():>4}  max/classe={counts.max():>4}"
        )
