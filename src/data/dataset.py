"""Dataset e DataLoader su cache memory-mapped.

Restituisce forme d'onda grezze float32 in [-1, 1]; spettrogramma e
augmentation avvengono sul batch in GPU, per tenere il DataLoader fuori dal
percorso critico.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .speech_commands import CLIP_SAMPLES, LABELS

INT16_SCALE = 32_768.0


class SpeechCommandsCached(Dataset):
    """Dataset sulla cache memmap.

    La memmap si apre pigramente e viene rimossa in `__getstate__`: con
    'spawn' (Windows) il Dataset viene picklato per ogni worker, e
    serializzare l'array da 2.7 GB fallirebbe. Ogni worker riapre la propria
    vista sullo stesso file, condividendo le pagine.
    """

    def __init__(self, cache_dir: Path, split: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.labels = np.load(self.cache_dir / f"{split}_label.npy")
        self.num_classes = len(LABELS)
        self._waves: np.ndarray | None = None

    @property
    def waves(self) -> np.ndarray:
        if self._waves is None:
            self._waves = np.load(
                self.cache_dir / f"{self.split}_wave.npy", mmap_mode="r"
            )
            if self._waves.shape[0] != self.labels.shape[0]:
                raise RuntimeError(f"cache incoerente per '{self.split}'")
        return self._waves

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_waves"] = None  # la memmap non attraversa il confine di processo
        return state

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # np.asarray copia dalla memmap: una vista su pagine condivise
        # farebbe fallire pin_memory
        wave = np.asarray(self.waves[idx], dtype=np.float32) / INT16_SCALE
        return torch.from_numpy(wave), int(self.labels[idx])


class EpochShuffleSampler(Sampler[int]):
    """Permutazione che dipende solo da (seed, epoca).

    Sostituisce `shuffle=True`, il cui seed viene estratto dal generatore
    globale: con `persistent_workers` un run ripreso consuma un'estrazione in
    piu' di uno continuo e l'ordine dei dati diverge, anche ripristinando
    tutti gli stati RNG. Stesso schema di `set_epoch` in `DistributedSampler`.
    """

    def __init__(self, num_samples: int, seed: int) -> None:
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        yield from torch.randperm(self.num_samples, generator=generator).tolist()


def make_loader(
    cache_dir: Path,
    split: str,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    seed: int = 0,
) -> DataLoader:
    """DataLoader con i default giusti per train vs eval.

    Sul training usa `EpochShuffleSampler`: chi lo usa deve chiamare
    `loader.sampler.set_epoch(epoch)` a ogni epoca.
    """
    is_train = split == "train"
    do_shuffle = is_train if shuffle is None else shuffle
    dataset = SpeechCommandsCached(cache_dir, split)

    sampler = EpochShuffleSampler(len(dataset), seed) if do_shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else do_shuffle,
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
        wave, _ = ds[0]
        assert wave.shape == (CLIP_SAMPLES,), wave.shape
        assert wave.dtype == torch.float32
        assert -1.0 <= float(wave.min()) and float(wave.max()) <= 1.0
        counts = np.bincount(ds.labels, minlength=len(LABELS))
        print(
            f"[check] {split:<10} n={len(ds):>6}  classi={int((counts > 0).sum()):>2}"
            f"  min/classe={counts.min():>4}  max/classe={counts.max():>4}"
        )
