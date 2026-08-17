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

from .speech_commands import CLIP_SAMPLES, LABELS

INT16_SCALE = 32_768.0


class SpeechCommandsCached(Dataset):
    """Dataset sulla cache memmap.

    NOTA SUL MULTIPROCESSING (Windows). Il DataLoader con num_workers > 0 usa
    il metodo 'spawn': l'oggetto Dataset viene PICKLATO e inviato a ogni
    worker. Se la memmap fosse un attributo dell'istanza, pickle proverebbe a
    serializzare l'intero array da 2.7 GB e fallirebbe con
    `OSError: [Errno 22] Invalid argument`.

    Soluzione: la memmap viene aperta pigramente alla prima lettura e
    rimossa dallo stato in `__getstate__`. Ogni worker riapre la propria vista
    sullo stesso file — che e' esattamente il comportamento voluto, perche' il
    sistema operativo condivide le pagine fra i processi.
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
        # np.asarray forza la copia dalla memmap: senza, il tensore
        # resterebbe una vista su pagine condivise e pin_memory fallirebbe.
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
        wave, _ = ds[0]
        assert wave.shape == (CLIP_SAMPLES,), wave.shape
        assert wave.dtype == torch.float32
        assert -1.0 <= float(wave.min()) and float(wave.max()) <= 1.0
        counts = np.bincount(ds.labels, minlength=len(LABELS))
        print(
            f"[check] {split:<10} n={len(ds):>6}  classi={int((counts > 0).sum()):>2}"
            f"  min/classe={counts.min():>4}  max/classe={counts.max():>4}"
        )
