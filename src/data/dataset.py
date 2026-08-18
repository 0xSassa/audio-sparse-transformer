"""Dataset e DataLoader su cache memory-mapped.

Il Dataset restituisce forme d'onda grezze float32 in [-1, 1]. Tutto il
resto — spettrogramma e augmentation — avviene sul batch in GPU: e' la
scelta che tiene il DataLoader fuori dal percorso critico.
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


class EpochShuffleSampler(Sampler[int]):
    """Permutazione che dipende SOLO da (seed, epoca).

    >>> PERCHE' NON `shuffle=True` <<<
    `RandomSampler.__iter__`, quando non riceve un generatore, estrae il
    proprio seed dal generatore GLOBALE della CPU. E
    `_MultiProcessingDataLoaderIter.__init__` ne consuma un'altra per il
    `base_seed` dei worker.

    Con `persistent_workers=True` l'iteratore viene costruito UNA VOLTA e poi
    riusato: un run ininterrotto consuma quindi un'estrazione per epoca,
    mentre un run RIPRESO ricostruisce l'iteratore in un processo nuovo e ne
    consuma una in piu'. Il seed del sampler risulta diverso, e da quel punto
    in poi l'ordine dei dati diverge — anche ripristinando perfettamente tutti
    gli stati RNG.

    Misurato: 0.32 punti di accuratezza di scarto dopo due sole epoche, in
    modalita' deterministica, cioe' con ogni altra fonte di rumore azzerata.

    Qui la permutazione e' una funzione pura di (seed, epoca): immune al
    numero di estrazioni fatte altrove, al confine di processo e all'ordine
    di costruzione degli iteratori. E' lo stesso schema di `set_epoch` in
    `DistributedSampler`.
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

    Sul training usa `EpochShuffleSampler`: l'ordine dipende solo da
    (seed, epoca), quindi resta identico anche riprendendo da checkpoint.
    Chi lo usa deve chiamare `loader.sampler.set_epoch(epoch)` a ogni epoca.
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
