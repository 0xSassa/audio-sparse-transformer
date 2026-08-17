"""
Pipeline dati per Google Speech Commands V2 (Warden, arXiv:1804.03209).

PROTOCOLLO — confermato dal testo di Kavaki & Mandel (ICASSP 2025, sez. 3.1):
    35 classi · 16 kHz · clip zero-paddate a 1 secondo esatto
    train 84_843   validation 9_981   test 11_005

Gli split sono quelli UFFICIALI del dataset (`validation_list.txt` e
`testing_list.txt`), costruiti sull'hash dello speaker ID e quindi
SPEAKER-DISJOINT. Uno split casuale gonfia l'accuratezza di punti interi.

I conteggi attesi sono un test di correttezza gratuito: se non tornano
esattamente, la pipeline e' sbagliata e ogni numero prodotto a valle
e' privo di valore. `build_splits()` fallisce esplicitamente in quel caso.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE_MD5 = "6b74f3901214cb2c2934e98196829835"

SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE  # esattamente 1 secondo

# Le 35 etichette di SC-V2, in ordine alfabetico: l'indice di classe e'
# la posizione in questa lista. Fissarla qui (invece di derivarla dal
# filesystem) rende la label map riproducibile su macchine diverse.
LABELS: tuple[str, ...] = (
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes",
    "zero",
)
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}

EXPECTED_COUNTS = {"train": 84_843, "validation": 9_981, "test": 11_005}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())  # 105_829


@dataclass(frozen=True)
class Splits:
    train: list[Path]
    validation: list[Path]
    test: list[Path]

    def as_dict(self) -> dict[str, list[Path]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download_and_extract(root: Path, *, force: bool = False) -> Path:
    """Scarica ed estrae SC-V2 in `root/speech_commands_v0.02`.

    L'archivio pesa ~2.3 GB. Ritorna la cartella estratta.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "speech_commands_v0.02"
    archive = root / "speech_commands_v0.02.tar.gz"

    if data_dir.exists() and (data_dir / "testing_list.txt").exists() and not force:
        print(f"[data] gia' presente: {data_dir}")
        return data_dir

    if not archive.exists() or force:
        print(f"[data] download da {URL} (~2.3 GB, puo' richiedere parecchio)")
        _download_with_progress(URL, archive)

    print(f"[data] verifica MD5 di {archive.name}")
    actual = _md5(archive)
    if actual != ARCHIVE_MD5:
        raise RuntimeError(
            f"MD5 non corrispondente: atteso {ARCHIVE_MD5}, ottenuto {actual}. "
            "Archivio corrotto o incompleto: cancellalo e riscarica."
        )

    print(f"[data] estrazione in {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)
    return data_dir


def _download_with_progress(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")

    def hook(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block * block_size, total)
        pct = 100.0 * done / total
        print(f"\r[data] {pct:5.1f}%  ({done / 1e9:.2f}/{total / 1e9:.2f} GB)", end="")

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.replace(dest)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Split ufficiali
# --------------------------------------------------------------------------

def build_splits(data_dir: Path, *, strict: bool = True) -> Splits:
    """Costruisce i tre split dagli elenchi ufficiali.

    `strict=True` (default) solleva se i conteggi non sono esattamente
    84_843 / 9_981 / 11_005.
    """
    data_dir = Path(data_dir)
    val_names = _read_list(data_dir / "validation_list.txt")
    test_names = _read_list(data_dir / "testing_list.txt")

    all_files: list[Path] = []
    for label in LABELS:
        label_dir = data_dir / label
        if not label_dir.is_dir():
            raise FileNotFoundError(f"cartella di classe mancante: {label_dir}")
        all_files.extend(sorted(label_dir.glob("*.wav")))

    if strict and len(all_files) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"trovati {len(all_files)} file, attesi {EXPECTED_TOTAL}. "
            "Estrazione incompleta, oppure la cartella _background_noise_ "
            "e' finita fra le classi."
        )

    train, validation, test = [], [], []
    for path in all_files:
        rel = f"{path.parent.name}/{path.name}"
        if rel in val_names:
            validation.append(path)
        elif rel in test_names:
            test.append(path)
        else:
            train.append(path)

    splits = Splits(train=train, validation=validation, test=test)

    for name, files in splits.as_dict().items():
        expected = EXPECTED_COUNTS[name]
        status = "OK" if len(files) == expected else "MISMATCH"
        print(f"[split] {name:<10} {len(files):>6}  (atteso {expected:>6})  {status}")
        if strict and len(files) != expected:
            raise RuntimeError(
                f"split '{name}': {len(files)} file invece di {expected}. "
                "Non proseguire: ogni risultato a valle sarebbe non confrontabile."
            )
    return splits


def _read_list(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"elenco ufficiale mancante: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


# --------------------------------------------------------------------------
# Cache memory-mapped
# --------------------------------------------------------------------------

def build_cache(splits: Splits, out_dir: Path) -> None:
    """Materializza ogni split in un `.npy` int16 di forma [N, 16000].

    Motivo: leggere ~85k file .wav a ogni epoca satura il DataLoader molto
    prima della GPU. Con la cache si legge da memmap (il SO tiene in page
    cache le parti calde) e le augmentation si spostano su GPU.

    Ingombro su disco, int16:
        train       84_843 x 16_000 x 2 B  ~=  2.72 GB
        validation   9_981 x 16_000 x 2 B  ~=  0.32 GB
        test        11_005 x 16_000 x 2 B  ~=  0.35 GB
    """
    import soundfile as sf  # import locale: serve solo qui

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    for name, files in splits.as_dict().items():
        n = len(files)
        wave_path = out_dir / f"{name}_wave.npy"
        label_path = out_dir / f"{name}_label.npy"

        waves = np.lib.format.open_memmap(
            wave_path, mode="w+", dtype=np.int16, shape=(n, CLIP_SAMPLES)
        )
        labels = np.empty(n, dtype=np.int64)

        for i, path in enumerate(files):
            audio, sr = sf.read(path, dtype="int16", always_2d=False)
            if sr != SAMPLE_RATE:
                raise ValueError(f"{path}: sample rate {sr}, atteso {SAMPLE_RATE}")
            if audio.ndim != 1:
                raise ValueError(f"{path}: atteso audio mono, shape {audio.shape}")

            # Zero-padding a destra / troncamento: il paper dichiara
            # esplicitamente il padding a 1 s.
            if audio.shape[0] < CLIP_SAMPLES:
                waves[i, : audio.shape[0]] = audio
                waves[i, audio.shape[0] :] = 0
            else:
                waves[i] = audio[:CLIP_SAMPLES]

            labels[i] = LABEL_TO_INDEX[path.parent.name]

            if (i + 1) % 5_000 == 0 or i + 1 == n:
                print(f"\r[cache] {name}: {i + 1}/{n}", end="")
        print()

        waves.flush()
        del waves
        np.save(label_path, labels)

        counts = np.bincount(labels, minlength=len(LABELS))
        meta[name] = {
            "n": n,
            "wave": wave_path.name,
            "label": label_path.name,
            "per_class_min": int(counts.min()),
            "per_class_max": int(counts.max()),
        }

    meta["labels"] = list(LABELS)
    meta["sample_rate"] = SAMPLE_RATE
    meta["clip_samples"] = CLIP_SAMPLES
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[cache] scritto {out_dir / 'meta.json'}")


def load_cache(cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (waves int16 memmap [N, 16000], labels int64 [N])."""
    cache_dir = Path(cache_dir)
    waves = np.load(cache_dir / f"{split}_wave.npy", mmap_mode="r")
    labels = np.load(cache_dir / f"{split}_label.npy")
    if waves.shape[0] != labels.shape[0]:
        raise RuntimeError(f"cache incoerente per '{split}'")
    return waves, labels
