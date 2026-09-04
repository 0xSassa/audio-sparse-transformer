"""Scarica Speech Commands V2, verifica gli split, costruisce la cache.

    python scripts/prepare_data.py --root data/raw --cache data/cache

Lo script si ferma se gli split non danno esattamente 84_843 / 9_981 / 11_005.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import sanity_check
from src.data.speech_commands import (
    build_cache,
    build_splits,
    download_and_extract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="prosegui anche se i conteggi degli split non tornano (sconsigliato)",
    )
    args = parser.parse_args()

    if args.skip_download:
        data_dir = args.root / "speech_commands_v0.02"
        if not data_dir.exists():
            print(f"[errore] {data_dir} non esiste: togli --skip-download")
            return 1
    else:
        data_dir = download_and_extract(args.root)

    splits = build_splits(data_dir, strict=not args.allow_mismatch)

    print("\n[cache] costruzione (~3.4 GB su disco, int16)")
    build_cache(splits, args.cache)

    print()
    sanity_check(args.cache)
    print("\nDati pronti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
