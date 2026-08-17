"""Verifica dell'ambiente. Eseguire PRIMA di qualsiasi altra cosa.

    python scripts/check_env.py

Controlla i due punti che su questa macchina possono far perdere giornate:
la build CUDA giusta per Blackwell (sm_120) e il backend audio.
"""

from __future__ import annotations

import importlib
import sys

REQUIRED_CAPABILITY = (12, 0)  # RTX 5050 Laptop, Blackwell


def main() -> int:
    ok = True
    print(f"python               : {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        print("torch                : NON INSTALLATO")
        print("\n  pip install torch==2.9.1+cu128 torchaudio==2.9.1 \\")
        print("      --index-url https://download.pytorch.org/whl/cu128")
        return 1

    print(f"torch                : {torch.__version__}")
    print(f"build CUDA           : {torch.version.cuda}")
    print(f"cuda disponibile     : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("  !! nessuna GPU visibile: il training andrebbe su CPU")
        ok = False
    else:
        cap = torch.cuda.get_device_capability()
        arch = torch.cuda.get_arch_list()
        print(f"device               : {torch.cuda.get_device_name(0)}")
        print(f"compute capability   : {cap}")
        print(f"arch compilate       : {arch}")
        if cap >= REQUIRED_CAPABILITY and f"sm_{cap[0]}{cap[1]}" not in arch:
            print(
                f"  !! la build non contiene sm_{cap[0]}{cap[1]}: al primo kernel "
                "otterrai 'no kernel image is available for execution on the "
                "device'. Reinstalla dall'indice cu128 o successivo."
            )
            ok = False
        else:
            # prova reale: un matmul in GPU
            try:
                a = torch.randn(256, 256, device="cuda")
                (a @ a).sum().item()
                print("kernel di prova      : OK")
            except Exception as exc:  # noqa: BLE001
                print(f"kernel di prova      : FALLITO ({exc})")
                ok = False

    for name, note in [
        ("torchaudio", "trasformate e dataset"),
        ("soundfile", "OBBLIGATORIO: senza, i .wav non si caricano"),
        ("numpy", ""),
        ("yaml", "file di configurazione"),
        ("fvcore", "secondo contatore di FLOPs"),
        ("einops", ""),
        ("optuna", "solo Fase 5"),
    ]:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"{name:<21}: {version}")
        except ImportError:
            level = "MANCANTE" if "OBBLIGATORIO" in note else "assente"
            print(f"{name:<21}: {level}  {note}")
            if "OBBLIGATORIO" in note:
                ok = False

    print()
    print("ambiente pronto" if ok else "ambiente NON pronto: risolvi i punti sopra")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
