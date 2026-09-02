"""Riproduzione di Kavaki & Mandel (ICASSP 2025) — Audio Sparse-Transformer.

Eseguito prima di qualunque import di torch: e' l'unico punto in cui
impostare CUBLAS_WORKSPACE_CONFIG abbia effetto. Senza, cuBLAS resta non
deterministico e `torch.use_deterministic_algorithms` non garantisce nulla
su matmul, einsum e Linear.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
