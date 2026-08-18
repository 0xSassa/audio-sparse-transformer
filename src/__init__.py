"""Riproduzione di Kavaki & Mandel (ICASSP 2025) — Audio Sparse-Transformer.

>>> QUESTO FILE VIENE ESEGUITO PRIMA DI QUALUNQUE IMPORT DI TORCH <<<
ed e' l'unico posto in cui impostare CUBLAS_WORKSPACE_CONFIG abbia effetto.

cuBLAS riusa un'area di memoria di lavoro condivisa fra chiamate. Quando piu'
stream la usano, l'ordine delle riduzioni puo' variare fra esecuzioni: due
somme degli stessi float in ordine diverso differiscono negli ultimi bit.
Senza questa variabile, `torch.use_deterministic_algorithms` non puo'
garantire nulla sulle operazioni cuBLAS — cioe' su matmul, einsum, Linear e
sull'intero backward.

PyTorch lo segnala con un UserWarning per ogni operazione coinvolta, ma
essendo un avviso e non un errore scorre via nei log senza che nessuno lo
noti. Impostandolo qui non puo' essere dimenticato da riga di comando.

:4096:8 alloca 8 workspace da 4 MB: e' il valore raccomandato da NVIDIA per
le prestazioni. L'alternativa :16:8 usa meno memoria ma e' piu' lenta.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
