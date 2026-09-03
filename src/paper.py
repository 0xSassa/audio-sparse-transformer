"""Numeri DICHIARATI da Kavaki & Mandel, ICASSP 2025 — trascritti dal PDF.

Unica fonte per i valori del paper: da qui li leggono l'analisi dei costi e i
test di regressione, e una trascrizione sbagliata si corregge in un posto solo.

Le tuple sono sempre (valore dell'asse, parametri in M, GFLOP, accuratezza %).
L'accuratezza del paper e' su TEST, la nostra su validation: il confronto
diretto fra le due colonne non e' alla pari e va detto ovunque compaia.
"""

from __future__ import annotations

# Tabella 2 — confronto su Google Speech Commands V2.
# EAT-S, HTS-AT e SimPF sono numeri CITATI dal paper, non rimisurati da noi
# ne' da loro.
TABLE2 = [
    # etichetta,               params M, GFLOP, accuratezza %
    ("Sparse-transformer",         2.87, 0.055, 96.88),
    ("Dense-transformer",          4.80, 0.645, 96.67),
    ("EAT-S",                      5.30, 0.373, 98.15),
    ("HTS-AT",                    28.80, 4.066, 98.00),
    ("SimPF (MobileNetV2)",        3.40, 0.244, 95.60),
]

# Tabella 3, meta' alta — ablation sul numero di token latenti N, a P=36.
# I parametri sono dichiarati COSTANTI a 2.87 M su tutta la riga: N non
# introduce pesi, cambia solo quante volte si usano quelli che ci sono.
TOKENS = [
    # N, params M, GFLOP, accuratezza %
    (4,  2.87, 0.05461, 96.88),
    (9,  2.87, 0.07366, 97.13),
    (16, 2.87, 0.101,   97.53),
    (25, 2.87, 0.135,   97.20),
    (36, 2.87, 0.179,   97.94),
]

# Tabella 3, meta' bassa — ablation sul numero di campioni per token P, a N=4.
# Qui i parametri dichiarati CRESCONO, e piu' che quadraticamente in P: e'
# la firma di `Ms` in R^{PxP}, i pesi di decoding spaziale generati dal token.
# La riga P=36 e' la configurazione di default, la stessa di TOKENS[0].
POINTS = [
    # P, params M, GFLOP, accuratezza %
    (16,  2.44, 0.05164, 96.61),
    (36,  2.87, 0.05461, 96.88),
    (64,  3.53, 0.05919, 96.98),
    (128, 5.35, 0.07154, 96.95),
]
