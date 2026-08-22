# Audio Sparse-Transformer — riproduzione

Progetto d'esame per **B031278 Deep Learning** (Fall 2025, prof. Paolo Frasconi),
MSc in Artificial Intelligence, Università di Firenze.

Riproduzione di:

> H. Salami Kavaki, M. I. Mandel, *Audio Sparse-Transformer for Speech
> Classification*, ICASSP 2025, pp. 1–5. DOI 10.1109/ICASSP49660.2025.10890475

con riferimento a:

> A. Gazneli, G. Zimerman, T. Ridnik, G. Sharir, A. Noy, *End-to-End Audio
> Strikes Back*, arXiv:2204.11479, 2022.

Il metodo è la trasposizione all'audio di **SparseFormer** (Gao et al.,
arXiv:2304.03768); la parametrizzazione delle regioni viene da Faster R-CNN,
il decoding adattivo da AdaMixer, il front-end da Xiao et al. 2021.

---

## Stato

| Fase | Contenuto | Stato |
|---|---|---|
| 0 | Lettura paper + ancestor, setup | completata (restano form risorse e ricevimento) |
| 1 | Prerequisiti teorici | in corso, in parallelo |
| 2 | Dati, split ufficiali, contatore FLOPs | **completata** |
| 3 | Baseline: early conv + dense transformer | run di riferimento fatto (94,09 % su validation); restano `flatten` e i seed 1-2 |
| 4 | Sparse feature extractor | **completata (seed 0): 95,57 %, +1,48 sul denso** |
| 5 | Ablation, seed multipli, HPO | da fare |
| 6 | Presentazione | da fare |

### Risultati finora (validation, modello EMA, seed 0)

| Run | Front-end | Accuratezza | Params | GFLOP |
|---|---|---|---|---|
| **`sparse_seed0`** — il metodo del paper | `c96` | **95,57 %** | 2 822 711 | 0,0485 |
| `dense_c96_seed0` — baseline controllato | `c96` | 94,09 % | 4 875 235 | 0,5275 |
| `dense_seed0` — lettura dei canali superata | `c196_proj96` | 94,67 % | 4 899 147 | 0,5742 |

Il modello sparso batte il baseline denso di **1,48 punti** con il 58 % dei
parametri e il 9,2 % dei FLOPs: il claim qualitativo del paper (+0,21 nella
loro Tabella 2) è riprodotto, con un margine più ampio. Il test set non è
ancora stato toccato, e i seed 1–2 mancano.

### Latenza: il guadagno in FLOPs cambia segno col batch

```
python scripts/benchmark_models.py
```

| Modello | GFLOP | ms @ batch 1 | ms @ batch 64 |
|---|---|---|---|
| Denso `c96` | 0,5275 | 2,95 | 11,08 |
| Sparso | 0,0485 | **4,65** | **6,49** |
| rapporto | 0,092× | 1,58× (più lento) | 0,59× (più veloce) |

Un fattore 10,9 sui FLOPs vale 0,63 a batch 1 e 1,71 a batch 64. Il regime
in cui il metodo perde è batch 1 — cioè il dispositivo edge che motiva il
lavoro. Dettagli e spiegazione in `Manuale_tecnico.pdf`, §12.13.

Il piano completo è in `Pipeline_progetto_DL.pdf`.

---

## Installazione

Richiede Python 3.12 e una GPU NVIDIA. Su Blackwell (sm_120, es. RTX 5050)
la build CUDA 12.8 **non è opzionale**: le wheel di default arrivano a sm_90
e falliscono a runtime con `no kernel image is available for execution on
the device`.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
python -m pip install -U pip setuptools wheel

# passo 1 — stack CUDA dall'indice PyTorch
pip install torch==2.9.1+cu128 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# passo 2 — il resto da PyPI
pip install -r requirements.txt

# verifica
python scripts/check_env.py
```

`check_env.py` deve stampare `compute capability : (12, 0)`, `kernel di prova : OK`
e trovare `soundfile`. Se non è così, fermarsi lì.

---

## Riproduzione

### 1. Dati

```bash
python scripts/prepare_data.py --root data/raw --cache data/cache
```

Scarica Speech Commands V2 (~2.3 GB), verifica gli split ufficiali e
costruisce una cache memory-mapped `int16` (~3.4 GB su disco).

Lo script **si ferma con errore** se gli split non danno esattamente:

| Split | Campioni |
|---|---|
| train | 84 843 |
| validation | 9 981 |
| test | 11 005 |

Sono i conteggi dichiarati nel paper e coincidono con gli elenchi ufficiali
`validation_list.txt` / `testing_list.txt`, che sono **speaker-disjoint**.
Uno split casuale gonfia l'accuratezza di punti interi e rende i numeri
non confrontabili.

### 2. Training

```bash
python -m src.train --config configs/dense.yaml --seed 0
```

Opzioni utili: `--epochs N` (sovrascrive la config), `--tag nome` (cartella del
run), `--resume` (riprende da `last.pt`), `--deterministic`.

Ogni run scrive in `results/<tag>/`: `config.yaml` e `resolved_config.json`
archiviati, `metrics.csv` per epoca, `best.pt` e `last.pt`, `summary.json`,
e i log TensorBoard in `tb/`.

Tempi misurati sulla RTX 5050 Laptop: **~80 s per epoca**, quindi circa
**2,2 ore** per un run da 100 epoche.

### 3. Valutazione finale

```bash
python scripts/evaluate.py --run dense_seed0
```

Il test set si tocca **una volta sola**. Lo script scrive
`TEST_EVALUATED.json` nella cartella del run e si rifiuta di ripartire, a
meno di `--force` — che però registra l'accaduto. Il checkpoint si seleziona
sul validation set, durante il training.

Si valuta sempre il **modello EMA**, non i pesi correnti.

---

## Target di riproduzione

Valori riportati da Kavaki & Mandel, Tabella 2, su Speech Commands V2:

| Modello | Params | FLOPs | Accuracy | Nota |
|---|---|---|---|---|
| Sparse-transformer (N=4, P=36) | 2,87 M | 0,055 G | **96,88 %** | obiettivo Fase 4 |
| Dense-transformer | 4,80 M | 0,645 G | 96,67 % | obiettivo Fase 3 |
| EAT-S | 5,30 M | 0,373 G | 98,15 % | *citato*, non rimisurato dagli autori |
| HTS-AT | 28,80 M | 4,066 G | 98,00 % | citato |
| SimPF (MobileNetV2) | 3,40 M | 0,244 G | 95,60 % | citato |

I nostri risultati andranno **accanto** a questi, con media e deviazione
standard su ≥ 3 seed.

---

## Assunzioni dichiarate

Il paper non specifica quanto segue. Ogni voce è una scelta nostra, ed è
marcata `[ASSUNZIONE]` anche in `configs/base.yaml`.

| Punto | Scelta | Impatto |
|---|---|---|
| **Parametri dello spettrogramma** — mai dichiarati: mel o lineare, `n_fft`, hop, numero di bin | log-mel, 25 ms / 10 ms, 64 bin (→ 101 frame) | **il più alto di tutti**: da trattare come iperparametro |
| Contraddizione interna nel paper (§3.2): feature «96 dimensional» ma «196 kernels» 7×7 stride 2 | da testare in entrambe le letture | medio |
| Numero di layer della early convolution | da fissare seguendo Xiao et al. 2021 | medio |
| Numero di epoche | 100, da verificare sulle curve | medio |
| Formula esatta di *PhaseMix* (EAT la descrive in una riga) | ampiezze mescolate, fasi interpolate sui vettori unitari | basso/medio — da confrontare col codice EAT |
| Parametro `alpha` del Beta per il mixing | 0,2 | basso |
| Convenzione FLOPs vs MAC del paper | da calibrare misurando EAT-S con entrambi i contatori | **alto sul claim principale** |

### Sulla misura dei FLOPs

`src/flops.py` usa **due contatori indipendenti**: `torch.utils.flop_counter`
(nativo) e `fvcore`. I due differiscono di ~2× perché fvcore conta MAC e li
chiama "flops". Il paper non dichiara la convenzione: va dedotta misurando
EAT-S e vedendo quale valore si avvicina a 0,373 G.

`grid_sample` (l'interpolazione bilineare del campionamento sparso) non è
contata da nessuno dei due: va aggiunta a mano con `grid_sample_flops()`.

---

## Struttura

```
configs/          YAML, uno per esperimento riportato
src/
  data/
    speech_commands.py   download, split ufficiali, cache memmap
    dataset.py           Dataset e DataLoader
    features.py          log-mel (GPU)
    augment.py           mixing e phasemix (GPU, a livello di batch)
  flops.py               doppio contatore + latenza
  utils.py               seed, config, EMA
scripts/
  check_env.py           verifica ambiente
  prepare_data.py        Fase 2
  build_docs.ps1         rigenera i PDF da docs/ (Edge headless)
tests/
  test_pipeline.py       forme, invarianti, contatore FLOPs (non serve il dataset)
docs/                    sorgenti HTML dei due PDF
results/                 CSV/JSON dei run e notebook delle figure
```

## Documentazione

| File | Contenuto |
|---|---|
| `Manuale_tecnico.pdf` | **il riferimento**: matematica, architettura, ogni scelta implementativa e la sua motivazione |
| `Pipeline_progetto_DL.pdf` | il piano: cosa fare, in che ordine, con quali criteri |
| `Diario_di_bordo.pdf` | il registro: cosa è stato fatto, decisioni, errori, cambi di direzione |

Entrambi si rigenerano dai sorgenti in `docs/`:

```powershell
powershell -File scripts\build_docs.ps1
```

I test girano senza dataset scaricato:

```bash
pytest -q
```

---

## Codice di terze parti

Nessun codice copiato. Il repository ufficiale di EAT
(`github.com/Alibaba-MIIL/AudioClassfication`) è stato **consultato** solo per
sciogliere ambiguità sulle augmentation; ogni uso sarà annotato nel punto
esatto del codice. Non esiste codice pubblico per Kavaki & Mandel: il metodo
è reimplementato dalle formule del paper e da SparseFormer.

## Dati

Non inclusi nel repository. Google Speech Commands V2:
`http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
