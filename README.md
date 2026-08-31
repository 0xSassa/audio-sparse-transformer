# Audio Sparse-Transformer — riproduzione

Progetto d'esame per **B031278 Deep Learning** (Fall 2025, prof. Paolo Frasconi),
MSc in Artificial Intelligence, Università di Firenze.

Riproduzione semplificata di:

> H. Salami Kavaki, M. I. Mandel, *Audio Sparse-Transformer for Speech
> Classification*, ICASSP 2025, pp. 1–5. DOI 10.1109/ICASSP49660.2025.10890475

Il metodo è la trasposizione all'audio di **SparseFormer** (Gao et al.,
arXiv:2304.03768); augmentation e parametri di training vengono da **EAT**
(Gazneli et al., arXiv:2204.11479). Non esiste codice pubblico per Kavaki &
Mandel: modello, training e misure sono reimplementati dalle formule del paper.
Dataset: Google Speech Commands V2, 35 classi, split ufficiali speaker-disjoint.

---

## Risultati

Modello EMA. Il test set è stato toccato **una volta sola per modello**, a
esperimenti chiusi; media ± deviazione standard su 3 seed dove indicato.

| Modello | **Test** (3 seed) | Validation | Params | GFLOP |
|---|---|---|---|---|
| **Denso `flatten`** — baseline adottato | **97,03 % ± 0,20** | 96,97 ± 0,04 | 5 197 795 | 0,5604 |
| **Sparso N=4** — configurazione del paper | **95,52 % ± 0,31** | 95,71 ± 0,37 | 2 822 711 | 0,0485 |
| Sparso N=16 | — | 96,46 % | 2 823 527 | 0,1487 |
| Sparso, regioni congelate su griglia — 3 seed | — | 94,94 % ± 0,22 | 2 822 711 | 0,0485 |
| Sparso, `L_rep`=1 | — | 94,98 % | 2 010 591 | 0,0349 |
| Sparso, regioni vincolate al piano | — | 95,21 % | 2 822 711 | 0,0485 |
| Denso `pool_freq` — ricostruzione scartata | — | 94,09 % | 4 875 235 | 0,5275 |

Tutti i 19 run sono in `results/summary_table.md`, la figura in
`results/accuracy_vs_flops.png`; si rigenerano con `python scripts/plot_results.py`.

Rispetto ai valori dichiarati (Tabella 2 del paper):

| | Params | FLOPs | Accuracy |
|---|---|---|---|
| Sparse-transformer | 2,87 M → 2,82 M (−1,6 %) | 0,055 G → 0,0485 (−11,8 %) | 96,88 % → **95,52 %** |
| Dense-transformer | 4,80 M → 5,20 M (+8,3 %) | 0,645 G → 0,5604 (−13,1 %) | 96,67 % → **97,03 %** |

Params e costo tornano: l'architettura è quella giusta. **È l'ordine fra i due
modelli che non si riproduce.**

---

## Perché i nostri numeri differiscono

Il paper dà il modello sparso **0,21 punti sopra** il proprio dense-transformer.
Noi lo troviamo **1,52 punti sotto** (*t* = 7,2 su 3 seed): non uno scarto di
taratura, un cambio di *segno*. Quattro fatti misurati lo spiegano.

### 1. Il margine del paper è più piccolo del rumore che lo misura

Il modello sparso campiona la mappa tempo-frequenza con `grid_sample`, la cui
derivata su CUDA non ha implementazione deterministica: due run identici a meno
del seed danno risultati diversi, e di parecchio.

| Deviazione fra seed (validation) | |
|---|---|
| Denso | 0,04 punti |
| Sparso | **0,37 punti** — dieci volte tanto |

Il margine di +0,21 riportato dal paper è misurato su **un solo seed**: vale
0,6 deviazioni di quel rumore, cioè meno di quanto lo stesso identico
esperimento si sposti da solo cambiando il seme casuale. Per questo qui ogni
confronto centrale gira su tre seed.

### 2. Il segno dipende da quanto è forte il baseline denso

Del dense-transformer il paper dichiara **solo** params e FLOPs, nessun
iperparametro: il baseline va ricostruito. Il punto delicato è come si passa
dalla mappa tempo-frequenza alla sequenza di token, e le due letture possibili
sono ugualmente compatibili con i numeri dichiarati. Le abbiamo addestrate
entrambe:

| Ricostruzione | Cosa fa | Validation | Lo sparso (95,71 %) risulta |
|---|---|---|---|
| `pool_freq` | **media via** l'asse frequenza: 96 valori per token | 94,09 % | **sopra** di 1,62 punti |
| `flatten` | **conserva** lo spettro: 96 × 16 = 1536 valori per token | 96,97 % | **sotto** di 1,26 punti |

Con `pool_freq` il denso butta via la frequenza prima di iniziare, e
confrontarlo con un modello che campiona nel piano tempo-frequenza è sbilanciato
in partenza. Abbiamo quindi adottato `flatten`, il baseline **più forte** — la
scelta sfavorevole alla tesi che stiamo riproducendo. È la ragione principale
per cui il nostro denso supera perfino quello degli autori (97,03 contro
96,67 %), e con esso il modello sparso.

### 3. La prestazione viene dall'architettura, non dalla saliency

L'idea difesa dal paper è che il modello impari *dove guardare*. Congelando le
regioni su una griglia regolare fissa — a parità **esatta** di parametri e
FLOPs, quindi un confronto controllato — il modello perde solo
**0,76 ± 0,25 punti** (95,71 → 94,94 %), cioè metà del divario che lo separa dal
denso. Il contributo del metodo non sta in *dove* guarda, ma nel guardare poco:
4 token invece di 51.

### 4. Il risparmio in FLOPs non diventa velocità

Il paper non dichiara se i suoi numeri siano FLOPs o MAC, che differiscono di un
fattore 2 esatto — quanto basta a invalidare il claim. La convenzione è stata
dedotta dai numeri del paper stesso (tornano in FLOPs) e il contatore è stato
scritto **prima** dei modelli, perché uno scritto dopo tende a confermare il
risultato atteso. Su questo il paper si riproduce: **11,6× di FLOPs in meno**.

In tempo, però:

| Modello | GFLOP | ms @ batch 1 | ms @ batch 64 |
|---|---|---|---|
| Denso `flatten` | 0,5604 | 2,97 | 11,77 |
| Sparso | 0,0485 | **4,79** | **6,48** |
| rapporto | 0,087× | 1,61× più **lento** | 0,55× più veloce |

Un fattore 11,6 sui FLOPs vale 0,62× a batch 1 e 1,82× a batch 64. Il regime in
cui il metodo *perde* è batch 1, cioè proprio il dispositivo edge che motiva il
lavoro: lì nessuna matrice satura la GPU, il tempo è tutto costo di lancio dei
kernel, e il modello sparso ne lancia di più. A batch 64 le matrici del denso
diventano grandi abbastanza da far pesare l'aritmetica, e l'ordine si rovescia.

> **La latenza dipende dallo stato della macchina, i FLOPs no.** Questi tempi
> sono misurati con l'alimentatore collegato; a batteria, con la GPU ferma a una
> frazione del clock di boost, salgono di ~3× e il vantaggio a batch 64
> sparisce. Chi rigenera `results/benchmark.json` controlli prima `nvidia-smi`,
> altrimenti confronta stati di clock e non modelli.

---

## Assunzioni dichiarate

Il paper non specifica quanto segue. Ogni voce è una scelta nostra, marcata
`[ASSUNZIONE]` anche in `configs/base.yaml`.

| Punto | Scelta | Impatto |
|---|---|---|
| **Parametri dello spettrogramma**, mai dichiarati: mel o lineare, `n_fft`, hop, bin | log-mel, 25 ms / 10 ms, 64 bin → 101 frame | **il più alto di tutti** |
| **Iperparametri del denso**: il paper dà solo params e FLOPs | ricostruiti fra 432 combinazioni; adottato `flatten` (§2) | **decide il segno del confronto** |
| **Convenzione FLOPs vs MAC** | FLOPs, dedotta dai numeri del paper | **alto sul claim principale** |
| Contraddizione nel §3.2: «96 dimensional feature» ma «196 kernels» 7×7 | c96; il 196 è un refuso — con c196 il costo dello sparso sbaglierebbe del +73 % | medio |
| Layer della early convolution | uno: il plurale del §3.2 è stato testato e scartato (2 conv → +164 % di FLOPs) | medio |
| Numero di epoche | 100; a 50 se ne perdono 0,80 (misurato) | medio |
| Formula di *PhaseMix*, che EAT descrive in una riga | ampiezze mescolate, fasi interpolate sui vettori unitari | basso |
| Codifica posizionale del denso, taciuta dal paper | assente, per simmetria con lo sparso | basso |

`grid_sample` non è contata dal contatore di FLOPs: va aggiunta a mano con
`grid_sample_flops()`. Vale lo 0,3 % del totale — irrilevante nel numero, ma è
il meccanismo che il paper sta difendendo, e ometterlo sarebbe scorretto.

---

## Installazione

Python 3.12 e una GPU NVIDIA. Su Blackwell (sm_120, es. RTX 5050) la build CUDA
12.8 **non è opzionale**: le wheel di default arrivano a sm_90 e falliscono a
runtime con `no kernel image is available for execution on the device`.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
python -m pip install -U pip setuptools wheel

pip install torch==2.9.1+cu128 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128     # passo 1: stack CUDA
pip install -r requirements.txt                            # passo 2: il resto

python scripts/check_env.py            # deve dire "ambiente pronto"
```

`requirements.txt` garantisce che l'installazione riesca; `requirements.lock.txt`
fissa le versioni **esatte** dei run in `results/`.

---

## Riproduzione

### 1. Dati

```bash
python scripts/prepare_data.py --root data/raw --cache data/cache
```

Scarica Speech Commands V2 (~2,3 GB, non incluso nel repository) e costruisce una
cache memory-mapped `int16`. Lo script **si ferma con errore** se gli split non
danno esattamente 84 843 / 9 981 / 11 005 campioni: sono i conteggi del paper e
coincidono con `validation_list.txt` e `testing_list.txt`, che sono
speaker-disjoint. Uno split casuale gonfia l'accuratezza di punti interi.

### 2. Training

Un'epoca costa ~80 s sulla RTX 5050 Laptop: ~2,2 ore per un run da 100 epoche.

```bash
# i due run del risultato principale, 3 seed ciascuno
python scripts/run_seeds.py --config configs/dense_flatten.yaml --seeds 0 1 2
python scripts/run_seeds.py --config configs/sparse.yaml        --seeds 0 1 2
```

`run_seeds.py` salta i seed già completati e aggrega media e deviazione in
`results/<prefix>_aggregate.json`.

Le altre righe della tabella. `--set` **richiede** `--tag`: senza, il run
scriverebbe nella cartella del run di base e lo sovrascriverebbe in silenzio.

```bash
# denso pool_freq, nelle due letture dei canali
python -m src.train --config configs/dense.yaml --seed 0 --tag dense_c96_seed0
python -m src.train --config configs/dense.yaml --seed 0 --tag dense_seed0 \
    --set model.channel_reading=c196_proj96

# regioni congelate sulla griglia (3 seed)
for s in 0 1 2; do
  python -m src.train --config configs/sparse.yaml --seed $s \
      --tag sparse_grid_seed$s --set model.region_mode=grid
done

# una sola ripetizione dell'estrattore; regioni vincolate al piano
python -m src.train --config configs/sparse.yaml --seed 0 \
    --tag sparse_rep1_seed0 --set model.repeats=1
python -m src.train --config configs/sparse.yaml --seed 0 \
    --tag sparse_clip_seed0 --set model.region_constraint=clip

# ablation su N a 50 epoche, piu' N=16 a 100 per separare N dal budget
for n in 4 9 16 25 36; do
  python -m src.train --config configs/sparse.yaml --seed 0 --epochs 50 \
      --tag sparse_N${n}_seed0 --set model.num_tokens=$n
done
python -m src.train --config configs/sparse.yaml --seed 0 \
    --tag sparse_N16_ep100_seed0 --set model.num_tokens=16
```

Altre opzioni: `--epochs N`, `--resume` (riprende da `last.pt`),
`--stop-after N` (ferma dopo N epoche lasciando lo schedule configurato per il
totale, per spezzare un run lungo su più sessioni), `--no-deterministic`.

Ogni run scrive in `results/<tag>/` le config archiviate, `metrics.csv` per
epoca, `best.pt` e `last.pt`, `summary.json`, `validation_report.json` e i log
TensorBoard. Nel repository sono versionati **solo** i file di testo.

### 3. Valutazione sul test set

```bash
python scripts/evaluate.py --run dense_flatten_seed0
```

Il test set si tocca **una volta sola**: lo script scrive `TEST_EVALUATED.json`
nella cartella del run e si rifiuta di ripartire, salvo `--force`, che però
registra l'accaduto. Il checkpoint si seleziona sul validation set durante il
training, e si valuta sempre il modello **EMA**.

### 4. Figure

```bash
python scripts/plot_results.py                       # accuracy_vs_flops.png + summary_table.md
python scripts/benchmark_models.py                   # benchmark.json: costo e latenza
python scripts/plot_sampling.py --run sparse_seed0   # sampling_trace.png
```

---

## Mappa dei run archiviati

Ogni cartella sotto `results/` è una riga di `results/summary_table.md`; i nomi
sono quelli di esecuzione e non vengono cambiati a posteriori, mentre
`resolved_config.json` riporta la configurazione realmente usata.

| Cartella | Configurazione | Epoche |
|---|---|---|
| `dense_flatten_seed{0,1,2}` | denso `flatten`, c96 — **baseline adottato** | 100 |
| `sparse_seed{0,1,2}` | sparso N=4 P=36 — **configurazione del paper** | 100 |
| `dense_c96_seed0` / `dense_seed0` | denso `pool_freq`, c96 / c196_proj96 | 100 |
| `sparse_grid_seed{0,1,2}` | sparso, regioni congelate sulla griglia | 100 |
| `sparse_rep1_seed0` / `sparse_clip_seed0` | `L_rep`=1 / regioni vincolate al piano | 100 |
| `sparse_N16_ep100_seed0` | sparso N=16 | 100 |
| `sparse_N{4,9,16,25,36}_seed0` | ablation su N | 50 |

Hanno un `test_report.json` solo `dense_flatten_seed{0,1,2}` e
`sparse_seed{0,1,2}`: gli unici sei run per cui il test set è stato toccato.

---

## Struttura e test

```
configs/    un YAML per esperimento; base.yaml e' ereditato
src/        data/ (split, cache, log-mel, augmentation), models/ (early conv,
            encoder, denso, estrattore sparso), train.py, flops.py, utils.py,
            metrics.py, tracking.py
scripts/    check_env, prepare_data, evaluate, run_seeds, benchmark_models,
            plot_results, plot_sampling
tests/      54 test in ~50 s, girano SENZA il dataset scaricato
results/    metriche, configurazioni e report di ogni run
```

I test coprono forme, invarianti delle augmentation, equivalenza dell'encoder
con `nn.MultiheadAttention`, contatore di FLOPs, EMA, riproducibilità dei seed,
geometria delle regioni e raggruppamento dei run:

```bash
pytest -q
```

---

## Codice di terze parti e dati

Nessun codice copiato. Il repository ufficiale di EAT
(`github.com/Alibaba-MIIL/AudioClassfication`) è stato **consultato** solo per
sciogliere ambiguità su augmentation e parametri dell'ottimizzatore; i punti in
cui ha deciso una scelta sono annotati nel codice e nei config.

Dati non inclusi. Google Speech Commands V2:
`http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
