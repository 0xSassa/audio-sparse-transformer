# Audio Sparse-Transformer — riproduzione

Progetto d'esame per B031278 Deep Learning (Fall 2025), MSc in Artificial
Intelligence, Università di Firenze.

Riproduzione di:

> H. Salami Kavaki, M. I. Mandel. *Audio Sparse-Transformer for Speech
> Classification.* ICASSP 2025, pp. 1–5. DOI 10.1109/ICASSP49660.2025.10890475

Dataset: Google Speech Commands V2, 35 classi, split ufficiali speaker-disjoint.
Il paper non ha codice pubblico, quindi modello, training e misure sono
reimplementati dalle sue formule; dove tace decidono le fonti che cita
(SparseFormer, AdaMixer, EAT) e la letteratura sullo stesso dataset (KWT, AST).

Obiettivo: riprodurre il metodo, non i suoi numeri. I due modelli sono addestrati
da zero su 3 seed, l'ablation è rifatta su entrambi gli assi, e le sezioni 3 e 4
elencano ogni scelta che il paper non dichiara.

---

## 1. Contenuto

Il codice sta in `src/`, un YAML per esperimento in `configs/`, i tre script di
esecuzione in `scripts/`. Dati e pesi non sono inclusi; i file di testo in
`results/` bastano a ricostruire ogni numero di questo documento.

```
configs/   base.yaml è ereditato dagli altri tre
src/       data/    split ufficiali, cache memory-mapped, log-mel, augmentation
           models/  early convolution, encoder, denso, estrattore sparso
           train.py, flops.py, metrics.py, utils.py, tracking.py
scripts/   prepare_data.py, run_seeds.py, evaluate.py
results/   config, metriche per epoca e report dei 22 run archiviati
```

---

## 2. Il metodo

Il transformer riceve N token latenti invece dei frame, con N molto minore del
numero di frame. Ogni token `t ∈ R^d` è accoppiato a una regione `b = (x,y,w,h)`
del piano tempo-frequenza; token e regioni iniziali sono parametri appresi, le
regioni su una griglia `√N × √N`, da cui il vincolo che N sia un quadrato
perfetto e l'ablation su N ∈ {4, 9, 16, 25, 36}. Il modello ripete `L_rep` volte
tre passi:

```
# 1. aggiustamento della regione (parametrizzazione alla Faster R-CNN: exp() dà
#    lati positivi per costruzione e rende l'update invariante di scala)
tx, ty, tw, th = Linear(t)
x' = x + tx·w        w' = w · exp(tw)
y' = y + ty·h        h' = h · exp(th)

# 2. campionamento: P offset relativi, standardizzati sull'asse dei punti e
#    divisi per tre deviazioni (SparseFormer), letti per interpolazione bilineare
{(Δxi, Δyi)}_P = Linear(LayerNorm(t))
x̃i = x + 0.5·Δxi·w        ỹi = y + 0.5·Δyi·h

# 3. decoding adattivo: una rete F con dimensione nascosta d/4 genera dal token
#    due matrici di pesi, e i P campioni si mescolano sui canali e poi sullo spazio
[Mc | Ms] = F(t)        Mc ∈ R^{C×C}   Ms ∈ R^{P×P}
x1 = GELU(x0 · Mc)      x2 = GELU(Ms · x1)      t' = t + Linear(x2)
```

L'interpolazione bilineare rende differenziabile la selezione, con derivata una
differenza finita fra celle adiacenti: per questo si campiona sull'uscita della
convoluzione iniziale e non sullo spettrogramma grezzo (§3.2). I pesi del
decoding sono generati dal token e non fissi: «simply using a linear layer for
this encoding is not effective».

I due modelli condividono tutto tranne come si arriva ai token, così il confronto
è controllato.

```
log-mel                [B, 1, 64, 101]
early convolution      [B, 96, 16, 51]   conv 7×7 s2 → ReLU → maxpool → LayerNorm(C)
  ├─ sparso  estrattore  [B, 4, 64]      4 token da 36 punti campionati
  │          ponte       [B, 4, 128]
  └─ denso   flatten     [B, 51, 1536]   51 frame, ognuno coi suoi 96×16 valori
             proiezione  [B, 51, 224]
8 transformer-encoder pre-norm, nessuna codifica posizionale
media sui token + classificatore lineare  [B, 35]
```

Del modello sparso il paper dichiara la configurazione completa (Tab. 1: N=4,
P=36, d_token=64, d_encoder=128, L_rep=3, L_enc=8) e il training: AdamW
β=(0,9, 0,99), lr 3·10⁻⁴, one-cycle, weight decay 10⁻⁵, EMA 0,995, batch 64,
augmentation *mixing* e *phasemix* di EAT. Nessuna riga di codice è copiata dai
repository di EAT e SparseFormer, consultati per sciogliere ambiguità.

---

## 3. Scelte dedotte dalla letteratura

Sette punti che il paper non dichiara. Nessuno è deciso a occhio: per ognuno
decide una fonte che il paper stesso cita, o la letteratura su questo dataset.

| Punto non dichiarato | Scelta | Fonte |
|---|---|---|
| Canali della early convolution: il §3.2 dice «96 dimensional feature» e due righe dopo «196 kernels» | 96 canali, un solo strato: conv 7×7 s2, ReLU, max pool, LayerNorm sui canali | SparseFormer, *Model configurations*: «ResNet-like early convolutional layers (a 7×7 stride-2 convolution, a ReLU, and a 3×3 stride-2 max pooling) to extract initial 96-d image features». L'altra lettura, 196 kernel più una proiezione 1×1, resta in `channel_reading: c196_proj96` |
| Su cosa opera l'ultimo `Linear` del decoding | sul tensore appiattito, P·C → d | AdaMixer, che il paper cita a fianco dell'eq. (9): «The final output … is flattened and transformed to the d_q dimension by a linear layer to add back to the content vector» |
| Pesi dell'estrattore condivisi fra le `L_rep` ripetizioni | non condivisi | SparseFormer li condivide e lo dichiara; il budget del paper riprodotto no. Condivisi: 2 010 591 parametri contro i 2,87 M dichiarati (−30 %). Non condivisi: 2 822 711 (−1,6 %) |
| Front-end tempo-frequenza | log-mel, finestra 25 ms, hop 10 ms, 101 frame per clip da 1 s | AST: «128-dimensional log Mel filterbank features computed with a 25 ms Hamming window every 10 ms»; KWT: «Time window length 30 ms, Time window stride 10 ms» |
| Tokenizzazione del baseline denso, di cui il paper dà solo parametri e FLOPs | `flatten`: un token per frame con tutte le sue frequenze, 96 × 16 = 1536 valori | KWT, sullo stesso dataset: «the spectrogram is first mapped to a higher dimension d, using a linear projection matrix W₀ ∈ R^{F×d} in the frequency domain», e la sua ablation sulle patch preferisce quelle a frequenza piena. L'alternativa, che media via la frequenza, resta in `configs/dense.yaml` |
| Loss | BCE su target multi-hot, senza label smoothing | EAT, da cui vengono le augmentation: «The loss is label-smoothing … for single-label classification tasks, and binary cross-entropy for the multi-label case. When applying mixing augmentations we use multi-label objective and use binary cross-entropy» |
| Larghezza dell'encoder denso | `dim`=224 | La profondità la dà il paper (8 encoder, e il denso «uses the same architecture»). Su `dim` gli unici vincoli sono i due numeri dichiarati, e nessun valore li soddisfa entrambi: `dim`=240 sbaglia il costo dell'1,2 % e i parametri del 24 %, `dim`=208 fa l'opposto. 224 minimizza il peggiore dei due scarti |

---

## 4. Assunzioni nostre

Sei punti che nessuna fonte fissa. Sono marcati `[ASSUNZIONE]` anche in
`configs/base.yaml`.

| Punto | Scelta |
|---|---|
| Bin mel dello spettrogramma | 64, fra i 40 di KWT e i 128 di AST; danno F=16 dopo lo stem |
| Stride del max pooling | `(2, 1)`, solo sull'asse frequenza: dimezzare anche il tempo lascerebbe 26 frame, cioè un passo di 40 ms. SparseFormer dimezza entrambi gli assi, che su un'immagine sono omogenei |
| Teste dell'attenzione | 4 nello sparso e 7 nel denso, cioè 32 dimensioni per testa in entrambi |
| Numero di epoche | 100 |
| Codifica posizionale del denso | assente, per simmetria con lo sparso |
| Limite sul delta logaritmico prima dell'esponenziale | `MAX_LOG_SCALE = 4,0`, perché `exp()` in float32 esplode oltre ~88. Non è inerte: dalla seconda ripetizione taglia i fattori di scala oltre e⁴ ≈ 55, ed è la deviazione dal paper che pesa di più |

---

## 5. Risultati

Modello EMA, checkpoint selezionato sul validation set. Il test set è stato
valutato una volta sola per modello, a esperimenti chiusi; media ± deviazione
standard su 3 seed.

| Modello | Test (3 seed) | Validation (3 seed) | Params | GFLOP |
|---|---|---|---|---|
| Denso `flatten`, baseline adottato | 97,03 % ± 0,20 | 96,97 ± 0,04 | 5 197 795 | 0,5604 |
| Sparso N=4, configurazione del paper | 95,52 % ± 0,31 | 95,71 ± 0,37 | 2 822 711 | 0,0485 |

Il paper (Tab. 2) dichiara 96,88 % con 2,87 M e 0,055 G per lo sparso, 96,67 %
con 4,80 M e 0,645 G per il denso: lo sparso 0,21 punti sopra il proprio
baseline. Qui sta un punto e mezzo sotto, e il nostro baseline usa la
tokenizzazione di KWT, più forte di quella che il paper lascia intravedere.

I parametri tornano entro il 2 % su tutte e nove le configurazioni dichiarate:
lungo N il paper li dà costanti, lungo P li fa più che raddoppiare per via di
`Ms ∈ R^{P×P}`, e i due andamenti si riproducono entrambi. I costi no, perché il
paper non dichiara né la convenzione di conteggio (FLOPs o MAC) né lo
spettrogramma, da cui dipende il termine fisso.

La metà alta della Tabella 3, l'accuratezza che cresce col numero di token, si
riproduce; la metà bassa lungo P no. Gli otto punti sono a 50 epoche e a un solo
seed; la colonna «Paper» non è alla pari, perché i suoi numeri sono su test e a
budget pieno.

| Config. | Val % | Paper % | Params | Dichiarati | GFLOP |
|---|---|---|---|---|---|
| N=4 (= P=36) | 94,77 | 96,88 | 2 822 711 | 2,87 M | 0,0485 |
| N=9 | 95,31 | 97,13 | 2 823 051 | 2,87 M | 0,0901 |
| N=16 | 96,13 | 97,53 | 2 823 527 | 2,87 M | 0,1487 |
| N=25 | 96,14 | 97,20 | 2 824 139 | 2,87 M | 0,2247 |
| N=36 | 96,08 | 97,94 | 2 824 887 | 2,87 M | 0,3184 |
| P=16 | 95,28 | 96,61 | 2 393 231 | 2,44 M | 0,0383 |
| P=64 | 94,62 | 96,98 | 3 492 527 | 3,53 M | 0,0664 |
| P=128 | 93,67 | 96,95 | 5 323 823 | 5,35 M | 0,1233 |

Cinque varianti che il paper non prevede, spente per default. Congelare le
regioni non cambia forma, parametri né FLOPs, e misura quanto vale la saliency
appresa.

| Variante | Cartella | Val % |
|---|---|---|
| Sparso N=16, a 100 epoche (separa N dal budget) | `sparse_N16_ep100_seed0` | 96,46 |
| Regioni congelate sulla griglia, 3 seed (`region_mode=grid`) | `sparse_grid_seed{0,1,2}` | 94,94 ± 0,22 |
| Una sola ripetizione dell'estrattore (`repeats=1`) | `sparse_rep1_seed0` | 94,98 |
| Regioni vincolate al piano (`region_constraint=clip`) | `sparse_clip_seed0` | 95,21 |
| Denso `pool_freq`, ricostruzione scartata (c96 / c196_proj96) | `dense_c96_seed0` / `dense_seed0` | 94,09 / 94,67 |

Le altre cartelle sono `dense_flatten_seed{0,1,2}` e `sparse_seed{0,1,2}` (i due
run principali, 100 epoche, gli unici sei con `test_report.json`),
`sparse_N{4,9,16,25,36}_seed0` e `sparse_P{16,64,128}_seed0` (ablation, 50
epoche). La configurazione realmente usata sta in `resolved_config.json`.

---

## 6. Riproduzione

Python 3.12 e una GPU NVIDIA. Su Blackwell (sm_120, es. RTX 5050) serve la build
CUDA 12.8: le wheel di default arrivano a sm_90 e falliscono a runtime con
`no kernel image is available for execution on the device`.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
python -m pip install -U pip setuptools wheel

pip install torch==2.9.1+cu128 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128     # 1: stack CUDA
pip install -r requirements.txt                            # 2: il resto
```

`requirements.lock.txt` fissa le versioni esatte dei run archiviati.

```bash
python scripts/prepare_data.py --root data/raw --cache data/cache
```

Scarica Speech Commands V2 (~2,3 GB, non incluso) e costruisce una cache
memory-mapped `int16`. Si ferma se gli split non danno esattamente
84 843 / 9 981 / 11 005 campioni: sono i conteggi del paper e vengono da
`validation_list.txt` e `testing_list.txt`, che Warden costruisce con un hash
dell'id del parlante. Uno split casuale non sarebbe speaker-disjoint.

Un'epoca costa ~60 s sulla RTX 5050 Laptop, cioè ~1,7 ore per un run da 100
epoche.

```bash
# i due run del risultato principale, 3 seed ciascuno
python scripts/run_seeds.py --config configs/dense_flatten.yaml --seeds 0 1 2
python scripts/run_seeds.py --config configs/sparse.yaml        --seeds 0 1 2
```

`run_seeds.py` salta i seed già completati e aggrega media e deviazione in
`results/<prefix>_aggregate.json`. Le altre righe delle tabelle si ottengono con
`--set`, che richiede `--tag`: senza, il run sovrascriverebbe in silenzio la
cartella del run di base.

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

# ablation su P a 50 epoche; il punto P=36 e' il run sparse_N4_seed0
for p in 16 64 128; do
  python -m src.train --config configs/sparse.yaml --seed 0 --epochs 50 \
      --tag sparse_P${p}_seed0 --set model.num_points=$p
done
```

Altre opzioni: `--epochs N`, `--resume` (riprende da `last.pt`), `--stop-after N`
(ferma dopo N epoche lasciando lo schedule configurato per il totale),
`--no-deterministic`. Ogni run scrive in `results/<tag>/` le config archiviate,
`metrics.csv` per epoca, `best.pt` e `last.pt`, `summary.json`,
`validation_report.json` e i log TensorBoard; nel repository sono versionati solo
i file di testo.

I run sono deterministici per default, tranne lo sparso: il backward di
`grid_sample` accumula con `atomicAdd` e su CUDA non ha implementazione
deterministica, quindi gira con `warn_only` e non è bit-riproducibile. Il livello
ottenuto finisce in `summary.json`.

```bash
python scripts/evaluate.py --run dense_flatten_seed0
```

Il test set si tocca una volta sola: lo script scrive `TEST_EVALUATED.json` nella
cartella del run e poi si rifiuta di ripartire, salvo `--force`, che resta
registrato. Si valuta sempre il modello EMA.

---

## 7. Riferimenti

- H. Salami Kavaki, M. I. Mandel. *Audio Sparse-Transformer for Speech
  Classification.* ICASSP 2025. Il paper riprodotto
- Z. Gao, Z. Tong, L. Wang, M. Z. Shou. *SparseFormer: Sparse Visual Recognition
  via Limited Latent Tokens.* arXiv:2304.03768. L'ancestor del metodo
- Z. Gao, L. Wang, B. Han, S. Guo. *AdaMixer: A Fast-Converging Query-Based
  Object Detector.* CVPR 2022. Il decoding adattivo
- S. Ren, K. He, R. Girshick, J. Sun. *Faster R-CNN.* NeurIPS 2015. La
  parametrizzazione delle regioni
- T. Xiao et al. *Early Convolutions Help Transformers See Better.* NeurIPS 2021
- A. Gazneli, G. Zimerman, T. Ridnik, G. Sharir, A. Noy. *End-to-End Audio
  Strikes Back (EAT).* arXiv:2204.11479. Augmentation, loss e ottimizzatore
- A. Berg, M. O'Connor, M. Tairum Cruz. *Keyword Transformer (KWT).*
  arXiv:2104.00769. Tokenizzazione del baseline denso
- Y. Gong, Y.-A. Chung, J. Glass. *AST: Audio Spectrogram Transformer.*
  arXiv:2104.01778
- P. Warden. *Speech Commands: A Dataset for Limited-Vocabulary Speech
  Recognition.* arXiv:1804.03209. Il dataset, non incluso:
  `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
