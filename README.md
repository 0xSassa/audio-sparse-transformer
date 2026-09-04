# Audio Sparse-Transformer — riproduzione

Progetto d'esame per B031278 Deep Learning (Fall 2025), MSc in Artificial
Intelligence, Università di Firenze.

Riproduzione di:

> H. Salami Kavaki, M. I. Mandel. *Audio Sparse-Transformer for Speech
> Classification.* ICASSP 2025, pp. 1–5. DOI 10.1109/ICASSP49660.2025.10890475

Il paper non ha codice pubblico: modello, training e misure sono reimplementati
dalle sue formule. Dove il paper tace, la scelta viene dalle fonti che esso
stesso indica — **SparseFormer**, di cui il metodo è la trasposizione all'audio,
ed **EAT**, da cui vengono augmentation e ottimizzatore — e dalla letteratura
sullo stesso dataset (**KWT**, **AST**). Dataset: Google Speech Commands V2,
35 classi, split ufficiali speaker-disjoint.

Obiettivo: riprodurre il **metodo**, non i suoi numeri. Entrambi i modelli del
paper sono implementati e addestrati da zero, l'ablation è rifatta su entrambi
gli assi, e le sezioni 3 e 4 elencano ogni punto in cui la nostra lettura si
discosta dal testo.

---

## 1. Contenuto

```
configs/   un YAML per esperimento; base.yaml è ereditato dagli altri
src/       data/    split ufficiali, cache memory-mapped, log-mel, augmentation
           models/  early convolution, encoder, denso, estrattore sparso
           train.py, flops.py, metrics.py, utils.py, tracking.py
scripts/   prepare_data.py, run_seeds.py, evaluate.py
results/   configurazione, metriche per epoca e report dei 22 run archiviati
```

Dati e pesi non sono inclusi. Di ogni run restano `config.yaml`,
`resolved_config.json`, `metrics.csv`, `summary.json` e i report di valutazione:
bastano a ricostruire ogni numero di questo documento.

---

## 2. Cosa abbiamo fatto

### Il metodo

Kavaki & Mandel sostituiscono la sequenza di frame che un transformer audio
riceve di solito con **N token latenti**, N molto minore del numero di frame.
Ogni token `t ∈ R^d` è accoppiato a una regione `b = (x, y, w, h)` del piano
tempo-frequenza; token e regioni iniziali sono parametri appresi, con le regioni
su una griglia `√N × √N` di lato pari a metà del piano — da cui il vincolo,
ereditato da SparseFormer, che N sia un quadrato perfetto, e la ragione per cui
l'ablation del paper usa N ∈ {4, 9, 16, 25, 36}. Il modello ripete `L_rep` volte
tre passi:

```
# 1. aggiustamento della regione, con la parametrizzazione dei detector alla
#    Faster R-CNN: exp() dà lati positivi per costruzione e rende l'update
#    invariante di scala, cioè la rete impara rapporti e non incrementi
tx, ty, tw, th = Linear(t)
x' = x + tx·w        w' = w · exp(tw)
y' = y + ty·h        h' = h · exp(th)

# 2. campionamento: P offset relativi, standardizzati sull'asse dei punti e
#    divisi per tre deviazioni (SparseFormer), traslati sul centro della
#    regione; i valori si leggono per interpolazione bilineare
{(Δxi, Δyi)}_P = Linear(LayerNorm(t))
x̃i = x + 0.5·Δxi·w        ỹi = y + 0.5·Δyi·h

# 3. decoding adattivo: una rete F con dimensione nascosta d/4 genera dal token
#    due matrici di pesi, e i P campioni si mescolano sui canali e poi sullo spazio
[Mc | Ms] = F(t)        Mc ∈ R^{C×C}   Ms ∈ R^{P×P}
x1 = GELU(x0 · Mc)      x2 = GELU(Ms · x1)      t' = t + Linear(x2)
```

L'interpolazione bilineare rende differenziabile la selezione: la derivata
rispetto alle coordinate è una differenza finita fra celle adiacenti, e per
questo si campiona sull'uscita di una convoluzione iniziale e non sullo
spettrogramma grezzo, dove quel gradiente sarebbe troppo rumoroso (§3.2, che cita
SparseFormer e Xiao et al., *Early convolutions help transformers see better*).
I pesi del decoding sono funzione del token, così ogni token decide come
mescolare i propri campioni: «simply using a linear layer for this encoding is
not effective». Gli N token finali entrano in uno stack di transformer-encoder,
senza codifica posizionale perché sono un insieme e non una sequenza, e si
classifica sulla media delle loro rappresentazioni.

### L'architettura

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

L'encoder è reimplementato senza `nn.MultiheadAttention`, con blocchi pre-norm
come in ViT e in SparseFormer. Il termine quadratico in n (`QKᵀ` e `AV`, 2n²d)
supera quello lineare (4nd² + 2rnd²) solo per n > (2+r)·d, cioè n > 768 con d=128
e r=4: le nostre sequenze sono molto più corte, quindi passare da 51 frame a 4
token non fa risparmiare quanto suggerisce (51/4)², e il guadagno si misura col
contatore di `src/flops.py` invece di dedurlo.

### Cosa è implementato alla lettera

Del modello sparso il paper dichiara la configurazione completa (Tab. 1: N=4,
P=36, d_token=64, d_encoder=128, L_rep=3, L_enc=8) e il training: AdamW con
β=(0,9, 0,99), lr 3·10⁻⁴, schedule one-cycle, weight decay 10⁻⁵, EMA 0,995,
batch 64 e le augmentation *mixing* e *phasemix*, entrambe di EAT — che pubblica
*phasemix* per esteso come Algoritmo 2, rimappaggio `λy = 0,5·λ + 0,5` incluso.
Nessuna riga di codice è copiata: i repository ufficiali di EAT e SparseFormer
sono stati consultati per sciogliere ambiguità, e ogni punto in cui una fonte ha
deciso una scelta è annotato nel codice e nei config.

---

## 3. Dove il paper tace: decide la letteratura che esso cita

| Punto non dichiarato | Scelta | Fonte |
|---|---|---|
| Canali della early convolution: il §3.2 dice «96 dimensional feature» e due righe dopo «196 kernels» | 96 canali, un solo strato: conv 7×7 s2 → ReLU → max pool → LayerNorm sui canali | SparseFormer, *Model configurations*: «ResNet-like early convolutional layers (a 7×7 stride-2 convolution, a ReLU, and a 3×3 stride-2 max pooling) to extract initial 96-d image features». La lettura alternativa — 196 kernel più una proiezione 1×1, che salva entrambe le frasi al prezzo di uno strato mai nominato — resta implementata come `channel_reading: c196_proj96` |
| Su cosa opera l'ultimo `Linear` del decoding | sul tensore appiattito, P·C → d | AdaMixer, che il paper cita proprio a fianco dell'eq. (9): «The final output … is flattened and transformed to the d_q dimension by a linear layer to add back to the content vector» |
| Condivisione dei pesi dell'estrattore fra le `L_rep` ripetizioni | pesi **non** condivisi | SparseFormer li condivide e lo dichiara, ma qui decide il budget del paper riprodotto: condividendoli il modello avrebbe 2 010 591 parametri contro i 2,87 M dichiarati (−30 %), senza condivisione 2 822 711 (−1,6 %). Non a caso 2 010 591 è anche il conto del modello a una sola ripetizione (sez. 5) |
| Front-end tempo-frequenza: il paper non ne dichiara nulla | log-mel, finestra 25 ms, hop 10 ms → 101 frame per clip da 1 s | AST: «128-dimensional log Mel filterbank features computed with a 25 ms Hamming window every 10 ms»; KWT: «Time window length 30 ms, Time window stride 10 ms». Su questo dataset l'hop di 10 ms è la convenzione |
| Tokenizzazione del baseline denso: il paper ne dichiara solo parametri e FLOPs | `flatten`: un token per frame, con tutte le sue frequenze (96 × 16 = 1536 valori) | KWT, sullo stesso dataset: «the spectrogram is first mapped to a higher dimension d, using a linear projection matrix W₀ ∈ R^{F×d} in the frequency domain», e la sua ablation sulle patch trova migliori proprio quelle a frequenza piena, una per frame. La lettura alternativa, che media via la frequenza, resta in `configs/dense.yaml` |
| Loss | BCE su target multi-hot, senza label smoothing | EAT, da cui vengono le augmentation: «The loss is label-smoothing … for single-label classification tasks, and binary cross-entropy for the multi-label case. When applying mixing augmentations we use multi-label objective and use binary cross-entropy». La stessa frase spiega perché qui non c'è label smoothing |
| Larghezza dell'encoder denso | `dim`=224 | La profondità la dà il paper (8 encoder, e il denso «uses the same architecture»), quindi resta solo `dim`, con i due soli numeri dichiarati come vincoli. Nessun valore li soddisfa entrambi: `dim`=240 azzera quasi lo scarto sul costo (−1,2 %) ma sbaglia i parametri del +24 %, `dim`=208 fa l'opposto. Si adotta il valore che minimizza il peggiore dei due scarti |

---

## 4. Assunzioni: scelte nostre, che nessuna fonte fissa

Ogni voce è marcata `[ASSUNZIONE]` anche in `configs/base.yaml`.

| Punto | Scelta |
|---|---|
| Bin mel dello spettrogramma | 64, fra i 40 di KWT e i 128 di AST; danno F=16 dopo lo stem |
| Stride del max pooling, che il paper non dà | `(2, 1)`: solo sull'asse frequenza |
| Teste dell'attenzione, non dichiarate | 4 nello sparso e 7 nel denso, cioè 32 dimensioni per testa in entrambi |
| Numero di epoche | 100 |
| Codifica posizionale del denso, taciuta dal paper | assente, per simmetria con lo sparso |
| Limite sul delta logaritmico prima dell'esponenziale | `MAX_LOG_SCALE = 4,0` |

Lo **stride del pooling** è l'unico punto in cui ci si discosta dall'ancestor.
SparseFormer dimezza entrambi gli assi, naturale su un'immagine, dove le due
dimensioni sono omogenee e dimezzarle insieme conserva le proporzioni. Una mappa
tempo-frequenza non ha quella simmetria: il tempo è l'asse lungo cui il modello
denso costruisce la propria sequenza, e su clip di un secondo dimezzarlo
lascerebbe 26 frame, cioè un passo di 40 ms.

Il **limite sul delta logaritmico** evita che un passo anomalo mandi `exp()` in
overflow (float32 esplode oltre ~88), con regioni infinite e NaN nel gradiente.
Né il paper né SparseFormer lo prevedono, e non è inerte: sulla prima ripetizione
non interviene mai, dalla seconda in poi taglia i fattori di scala richiesti
oltre e⁴ ≈ 55. È la deviazione dal paper che pesa di più fra quelle dichiarate
qui.

---

## 5. Risultati

Modello EMA, checkpoint selezionato sul validation set durante il training. Il
test set è stato toccato una volta sola per modello, a esperimenti chiusi; media
± deviazione standard su 3 seed.

| Modello | Test (3 seed) | Validation (3 seed) | Params | GFLOP |
|---|---|---|---|---|
| Denso `flatten`, baseline adottato | **97,03 % ± 0,20** | 96,97 ± 0,04 | 5 197 795 | 0,5604 |
| Sparso N=4, configurazione del paper | **95,52 % ± 0,31** | 95,71 ± 0,37 | 2 822 711 | 0,0485 |

Dichiarati dal paper (Tab. 2): sparso 96,88 % con 2,87 M e 0,055 G; denso
96,67 % con 4,80 M e 0,645 G. **Il segno del confronto si rovescia**: il paper dà
il modello sparso 0,21 punti sopra il proprio denso, qui lo troviamo un punto e
mezzo sotto. Il nostro baseline denso è ricostruito con la tokenizzazione di KWT,
quindi è più forte di quello che il paper lascia intravedere.

### Ablation

La metà alta della Tabella 3 — l'accuratezza che cresce col numero di token — si
riproduce; la metà bassa, lungo il numero di campioni per token, no. Gli otto
punti sono a 50 epoche e a un solo seed, per essere confrontabili fra loro; la
colonna «Paper» non è alla pari, perché i suoi numeri sono su test e a budget
pieno.

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

**I parametri tornano entro il 2 % su tutte e nove le configurazioni dichiarate**
(N=4 e P=36 sono lo stesso punto), e i due assi si controllano a vicenda: lungo N
il paper li dà costanti, lungo P li fa più che raddoppiare, per via di
`Ms ∈ R^{P×P}`. Riprodurre per caso entrambi gli andamenti è improbabile, ed è la
verifica più forte che il progetto abbia della propria lettura del metodo; si
rifà costruendo i nove modelli con
`src.models.sparse_model.SparseAudioTransformer` e sommandone i parametri, senza
addestrare nulla. I costi no, e la ragione è che il paper non dichiara né la
convenzione di conteggio (FLOPs o MAC) né lo spettrogramma da cui dipende il
termine fisso: senza quel dato il front-end non è ricostruibile, e recuperarlo
sarebbe un'assunzione travestita da deduzione.

### Varianti che il paper non prevede, spente per default

| Variante | Cartella | Val % |
|---|---|---|
| Sparso N=16, a 100 epoche (separa N dal budget) | `sparse_N16_ep100_seed0` | 96,46 |
| Regioni congelate sulla griglia, 3 seed (`region_mode=grid`) | `sparse_grid_seed{0,1,2}` | 94,94 ± 0,22 |
| Una sola ripetizione dell'estrattore (`repeats=1`) | `sparse_rep1_seed0` | 94,98 |
| Regioni vincolate al piano (`region_constraint=clip`) | `sparse_clip_seed0` | 95,21 |
| Denso `pool_freq`, ricostruzione scartata (c96 / c196_proj96) | `dense_c96_seed0` / `dense_seed0` | 94,09 / 94,67 |

Congelare le regioni non cambia forma, parametri né FLOPs, e chiede se la
saliency appresa serva davvero. Le altre cartelle sono `dense_flatten_seed{0,1,2}` e `sparse_seed{0,1,2}` (i due
run principali, 100 epoche, gli unici sei con `test_report.json`),
`sparse_N{4,9,16,25,36}_seed0` e `sparse_P{16,64,128}_seed0` (ablation, 50
epoche). I nomi sono quelli di esecuzione e non vengono cambiati a posteriori: la
configurazione realmente usata sta in `resolved_config.json`.

---

## 6. Riproduzione

### Installazione

Python 3.12 e una GPU NVIDIA. Su Blackwell (sm_120, es. RTX 5050) la build CUDA
12.8 non è opzionale: le wheel di default arrivano a sm_90 e falliscono a runtime
con `no kernel image is available for execution on the device`.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
python -m pip install -U pip setuptools wheel

pip install torch==2.9.1+cu128 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128     # 1: stack CUDA
pip install -r requirements.txt                            # 2: il resto

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

L'ultimo comando deve stampare `True` e il nome della GPU. `requirements.lock.txt`
fissa le versioni esatte dei run archiviati in `results/`.

### Dati

```bash
python scripts/prepare_data.py --root data/raw --cache data/cache
```

Scarica Speech Commands V2 (~2,3 GB, non incluso) e costruisce una cache
memory-mapped `int16`. Lo script si ferma se gli split non danno esattamente
84 843 / 9 981 / 11 005 campioni: sono i conteggi del paper e vengono da
`validation_list.txt` e `testing_list.txt`, che Warden definisce con un hash
dell'id del parlante. Sono quindi speaker-disjoint, mentre uno split casuale
misurerebbe anche quanto il modello riconosce le voci invece delle parole.

### Training

Un'epoca costa ~60 s sulla RTX 5050 Laptop: ~1,7 ore per un run da 100 epoche.

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

I run sono deterministici per default. Il backward di `grid_sample` non ha
implementazione deterministica su CUDA — accumula con `atomicAdd`, quindi
l'ordine delle somme in virgola mobile varia fra esecuzioni — perciò il modello
sparso gira con `warn_only` e non è bit-riproducibile. Il livello effettivamente
ottenuto finisce in `summary.json` accanto ai risultati.

### Valutazione sul test set

```bash
python scripts/evaluate.py --run dense_flatten_seed0
```

Il test set si tocca una volta sola: lo script scrive `TEST_EVALUATED.json` nella
cartella del run e poi si rifiuta di ripartire, salvo `--force`, che però resta
registrato. Si valuta sempre il modello EMA, sul checkpoint selezionato in
validation.

---

## 7. Riferimenti

- H. Salami Kavaki, M. I. Mandel. *Audio Sparse-Transformer for Speech
  Classification.* ICASSP 2025 — il paper riprodotto
- Z. Gao, Z. Tong, L. Wang, M. Z. Shou. *SparseFormer: Sparse Visual Recognition
  via Limited Latent Tokens.* arXiv:2304.03768 — l'ancestor del metodo
- Z. Gao, L. Wang, B. Han, S. Guo. *AdaMixer: A Fast-Converging Query-Based
  Object Detector.* CVPR 2022 — il decoding adattivo
- S. Ren, K. He, R. Girshick, J. Sun. *Faster R-CNN.* NeurIPS 2015 — la
  parametrizzazione delle regioni
- T. Xiao et al. *Early Convolutions Help Transformers See Better.* NeurIPS 2021
- A. Gazneli, G. Zimerman, T. Ridnik, G. Sharir, A. Noy. *End-to-End Audio
  Strikes Back (EAT).* arXiv:2204.11479 — augmentation, loss e ottimizzatore
- A. Berg, M. O'Connor, M. Tairum Cruz. *Keyword Transformer (KWT).*
  arXiv:2104.00769 — tokenizzazione del baseline denso
- Y. Gong, Y.-A. Chung, J. Glass. *AST: Audio Spectrogram Transformer.*
  arXiv:2104.01778
- P. Warden. *Speech Commands: A Dataset for Limited-Vocabulary Speech
  Recognition.* arXiv:1804.03209 — il dataset, non incluso:
  `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
