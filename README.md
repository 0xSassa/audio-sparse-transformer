# Audio Sparse-Transformer — riproduzione

Progetto d'esame per B031278 Deep Learning (Fall 2025, prof. Paolo Frasconi),
MSc in Artificial Intelligence, Università di Firenze.

Riproduzione di:

> H. Salami Kavaki, M. I. Mandel, *Audio Sparse-Transformer for Speech
> Classification*, ICASSP 2025, pp. 1–5. DOI 10.1109/ICASSP49660.2025.10890475

Non esiste codice pubblico per Kavaki & Mandel: modello, training e misure sono
reimplementati dalle formule del paper. Dove il paper tace, la scelta viene
dalle due fonti che esso stesso indica — SparseFormer, di cui il metodo è la
trasposizione all'audio, ed EAT, da cui vengono augmentation e ottimizzatore —
e dalla letteratura sullo stesso dataset (KWT, AST). Dataset: Google Speech
Commands V2, 35 classi, split ufficiali speaker-disjoint.

L'obiettivo è riprodurre il metodo e l'architettura. I due modelli del paper
sono implementati e addestrati da zero, la sua ablation è rifatta su entrambi
gli assi, e la sezione 6 dichiara dove le nostre scelte si discostano dal testo.

---

## 1. Cosa contiene il repository

```
configs/    un YAML per esperimento; base.yaml è ereditato dagli altri
src/        data/     split ufficiali, cache memory-mapped, log-mel, augmentation
            models/   early convolution, encoder, modello denso, estrattore sparso
            train.py, flops.py, metrics.py, utils.py, tracking.py
            paper.py  i numeri dichiarati da Kavaki & Mandel, in un posto solo
scripts/    preparazione dei dati, training multi-seed, valutazione sul test set
results/    metriche, configurazioni e report dei 22 run archiviati
```

I dati non sono inclusi (la consegna prevede il codice) e i pesi nemmeno. Di
ogni run sono versionati configurazione, `metrics.csv` per epoca, `summary.json`
e i report di valutazione: bastano a ricostruire ogni numero di questo documento.

---

## 2. Il metodo

Un transformer per audio riceve di solito una sequenza di frame temporali, e il
costo dell'attenzione cresce con il quadrato della sua lunghezza. Kavaki &
Mandel sostituiscono quella sequenza con N token latenti, con N molto minore del
numero di frame.

Ogni token `t ∈ R^d` è accoppiato a una regione `b = (x, y, w, h)` del piano
tempo-frequenza. Token e regioni iniziali sono parametri appresi: le regioni
partono da una griglia `√N × √N` con lato pari a metà del piano — da cui il
vincolo, ereditato da SparseFormer, che N sia un quadrato perfetto, e la ragione
per cui l'ablation del paper usa N ∈ {4, 9, 16, 25, 36}. Il modello ripete
`L_rep` volte tre passi.

**Aggiustamento della regione.** Il token produce un delta con uno strato
lineare, e la regione si sposta e si ridimensiona con la parametrizzazione dei
detector alla Faster R-CNN:

```
tx, ty, tw, th = Linear(t)
x' = x + tx·w        w' = w · exp(tw)
y' = y + ty·h        h' = h · exp(th)
```

L'esponenziale garantisce dimensioni positive per costruzione e rende
l'aggiornamento invariante di scala: la rete impara rapporti, non incrementi.

**Campionamento.** Il token genera P offset relativi, standardizzati sull'asse
dei punti e divisi per tre deviazioni, e i P punti si ottengono traslando il
centro della regione:

```
{(Δxi, Δyi)}_P = Linear(LayerNorm(t))
x̃i = x + 0.5·Δxi·w        ỹi = y + 0.5·Δyi·h
```

I valori nei punti vengono da un'interpolazione bilineare, che rende
differenziabile la selezione: la derivata rispetto alle coordinate è una
differenza finita fra celle adiacenti. È il motivo per cui si campiona
sull'uscita di una convoluzione iniziale e non sullo spettrogramma grezzo, dove
quel gradiente sarebbe troppo rumoroso — il paper lo dice al §3.2 citando
SparseFormer, che a sua volta rimanda a Xiao et al., *Early convolutions help
transformers see better*.

**Decoding.** Una rete `F` con dimensione nascosta `d/4` genera dal token due
matrici di pesi, e i P campioni vengono mescolati prima sui canali e poi sullo
spazio:

```
[Mc | Ms] = F(t)        Mc ∈ R^{C×C}   Ms ∈ R^{P×P}
x1 = GELU(x0 · Mc)      x2 = GELU(Ms · x1)
t' = t + Linear(x2)
```

I pesi sono funzione del token, quindi ogni token decide come mescolare i propri
campioni invece di subire pesi buoni in media. Il paper è esplicito sul punto:
«simply using a linear layer for this encoding is not effective».

Gli N token finali entrano in uno stack di transformer-encoder, senza codifica
posizionale perché i token sono un insieme e non una sequenza, e la
classificazione avviene sulla media delle loro rappresentazioni finali.

Configurazione dichiarata (Tabella 1): N=4 token, P=36 campioni, d_token=64,
d_encoder=128, L_rep=3, L_enc=8.

---

## 3. L'architettura

I due modelli condividono tutto tranne come si arriva ai token, così il
confronto è controllato.

```
spettrogramma log-mel   [B, 1, 64, 101]
  early convolution     [B, 96, 16, 51]   conv 7×7 s2 → ReLU → maxpool → LayerNorm(C)
      |
      |--- sparso:  estrattore    [B, 4, 64]     4 token da 4×36 punti campionati
      |             ponte lineare [B, 4, 128]
      |
      |--- denso:   flatten       [B, 51, 1536]  51 frame, ognuno coi suoi 96×16 valori
      |             proiezione    [B, 51, 224]
      |
  8 transformer-encoder, nessuna codifica posizionale
  media sui token + classificatore lineare      [B, 35]
```

L'encoder è reimplementato senza `nn.MultiheadAttention`, con blocchi pre-norm
come in ViT e in SparseFormer. Il costo di un blocco, per n elementi in
dimensione d:

| termine | costo | in n |
|---|---|---|
| proiezioni Q, K, V, O | 4 n d² | lineare |
| punteggi Q Kᵀ | n² d | quadratico |
| aggregazione A V | n² d | quadratico |
| feed-forward (ratio r) | 2 r n d² | lineare |

Il termine quadratico domina solo per n > (2+r)·d, cioè n > 768 con d=128 e r=4.
Le nostre sequenze sono molto più corte, quindi passare da 51 frame a 4 token non
fa risparmiare quanto suggerisce il rapporto (51/4)², e il guadagno si misura col
contatore invece di dedurlo.

---

## 4. L'implementazione

### Cosa il paper dichiara

Del modello sparso: la configurazione completa (Tabella 1), l'ottimizzatore
AdamW con β=(0,9, 0,99), il learning rate 3·10⁻⁴, lo schedule one-cycle, il
weight decay 10⁻⁵, l'EMA a 0,995, il batch da 64 e le due augmentation *mixing*
e *phasemix*. Tutto questo è implementato alla lettera; AdamW e one-cycle sono
cablati nel codice, perché non c'è nulla da scegliere.

Del modello denso il paper dichiara soltanto parametri e FLOPs, e dello
spettrogramma non dichiara nulla: sono le due ricostruzioni della sezione 6.

### Le fonti oltre al paper

**SparseFormer** (Gao et al., arXiv:2304.03768) è l'ancestor dichiarato: da lì
vengono la sequenza esatta della early convolution, la formula della
normalizzazione a tre deviazioni, l'inizializzazione delle regioni su griglia e
la LayerNorm sui canali dopo il pooling.

**EAT** (Gazneli et al., arXiv:2204.11479) è la fonte delle augmentation e dei
parametri di training. *PhaseMix* è pubblicato per esteso come Algoritmo 2,
incluso il rimappaggio `λy = 0,5·λ + 0,5`, ed è implementato identico. Da EAT
vengono anche la loss, i parametri di one-cycle e l'esclusione dal weight decay
di ogni parametro monodimensionale (bias e norme).

**KWT** (Berg et al., arXiv:2104.00769) e **AST** (Gong et al., arXiv:2104.01778)
fissano le convenzioni sullo spettrogramma di questo dataset e la tokenizzazione
del baseline denso.

Nessuna riga di codice è copiata: i repository ufficiali di EAT e SparseFormer
sono stati consultati per sciogliere ambiguità, e i punti in cui una delle due
fonti ha deciso una scelta sono annotati nel codice e nei config.

### Determinismo

I run sono deterministici per default. Il backward di `grid_sample` non ha
implementazione deterministica su CUDA — accumula con `atomicAdd`, quindi
l'ordine delle somme in virgola mobile varia fra esecuzioni — perciò il modello
sparso gira con `warn_only`. Il livello ottenuto finisce in `summary.json`
accanto ai risultati, così un run non bit-riproducibile non si spaccia per tale.

---

## 5. Risultati

Modello EMA, selezionato sul validation set durante il training. Il test set è
stato toccato una volta sola per modello, a esperimenti chiusi; media ±
deviazione standard su 3 seed.

| Modello | **Test** (3 seed) | Validation | Params | GFLOP |
|---|---|---|---|---|
| **Denso `flatten`**, baseline adottato | **97,03 % ± 0,20** | 96,97 ± 0,04 | 5 197 795 | 0,5604 |
| **Sparso N=4**, configurazione del paper | **95,52 % ± 0,31** | 95,71 ± 0,37 | 2 822 711 | 0,0485 |

Rispetto ai valori dichiarati (Tabella 2):

| | Params | FLOPs | Accuracy |
|---|---|---|---|
| Sparse-transformer | 2,87 M → 2,82 M (−1,6 %) | 0,055 G → 0,0485 | 96,88 % → **95,52 %** |
| Dense-transformer | 4,80 M → 5,20 M (+8,3 %) | 0,645 G → 0,5604 | 96,67 % → **97,03 %** |

Il paper dà il modello sparso 0,21 punti sopra il proprio dense-transformer; qui
lo troviamo circa un punto e mezzo sotto. Il segno del confronto si rovescia, e
la sezione 6 dice da quali scelte dipende.

### Le ablation

La metà alta della Tabella 3 — l'accuratezza che sale col numero di token — si
riproduce; la metà bassa, lungo il numero di campioni per token, no. Gli otto
punti delle due ablation sono a 50 epoche, per renderli confrontabili fra loro;
le cinque righe successive sono a 100 epoche.

La colonna «Paper» va letta con due cautele: i numeri del paper sono su test e
i nostri su validation, e i nostri sono a metà del budget di epoche. Fra le
nostre configurazioni il confronto è invece alla pari.

| Configurazione | Validation | Paper | Params | GFLOP |
|---|---|---|---|---|
| N=4 (50 ep) | 94,77 % | 96,88 % | 2 822 711 | 0,0485 |
| N=9 | 95,31 % | 97,13 % | 2 823 051 | 0,0901 |
| N=16 | 96,13 % | 97,53 % | 2 823 527 | 0,1487 |
| N=25 | 96,14 % | 97,20 % | 2 824 139 | 0,2247 |
| N=36 | 96,08 % | 97,94 % | 2 824 887 | 0,3184 |
| P=16 (50 ep) | 95,28 % | 96,61 % | 2 393 231 | 0,0383 |
| P=64 | 94,62 % | 96,98 % | 3 492 527 | 0,0664 |
| P=128 | 93,67 % | 96,95 % | 5 323 823 | 0,1233 |
| Sparso N=16, 100 epoche | 96,46 % | — | 2 823 527 | 0,1487 |
| Sparso, regioni congelate su griglia, 3 seed | 94,94 % ± 0,22 | — | 2 822 711 | 0,0485 |
| Sparso, `L_rep`=1 | 94,98 % | — | 2 010 591 | 0,0349 |
| Sparso, regioni vincolate al piano | 95,21 % | — | 2 822 711 | 0,0485 |
| Denso `pool_freq` (c96), ricostruzione scartata | 94,09 % | — | 4 875 235 | 0,5275 |

Regioni congelate, regioni vincolate e `L_rep`=1 sono opzioni che il paper non
prevede, spente per default: `region_mode=grid` congela regioni e aggiustamento
senza cambiare forma, parametri né FLOPs, e chiede se la saliency appresa serva
davvero; `region_constraint=clip` riporta le regioni dentro `[0,1]²`;
`repeats=1` toglie l'iterazione. Il confronto resta controllato in tutti e tre i
casi.

I 22 run sono elencati nella sezione 9; di ciascuno, `results/<tag>/summary.json`
porta accuratezza, parametri e FLOPs, e `metrics.csv` la curva per epoca.

### Il conteggio dei parametri, verificato senza addestrare

Le nove configurazioni della Tabella 3 si controllano costruendo il modello e
contando i pesi, senza addestrare nulla. I due assi si controllano a vicenda:
lungo N il paper dichiara i parametri costanti a 2,87 M, lungo P li fa più che
raddoppiare (2,44 → 5,35 M) per via di `Ms ∈ R^{P×P}`. **Tutte e nove le
configurazioni tornano entro il 2 % del valore dichiarato.** Riprodurre per caso
entrambi gli andamenti è improbabile, ed è la verifica più forte che il progetto
abbia della propria lettura del metodo. Il conteggio si rifà costruendo i nove
modelli con `src.models.sparse_model.SparseAudioTransformer` e sommandone i
parametri; i valori dichiarati dal paper sono trascritti in `src/paper.py`.

---

## 6. Differenze, assunzioni e deduzioni

### Deduzioni: dove il paper tace, decide la letteratura che esso cita

**Early convolution: 96 canali, un solo strato.** Il §3.2 di Kavaki & Mandel
dice «96 dimensional feature» e, due righe dopo, «196 kernels»: i due numeri non
possono valere insieme. La contraddizione si scioglie su SparseFormer, che nella
sezione *Model configurations* scrive: «We use ResNet-like early convolutional
layers (a 7×7 stride-2 convolution, a ReLU, and a 3×3 stride-2 max pooling) to
extract initial 96-d image features». Novantasei canali, uno strato solo, e la
sequenza esatta implementata qui. La lettura alternativa — 196 kernel seguiti da
una proiezione 1×1 a 96, che soddisfa entrambe le frasi al prezzo di uno strato
mai nominato — resta implementata come `channel_reading: c196_proj96`.

**L'ultimo `Linear` del decoder opera sul tensore appiattito P·C → d.** Il paper
non lo dice. Lo dice AdaMixer (Gao et al., CVPR 2022), che Kavaki & Mandel citano
proprio a fianco dell'equazione (9): «The final output … is flattened and
transformed to the d_q dimension by a linear layer to add back to the content
vector».

**Pesi non condivisi fra le ripetizioni.** SparseFormer condivide i pesi del
proprio estrattore fra le ripetizioni e lo dichiara — «the focusing Transformer
in our design is lightweight, and parameters are shared between repeating
stages» — ma qui a decidere è il budget del paper riprodotto: con pesi condivisi
il modello avrebbe 2 010 591 parametri contro i 2,87 M dichiarati, cioè −30 %;
senza condivisione 2 822 711, cioè −1,6 %. Il primo numero è lo stesso del
modello a una sola ripetizione nella tabella delle ablation, e non per caso:
riusare tre volte gli stessi pesi ne costa quanti riusarli una volta.

**Spettrogramma log-mel, finestra 25 ms e hop 10 ms.** Il paper non dichiara
nulla del front-end tempo-frequenza. AST calcola «128-dimensional log Mel
filterbank features computed with a 25 ms Hamming window every 10 ms», KWT usa
«Time window length 30 ms, Time window stride 10 ms» con 40 bin: su questo
dataset l'hop di 10 ms è la convenzione, e su un secondo di audio dà 101 frame.
I 64 bin adottati stanno fra i 40 di KWT e i 128 di AST.

**Baseline denso `flatten`.** Del denso il paper dichiara solo parametri e
FLOPs, quindi la tokenizzazione va decisa. KWT, sullo stesso dataset, proietta
ogni finestra temporale con tutte le sue frequenze — «the spectrogram is first
mapped to a higher dimension d, using a linear projection matrix W₀ ∈ R^{F×d} in
the frequency domain» — e la sua ablation sulle patch trova migliori proprio le
patch a frequenza piena, una per frame. Adottiamo la stessa tokenizzazione: ogni
token porta 96 × 16 = 1536 valori. La lettura alternativa, che media via l'asse
frequenza e lascia 96 valori per token, resta come `configs/dense.yaml` ed è
l'ultima riga della tabella delle ablation.

Restano da fissare larghezza e profondità dell'encoder denso. La profondità la
dà il paper — dichiara 8 encoder e dice che il denso «uses the same
architecture» — quindi si cerca solo `dim`, usando come vincoli i due soli
numeri dichiarati, 4,80 M parametri e 0,645 GFLOP. Nessun valore li soddisfa
entrambi: a `depth`=8, `dim`=240 azzera quasi lo scarto sul costo (−1,2 %) ma
sbaglia i parametri del +24 %, e `dim`=208 fa l'opposto (−6,1 % sui parametri,
−24,2 % sul costo). Si adotta `dim`=224, il valore che rende più piccolo il
peggiore dei due scarti: +8,3 % e −13,1 %. Le teste sono 7, cioè 32 dimensioni
per testa come nell'encoder sparso.

**Loss: BCE su target multi-hot, senza label smoothing.** Kavaki & Mandel non la
dichiarano, ma la prescrive EAT, da cui vengono le augmentation: «The loss is
label-smoothing with a noise parameter set to 0.1 for single-label
classification tasks, and binary cross-entropy for the multi-label case. When
applying mixing augmentations we use multi-label objective and use binary
cross-entropy». La stessa frase spiega perché qui non c'è label smoothing: in
EAT vale per il caso a etichetta singola, non quando si mescola.

### Assunzioni: scelte nostre, che nessuna fonte fissa

Ogni voce è marcata `[ASSUNZIONE]` anche in `configs/base.yaml`.

| Punto | Scelta |
|---|---|
| Bin mel dello spettrogramma | 64, da cui F=16 dopo lo stem |
| Stride del max pooling, che il paper non dà | `(2, 1)`: solo sull'asse frequenza |
| Larghezza dell'encoder denso | `dim`=224, 7 teste, ffn ratio 4; `depth`=8 ereditato dallo sparso |
| Teste dell'encoder sparso, non dichiarate | 4, cioè 32 dimensioni per testa |
| Numero di epoche | 100 |
| Codifica posizionale del denso, taciuta dal paper | assente, per simmetria con lo sparso |
| Limite sul delta logaritmico prima dell'esponenziale | `MAX_LOG_SCALE = 4,0` |

Lo **stride del pooling** merita una riga in più, perché è l'unico punto in cui
ci si discosta dall'ancestor. SparseFormer usa un max pooling 3×3 con stride 2 su
entrambi gli assi, e su un'immagine è la scelta naturale: le due dimensioni sono
omogenee — sono entrambe spazio, misurate in pixel — e dimezzarle insieme
conserva le proporzioni. Una mappa tempo-frequenza non ha quella simmetria: il
tempo è l'asse lungo cui il modello denso costruisce la propria sequenza, e su
clip di un secondo dimezzarlo lascerebbe 26 frame, cioè un passo di 40 ms. Il
pooling si applica quindi al solo asse della frequenza.

Il **limite sul delta logaritmico** serve perché `exp()` in float32 va a infinito
oltre ~88, e un solo passo anomalo produrrebbe regioni infinite e NaN nel
gradiente. Il valore 4 lascia a una regione di moltiplicare il proprio lato per
e⁴ ≈ 55 a ogni ripetizione — molto oltre qualunque regione che stia ancora nel
piano, e molto sotto la soglia di overflow. Né il paper né SparseFormer lo
prevedono, ed è inerte finché le regioni restano dentro il piano.

### Dove i numeri del paper non si ritrovano

I parametri tornano su tutte e nove le configurazioni dichiarate, entro il 2 %.
I costi no, e la ragione è che il paper non dichiara né la convenzione di
conteggio — FLOPs o MAC — né lo spettrogramma da cui dipende il termine fisso del
costo: senza quel dato il front-end non è ricostruibile, e ogni tentativo di
recuperarlo sarebbe un'assunzione travestita da deduzione. L'accuratezza
riprodotta del modello sparso è 95,52 % contro il 96,88 % dichiarato, e il
baseline denso lo supera: quel baseline è ricostruito seguendo la tokenizzazione
di KWT, quindi è più forte di quello che il paper lascia intravedere.

---

## 7. Installazione

Python 3.12 e una GPU NVIDIA. Su Blackwell (sm_120, es. RTX 5050) la build CUDA
12.8 non è opzionale: le wheel di default arrivano a sm_90 e falliscono a runtime
con `no kernel image is available for execution on the device`.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
python -m pip install -U pip setuptools wheel

pip install torch==2.9.1+cu128 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128     # passo 1: stack CUDA
pip install -r requirements.txt                            # passo 2: il resto

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

L'ultimo comando deve stampare `True` e il nome della GPU: se stampa `False`, la
build CUDA è quella sbagliata. `requirements.txt` garantisce che l'installazione
riesca; `requirements.lock.txt` fissa le versioni esatte dei run in `results/`.

---

## 8. Riproduzione

### 8.1 Dati

```bash
python scripts/prepare_data.py --root data/raw --cache data/cache
```

Scarica Speech Commands V2 (~2,3 GB, non incluso nel repository) e costruisce una
cache memory-mapped `int16`. Lo script si ferma con errore se gli split non danno
esattamente 84 843 / 9 981 / 11 005 campioni: sono i conteggi del paper e
coincidono con `validation_list.txt` e `testing_list.txt`, che sono
speaker-disjoint per costruzione: Warden li definisce con un hash dell'id del
parlante, così le registrazioni di una stessa persona non si dividono fra i tre
insiemi. Uno split casuale le dividerebbe, e misurerebbe anche quanto il modello
riconosce le voci invece delle parole.

### 8.2 Training

Un'epoca costa ~60 s sulla RTX 5050 Laptop: ~1,7 ore per un run da 100 epoche.

```bash
# i due run del risultato principale, 3 seed ciascuno
python scripts/run_seeds.py --config configs/dense_flatten.yaml --seeds 0 1 2
python scripts/run_seeds.py --config configs/sparse.yaml        --seeds 0 1 2
```

`run_seeds.py` salta i seed già completati e aggrega media e deviazione in
`results/<prefix>_aggregate.json`.

Le altre righe della tabella. `--set` richiede `--tag`: senza, il run scriverebbe
nella cartella del run di base e lo sovrascriverebbe in silenzio.

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
(ferma dopo N epoche lasciando lo schedule configurato per il totale, per
spezzare un run lungo su più sessioni), `--no-deterministic`.

Ogni run scrive in `results/<tag>/` le config archiviate, `metrics.csv` per
epoca, `best.pt` e `last.pt`, `summary.json`, `validation_report.json` e i log
TensorBoard. Nel repository sono versionati solo i file di testo.

### 8.3 Valutazione sul test set

```bash
python scripts/evaluate.py --run dense_flatten_seed0
```

Il test set si tocca una volta sola: lo script scrive `TEST_EVALUATED.json` nella
cartella del run e si rifiuta di ripartire, salvo `--force`, che però registra
l'accaduto. Il checkpoint si seleziona sul validation set durante il training, e
si valuta sempre il modello EMA.

---

## 9. Mappa dei run archiviati

I nomi delle cartelle sono quelli di esecuzione e non vengono cambiati a
posteriori, mentre `resolved_config.json` riporta la configurazione realmente
usata.

| Cartella | Configurazione | Epoche |
|---|---|---|
| `dense_flatten_seed{0,1,2}` | denso `flatten`, c96, baseline adottato | 100 |
| `sparse_seed{0,1,2}` | sparso N=4 P=36, configurazione del paper | 100 |
| `dense_c96_seed0` / `dense_seed0` | denso `pool_freq`: la ricostruzione scartata, nelle due letture dei canali (c96 / c196_proj96) | 100 |
| `sparse_grid_seed{0,1,2}` | sparso, regioni congelate sulla griglia | 100 |
| `sparse_rep1_seed0` / `sparse_clip_seed0` | `L_rep`=1 / regioni vincolate al piano | 100 |
| `sparse_N16_ep100_seed0` | sparso N=16 | 100 |
| `sparse_N{4,9,16,25,36}_seed0` | ablation su N | 50 |
| `sparse_P{16,64,128}_seed0` | ablation su P | 50 |

Hanno un `test_report.json` solo `dense_flatten_seed{0,1,2}` e
`sparse_seed{0,1,2}`: gli unici sei run per cui il test set è stato toccato.

---

## 10. Riferimenti

- H. Salami Kavaki, M. I. Mandel. *Audio Sparse-Transformer for Speech
  Classification.* ICASSP 2025. — il paper riprodotto
- Z. Gao, Z. Tong, L. Wang, M. Z. Shou. *SparseFormer: Sparse Visual Recognition
  via Limited Latent Tokens.* arXiv:2304.03768 — l'ancestor del metodo
- Z. Gao, L. Wang, B. Han, S. Guo. *AdaMixer: A Fast-Converging Query-Based
  Object Detector.* CVPR 2022 — il decoding adattivo
- S. Ren, K. He, R. Girshick, J. Sun. *Faster R-CNN.* NeurIPS 2015 — la
  parametrizzazione delle regioni
- T. Xiao et al. *Early Convolutions Help Transformers See Better.* NeurIPS 2021
- A. Gazneli, G. Zimerman, T. Ridnik, G. Sharir, A. Noy. *End-to-End Audio
  Strikes Back.* arXiv:2204.11479 — augmentation, loss e ottimizzatore
- A. Berg, M. O'Connor, M. Tairum Cruz. *Keyword Transformer.* arXiv:2104.00769
- Y. Gong, Y.-A. Chung, J. Glass. *AST: Audio Spectrogram Transformer.*
  arXiv:2104.01778
- P. Warden. *Speech Commands: A Dataset for Limited-Vocabulary Speech
  Recognition.* arXiv:1804.03209 — il dataset, non incluso nel repository:
  `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
