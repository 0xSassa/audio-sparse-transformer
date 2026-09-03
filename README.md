# Audio Sparse-Transformer — riproduzione

Progetto d'esame per B031278 Deep Learning (Fall 2025, prof. Paolo Frasconi),
MSc in Artificial Intelligence, Università di Firenze.

Riproduzione di:

> H. Salami Kavaki, M. I. Mandel, *Audio Sparse-Transformer for Speech
> Classification*, ICASSP 2025, pp. 1–5. DOI 10.1109/ICASSP49660.2025.10890475

Non esiste codice pubblico per Kavaki & Mandel: modello, training e misure sono
reimplementati dalle formule del paper. Il metodo è la trasposizione all'audio
di SparseFormer (Gao et al., arXiv:2304.03768); augmentation e parametri di
training vengono da EAT (Gazneli et al., arXiv:2204.11479). Dataset: Google
Speech Commands V2, 35 classi, split ufficiali speaker-disjoint.

L'obiettivo è riprodurre il metodo e l'architettura, non i numeri. I parametri
tornano su nove configurazioni dichiarate; i costi e le accuratezze no, e le
sezioni 4 e 6 dicono da cosa dipende.

---

## 1. Il metodo

Un transformer per audio riceve di solito una sequenza di frame temporali, e il
costo dell'attenzione cresce con il quadrato della sua lunghezza. Kavaki &
Mandel sostituiscono quella sequenza con N token latenti, con N molto minore del
numero di frame.

Ogni token `t ∈ R^d` è accoppiato a una regione `b = (x, y, w, h)` del piano
tempo-frequenza. Token e regioni iniziali sono parametri appresi. Il modello
ripete L_rep volte tre passi.

**Aggiustamento della regione.** Il token produce un delta con uno strato
lineare, e la regione si sposta e si ridimensiona con la parametrizzazione di
Faster R-CNN:

```
tx, ty, tw, th = Linear(t)
x' = x + tx·w        w' = w · exp(tw)
y' = y + ty·h        h' = h · exp(th)
```

L'esponenziale garantisce dimensioni positive per costruzione e rende
l'aggiornamento invariante di scala: la rete impara rapporti, non incrementi.

**Campionamento.** Il token genera P offset relativi, normalizzati a tre
deviazioni standard sull'asse dei punti, e i P punti si ottengono traslando il
centro della regione:

```
{(Δxi, Δyi)}_P = Linear(LayerNorm(t))
x̃i = x + 0.5·Δxi·w        ỹi = y + 0.5·Δyi·h
```

I valori nei punti vengono da un'interpolazione bilineare, che rende
differenziabile la selezione: la derivata rispetto alle coordinate è una
differenza finita fra celle adiacenti. È il motivo per cui si campiona
sull'uscita di una convoluzione iniziale e non sullo spettrogramma grezzo, dove
quel gradiente sarebbe troppo rumoroso (paper, §3.2).

**Decoding.** Una rete `F` genera dal token due matrici di pesi, e i P campioni
vengono mescolati prima sui canali e poi sullo spazio:

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
classificazione avviene sulla loro media.

Configurazione dichiarata (Tabella 1): N=4 token, P=36 campioni, d_token=64,
d_encoder=128, L_rep=3, L_enc=8.

---

## 2. L'architettura

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

L'encoder è reimplementato senza `nn.MultiheadAttention`. I blocchi sono
pre-norm: il post-norm di Vaswani et al. richiede warmup del learning rate per
non divergere, col pre-norm il ramo residuo resta pulito.

Il costo di un blocco, per n elementi in dimensione d:

| termine | costo | in n |
|---|---|---|
| proiezioni Q, K, V, O | 4 n d² | lineare |
| punteggi Q Kᵀ | n² d | quadratico |
| aggregazione A V | n² d | quadratico |
| feed-forward (ratio r) | 2 r n d² | lineare |

Il termine quadratico domina solo per n > (2+r)·d, cioè n > 768 con d=128 e r=4.
Le nostre sequenze sono molto più corte, quindi passare da 51 frame a 4 token non
fa risparmiare quanto suggerisce il rapporto (51/4)², e il guadagno va misurato
invece che dedotto.

---

## 3. L'implementazione

### Cosa il paper dichiara e cosa no

Del modello sparso il paper dichiara la configurazione completa (Tabella 1),
l'ottimizzatore, il learning rate, lo schedule, il weight decay, l'EMA, il batch
e le due augmentation. Del modello denso dichiara soltanto parametri e FLOPs.
Dello spettrogramma non dichiara nulla.

Ogni scelta non dichiarata è marcata `[ASSUNZIONE]` in `configs/base.yaml` ed
elencata nella sezione 7.

### Deduzioni, non assunzioni

Quattro valori non sono stati scelti a intuito: due si leggono nell'ancestor del
metodo, due si deducono da vincoli numerici del paper.

`channel_reading = c96`, con un solo strato convolutivo. Il §3.2 di Kavaki &
Mandel dice «96 dimensional feature» e, due righe dopo, «196 kernels»: i due
numeri non possono valere insieme. La contraddizione si scioglie su
SparseFormer, di cui questo metodo è la trasposizione all'audio, che nella
sezione *Model configurations* scrive:

> «We use ResNet-like early convolutional layers (a 7×7 stride-2 convolution, a
> ReLU, and a 3×3 stride-2 max pooling) to extract initial 96-d image features.»

Novantasei canali, uno strato solo, e la sequenza esatta implementata qui. Il
«96 dimensional feature» di Kavaki & Mandel cita quella frase. Il 196 è
plausibilmente contaminazione dai 196 patch in cui ViT divide un'immagine,
numero che SparseFormer richiama nella propria introduzione. Indipendentemente,
due strati convolutivi porterebbero il costo del denso a +164 % contro il +11 %
di uno solo, fuori da qualunque lettura dei numeri dichiarati.

La lettura alternativa, 196 kernel seguiti da una proiezione 1×1 a 96, resta
implementata come `channel_reading: c196_proj96` ed è una buona alternativa:
soddisfa entrambe le frasi del §3.2 al prezzo di uno strato che il paper non
nomina, e sui parametri dichiarati va anche leggermente meglio (−0,8 % sullo
sparso contro −1,6 %). Costa 0,58 punti sul denso: 94,67 % contro 94,09 %, run
`dense_seed0` e `dense_c96_seed0`.

Pesi non condivisi fra le ripetizioni. SparseFormer condivide i pesi del proprio
estrattore fra le ripetizioni, e la sua figura 2 lo dichiara. Qui non sono
condivisi, e a imporlo è il budget del paper riprodotto: con pesi condivisi il
modello avrebbe 2 010 591 parametri contro i 2,87 M dichiarati, cioè −30 %;
senza condivisione 2 822 711, cioè −1,6 %. Solo la seconda lettura sta dentro i
numeri.

`pool_stride = (2, 1)`. Il paper dice «max pooling» senza dare lo stride, e
SparseFormer usa 3×3 stride 2 su entrambi gli assi. Su un'immagine dimezzare
entrambi gli assi è naturale; su una mappa tempo-frequenza dimezzare anche il
tempo porta la sequenza del denso a 26 frame e il suo costo al 45 % di quello
dichiarato, contro l'87 % che si ottiene con (2,1). È l'unico punto in cui ci si
discosta dall'ancestor, ed è dedotto da un vincolo sul modello denso, i cui
iperparametri abbiamo ricostruito noi: è quindi la più fragile delle quattro.

### Dove ci siamo discostati dal paper

`MAX_LOG_SCALE = 4.0` limita il delta logaritmico prima dell'esponenziale. Né il
paper né SparseFormer lo prevedono: le equazioni non pongono vincoli sulle
dimensioni delle regioni. Serve perché `exp()` in float32 va a infinito oltre
~88, e un solo passo anomalo produrrebbe regioni infinite e NaN nel gradiente.

Quanto morde, contato su tutto il validation set di ogni run:

| run | rip. 1 | rip. 2 | rip. 3 | geometria delle regioni |
|---|---|---|---|---|
| `sparse_seed1` | 0 % | 0 % | 0 % | dentro il piano |
| `sparse_grid_seed0` | 0 % | 0 % | 0 % | congelate sulla griglia |
| `sparse_seed0` | 0 % | 46 % | 50 % | uscita dal piano |
| `sparse_seed2` | 0 % | 26 % | 50 % | uscita dal piano |

Il limite è inerte finché le regioni restano nel piano e interviene solo dove
sono già esplose. Non dà forma a un modello sano: impedisce a un modello degenere
di produrre NaN.

L'ultimo `Linear` del decoder opera sul tensore appiattito P·C → d. Il paper non
lo dice. La lettura è stata scelta col conteggio dei parametri fatto a mano prima
di implementare: appiattito dava 2,85 M contro i 2,87 M dichiarati, ridotto sui
punti 2,20 M.

La loss è una BCE su target multi-hot, il ramo `bce` di EAT. Il paper non
dichiara la loss, ma il mixing produce due etichette per campione, quindi la
cross-entropy non è applicabile.

### Due varianti nostre, spente per default

`region_constraint = clip` riporta le regioni dentro `[0,1]²`. Il default è
`none`, la lettura letterale del paper. `region_mode = grid` congela regioni e
aggiustamento senza cambiare forma, parametri né FLOPs, e risponde alla domanda
che il paper non pone: la saliency appresa serve, o basta una griglia fissa?

### Determinismo

Il backward di `grid_sample` non ha implementazione deterministica su CUDA:
accumula con `atomicAdd`, quindi l'ordine delle somme in virgola mobile varia fra
esecuzioni. In modalità stretta PyTorch solleva un errore, quindi il modello
sparso gira con `warn_only`. Il livello ottenuto finisce in `summary.json`
accanto ai risultati, così un run non bit-riproducibile non si spaccia per tale.

---

## 4. La ricostruzione delle quantità non dichiarate

Per calcolare il costo del modello sparso servono due quantità che il paper non
dichiara: la convenzione di conteggio e i parametri dello spettrogramma. La
prima si recupera dai suoi stessi numeri. La seconda no, ed è un risultato anche
quello.

### La convenzione: sono MAC

Il costo si separa in due termini indipendenti:

```
FLOPs(N) = front-end + pendenza × N
```

Il front-end dipende dallo spettrogramma. La pendenza è il costo di un token e
dipende solo da quantità che il paper dichiara: P, C, d_token, d_encoder, L_rep,
L_enc. Non contiene nessuna nostra scelta, quindi se la lettura del metodo è
corretta deve combaciare.

Misurata in sei combinazioni — `n_mels` da 32 a 128, entrambe le letture dei
canali — vale 8,44 MFLOP per token in tutte e sei. Confrontata con quella
implicita nella Tabella 3:

| | pendenza (MFLOP per token) | rapporto |
|---|---|---|
| nostra | 8,44 | |
| paper letto come FLOPs | 3,88 | 2,17× |
| paper letto come MAC | 7,76 | **1,09×** |

I numeri del paper sono MAC. La convenzione FLOPs adottata in origine era stata
dedotta perché i conti tornavano sulla configurazione di default, cioè su un
punto solo.

Il residuo del 9 % non è preoccupante: due contatori standard sullo stesso
modello divergono già dell'1-5 %. In `results/benchmark.json`, prodotto quando
il repository aveva anche fvcore, il rapporto fra i due contatori vale 1,99 sul
denso e 1,95 sullo sparso, non 2 esatto.

### Il front-end: non si recupera

Fissata la convenzione, il termine fisso implicito nei numeri del paper è
78 MFLOP contro i nostri 14,3: un front-end che lo produca dovrebbe elaborare
circa cinque volte le celle tempo-frequenza del nostro.

Un front-end così non sta nelle convenzioni della letteratura. AST calcola «a
25 ms Hamming window every 10 ms», KWT usa «Time window length 30 ms, Time
window stride 10 ms»: su questo dataset l'hop di 10 ms è universale, e su un
secondo di audio dà circa 100 frame. Tenendo l'hop lì e facendo variare i soli
bin mel, il costo dichiarato non si raggiunge mai:

| bin mel, hop 10 ms | 40 | 64 | 80 | 96 | 128 |
|---|---|---|---|---|---|
| scarto dal costo dichiarato | −61 % | −56 % | −52 % | −49 % | −42 % |

Per chiudere servirebbe un hop di 4-5 ms, che nessun lavoro su Speech Commands
usa. Recuperare il front-end richiederebbe quindi di assumere che il paper sia
fuori convenzione, e sarebbe un'assunzione travestita da deduzione.

Il modello denso non entra in questo conto, ed è deliberato: `dim` e `depth` del
denso non sono dichiarati, li abbiamo ricostruiti noi, e usarli per risolvere lo
spettrogramma significherebbe dedurre una quantità del paper da un modello
nostro.

Resta che nemmeno con la convenzione corretta il costo dichiarato del modello
sparso si riproduce da un front-end plausibile. Si somma al §5, dove la loro
tabella dei costi è già incoerente con la loro tabella dei parametri.

Rifare i conti: `python scripts/cost_ablation.py`.

---

## 5. Risultati

Modello EMA. Il test set è stato toccato una volta sola per modello, a
esperimenti chiusi; media ± deviazione standard su 3 seed dove indicato.

| Modello | **Test** (3 seed) | Validation | Params | GFLOP |
|---|---|---|---|---|
| **Denso `flatten`**, baseline adottato | **97,03 % ± 0,20** | 96,97 ± 0,04 | 5 197 795 | 0,5604 |
| **Sparso N=4**, configurazione del paper | **95,52 % ± 0,31** | 95,71 ± 0,37 | 2 822 711 | 0,0485 |
| Sparso N=16 | — | 96,46 % | 2 823 527 | 0,1487 |
| Sparso, regioni congelate su griglia, 3 seed | — | 94,94 % ± 0,22 | 2 822 711 | 0,0485 |
| Sparso, `L_rep`=1 | — | 94,98 % | 2 010 591 | 0,0349 |
| Sparso, regioni vincolate al piano | — | 95,21 % | 2 822 711 | 0,0485 |
| Denso `pool_freq`, ricostruzione scartata | — | 94,09 % | 4 875 235 | 0,5275 |

Tutti i 22 run sono in `results/summary_table.md`, la figura in
`results/accuracy_vs_flops.png`.

Rispetto ai valori dichiarati (Tabella 2):

| | Params | FLOPs | Accuracy |
|---|---|---|---|
| Sparse-transformer | 2,87 M → 2,82 M (−1,6 %) | 0,055 G → 0,0485 (−11,8 %) | 96,88 % → **95,52 %** |
| Dense-transformer | 4,80 M → 5,20 M (+8,3 %) | 0,645 G → 0,5604 (−13,1 %) | 96,67 % → **97,03 %** |

La Tabella 3 si verifica senza addestrare: nove configurazioni dichiarate, cinque
lungo il numero di token N e quattro lungo il numero di campioni P, di cui il
paper dà parametri e costo.

| | entro il 3 % del dichiarato |
|---|---|
| parametri | 9 su 9 |
| FLOPs | 1 su 9 |

I due assi si controllano a vicenda: lungo N il paper dichiara i parametri
costanti a 2,87 M, lungo P li fa più che raddoppiare (2,44 → 5,35 M) per via di
`Ms ∈ R^{P×P}`. Riprodurre per caso entrambi gli andamenti è improbabile, ed è la
verifica più forte che il progetto abbia della propria lettura del metodo.

---

## 6. Perché i nostri numeri differiscono

Il paper dà il modello sparso 0,21 punti sopra il proprio dense-transformer. Noi
lo troviamo 1,52 punti sotto (*t* = 7,10 su 3+3 seed, p = 0,004): il segno del
confronto si rovescia. Cinque fatti misurati lo spiegano.

### 6.1 Il margine del paper è più piccolo del rumore che lo misura

Il modello sparso campiona con `grid_sample`, la cui derivata su CUDA non è
deterministica: due run identici a meno del seed danno risultati diversi.

| deviazione fra seed (validation) | |
|---|---|
| denso | 0,04 punti |
| sparso | 0,37 punti, dieci volte tanto |

Il margine di +0,21 riportato dal paper è misurato su un solo seed: vale 0,6
deviazioni di quel rumore. Per questo qui ogni confronto centrale gira su tre
seed.

### 6.2 Il segno dipende da quanto è forte il baseline denso

Del denso il paper dichiara solo parametri e FLOPs, quindi il baseline va
ricostruito. Resta da decidere come si passa dalla mappa tempo-frequenza alla
sequenza di token, e le due letture possibili sono ugualmente compatibili con i
numeri dichiarati. Le abbiamo addestrate entrambe:

| ricostruzione | cosa fa | validation | lo sparso (95,71 %) risulta |
|---|---|---|---|
| `pool_freq` | media via l'asse frequenza: 96 valori per token | 94,09 % | sopra di 1,62 punti |
| `flatten` | conserva lo spettro: 96 × 16 = 1536 valori per token | 96,97 % | sotto di 1,26 punti |

Con `pool_freq` il denso butta via la frequenza prima di iniziare, e confrontarlo
con un modello che campiona nel piano tempo-frequenza è sbilanciato in partenza.
Abbiamo adottato `flatten`, il baseline più forte, cioè la scelta sfavorevole
alla tesi che stiamo riproducendo.

### 6.3 Il campionamento degenera, e si vede

Niente, nel metodo, tiene le regioni dentro il piano: l'aggiustamento sposta il
centro e moltiplica il lato per un esponenziale senza vincoli. I punti che
finiscono fuori vengono serviti da `padding_mode="border"` e leggono il bordo
della feature map. Continuano a produrre numeri e smettono di guardare il
segnale, senza alcun sintomo nella loss.

Punti che restano dentro il piano, contati su tutto il validation set:

| run | validation | rip. 1 | rip. 2 | rip. 3 |
|---|---|---|---|---|
| `sparse_seed0` | 95,57 % | 98,6 % | 49,9 % | 49,6 % |
| `sparse_seed1` | 96,12 % | 99,3 % | 99,9 % | 97,7 % |
| `sparse_seed2` | 95,43 % | 71,5 % | 49,9 % | 49,9 % |

Su due seed su tre metà del campionamento legge il bordo entro la seconda
ripetizione, e il lato medio delle regioni arriva a 54 volte il piano. Il seed
che non degenera è il migliore dei tre. Su tre run è una correlazione e non una
spiegazione, ma l'ablation su P la rende molto più solida.

### 6.4 Sull'asse P la degenerazione è deterministica

La metà bassa della Tabella 3 non si riproduce, e va nella direzione opposta a
quella dichiarata. Quattro quantità si muovono insieme, in modo monotono:

| P | validation | paper | loss di training | dentro il piano, rip. 3 | scale tagliate |
|---|---|---|---|---|---|
| 16 | 95,28 % | 96,61 % | 0,0765 | 98,9 % | 0 % |
| 36 | 94,77 % | 96,88 % | 0,0814 | 49,7 % | 49 % |
| 64 | 94,62 % | 96,98 % | 0,0857 | 0,0 % | 99,8 % |
| 128 | 93,67 % | 96,95 % | 0,0946 | 0,0 % | 99,9 % |

L'escursione è 1,61 punti contro 0,37 di deviazione fra seed, quindi non è
rumore. Non è nemmeno sovra-adattamento: peggiora anche la loss di training, e un
modello con più del doppio dei parametri fitta peggio i dati. Da P=64 in su la
terza ripetizione dell'estrattore non campiona più il segnale.

L'intera ablation del paper sullo stesso asse copre 0,37 punti, cioè una
deviazione del nostro rumore, su un seed solo.

A P=128 il modello sparso ha 5 323 823 parametri, più del denso adottato
(5 197 795), e resta 3,4 punti sotto di esso.

### 6.5 Il risparmio in FLOPs non diventa velocità

| modello | GFLOP | ms @ batch 1 | ms @ batch 64 |
|---|---|---|---|
| denso `flatten` | 0,5604 | 2,97 | 11,77 |
| sparso | 0,0485 | 4,79 | 6,48 |
| rapporto | 0,087× | 1,61× più lento | 0,55× più veloce |

Un fattore 11,6 sui FLOPs vale 0,62× a batch 1 e 1,82× a batch 64. Il regime in
cui il metodo perde è batch 1, cioè proprio il dispositivo edge che motiva il
lavoro: lì nessuna matrice satura la GPU, il tempo è tutto costo di lancio dei
kernel, e il modello sparso ne lancia di più.

> La latenza dipende dallo stato della macchina, i FLOPs no. Questi tempi sono
> misurati con l'alimentatore collegato; a batteria, con la GPU a una frazione
> del clock di boost, salgono di ~3× e il vantaggio a batch 64 sparisce. Chi
> rigenera `results/benchmark.json` controlli prima `nvidia-smi`.

### Un'ipotesi sul meccanismo, non verificata

I cinque fatti restano cinque, e non abbiamo una causa unica dimostrata. La
candidata è che il campionamento degeneri perché il gradiente sulle coordinate è
troppo rumoroso: quel gradiente è una differenza finita fra celle adiacenti
della feature map, ed è lo stesso che il paper definisce «very noisy» al §3.2,
motivandoci l'esistenza della convoluzione iniziale.

Non abbiamo modo di verificarla. La ricostruzione del §4 mostra che il front-end
del paper non si recupera, quindi non possiamo confrontare la nostra risoluzione
con la loro. E un argomento sulla sola risoluzione sarebbe comunque debole: KWT
raggiunge 97,7 % su questo stesso task con 40 × 98 celle, meno delle nostre
64 × 101. Se sopravvive, l'argomento riguarda il gradiente sulle coordinate, che
è specifico del campionamento e che nessuno degli altri modelli usa perché
nessuno campiona.

L'esperimento che la deciderebbe: addestrare il modello sparso su una mappa più
fine, per esempio `--set features.n_mels=160 --set features.hop_length=80`, e
guardare se la geometria resta dentro il piano. Non direbbe quale front-end
abbiano usato loro, ma direbbe se è la risoluzione a governare il collasso. Non
è stato fatto.

---

## 7. Assunzioni dichiarate

Il paper non specifica quanto segue. Ogni voce è una scelta nostra, marcata
`[ASSUNZIONE]` anche in `configs/base.yaml`.

| Punto | Scelta | Impatto |
|---|---|---|
| Parametri dello spettrogramma, mai dichiarati: mel o lineare, `n_fft`, hop, bin | log-mel, 25 ms / 10 ms, 64 bin → 101 frame | il più alto di tutti, vedi §4 |
| Iperparametri del denso: il paper dà solo parametri e FLOPs | ricostruiti fra 432 combinazioni; adottato `flatten` (§6.2) | decide il segno del confronto |
| Convenzione FLOPs contro MAC | FLOPs; la §4 mostra che i numeri del paper sono MAC | alto sul claim principale |
| Contraddizione nel §3.2: «96 dimensional feature» ma «196 kernels» 7×7 | c96; con c196 il costo dello sparso sbaglierebbe del +73 % | medio |
| Layer della early convolution | uno: il plurale del §3.2 è stato testato e scartato (2 conv → +164 % di FLOPs, misura non più rigenerabile) | medio |
| Numero di epoche | 100; a 50 se ne perdono 0,80 (misurato) | medio |
| Formula di *PhaseMix*, che EAT descrive in una riga | ampiezze mescolate, fasi interpolate sui vettori unitari | basso |
| Codifica posizionale del denso, taciuta dal paper | assente, per simmetria con lo sparso; non testata | basso |

`grid_sample` non è una moltiplicazione di matrici, quindi nessun contatore la
vede: va aggiunta a mano con `grid_sample_flops()`. Sono 432 punti su 96 canali,
0,292 MFLOP, lo 0,60 % del costo del modello sparso. Irrilevante nel numero, ma è
il meccanismo che il paper sta difendendo.

---

## 8. Installazione

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

python scripts/check_env.py            # deve dire "ambiente pronto"
```

`requirements.txt` garantisce che l'installazione riesca; `requirements.lock.txt`
fissa le versioni esatte dei run in `results/`.

---

## 9. Riproduzione

### 9.1 Dati

```bash
python scripts/prepare_data.py --root data/raw --cache data/cache
```

Scarica Speech Commands V2 (~2,3 GB, non incluso nel repository) e costruisce una
cache memory-mapped `int16`. Lo script si ferma con errore se gli split non danno
esattamente 84 843 / 9 981 / 11 005 campioni: sono i conteggi del paper e
coincidono con `validation_list.txt` e `testing_list.txt`, che sono
speaker-disjoint. Uno split casuale gonfia l'accuratezza di punti interi.

### 9.2 Training

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

### 9.3 Verifica dell'archivio

```bash
python scripts/verify_runs.py
```

Non riaddestra nulla: ricostruisce ogni modello dalla configurazione archiviata e
controlla che parametri e FLOPs coincidano con quelli registrati, che
`summary.json` coincida con `metrics.csv`, che gli aggregati tornino dai singoli
seed e che nessuna valutazione di test sia stata forzata. È il modo di stabilire
che i risultati in `results/` vengono da questo codice.

Segnala anche i run prodotti con l'albero git modificato. Il controllo sul
modello passa per tutti e ventidue, quindi quelle modifiche non riguardavano la
definizione dei modelli, ma resta un limite dichiarato.

### 9.4 Valutazione sul test set

```bash
python scripts/evaluate.py --run dense_flatten_seed0
```

Il test set si tocca una volta sola: lo script scrive `TEST_EVALUATED.json` nella
cartella del run e si rifiuta di ripartire, salvo `--force`, che però registra
l'accaduto. Il checkpoint si seleziona sul validation set durante il training, e
si valuta sempre il modello EMA.

### 9.5 Figure e misure

```bash
python scripts/plot_results.py                       # accuracy_vs_flops.png + summary_table.md
python scripts/benchmark_models.py                   # benchmark.json: costo e latenza
python scripts/plot_sampling.py --run sparse_seed0   # sampling_trace.png
```

Due misure che non richiedono training, ma solo di costruire il modello o di
rileggere un run archiviato:

```bash
python scripts/cost_ablation.py                        # cost_ablation.md: la §4
python scripts/region_geometry.py --run sparse_seed0   # region_geometry.json: la §6.3
```

---

## 10. Mappa dei run archiviati

Ogni cartella sotto `results/` è una riga di `results/summary_table.md`; i nomi
sono quelli di esecuzione e non vengono cambiati a posteriori, mentre
`resolved_config.json` riporta la configurazione realmente usata.

| Cartella | Configurazione | Epoche |
|---|---|---|
| `dense_flatten_seed{0,1,2}` | denso `flatten`, c96, baseline adottato | 100 |
| `sparse_seed{0,1,2}` | sparso N=4 P=36, configurazione del paper | 100 |
| `dense_c96_seed0` / `dense_seed0` | denso `pool_freq`, c96 / c196_proj96 | 100 |
| `sparse_grid_seed{0,1,2}` | sparso, regioni congelate sulla griglia | 100 |
| `sparse_rep1_seed0` / `sparse_clip_seed0` | `L_rep`=1 / regioni vincolate al piano | 100 |
| `sparse_N16_ep100_seed0` | sparso N=16 | 100 |
| `sparse_N{4,9,16,25,36}_seed0` | ablation su N | 50 |
| `sparse_P{16,64,128}_seed0` | ablation su P | 50 |

Hanno un `test_report.json` solo `dense_flatten_seed{0,1,2}` e
`sparse_seed{0,1,2}`: gli unici sei run per cui il test set è stato toccato. I run
sparsi con un checkpoint in locale hanno anche un `region_geometry.json`.

---

## 11. Struttura e test

```
configs/    un YAML per esperimento; base.yaml e' ereditato
src/        data/ (split, cache, log-mel, augmentation), models/ (early conv,
            encoder, denso, estrattore sparso), train.py, flops.py, utils.py,
            metrics.py, tracking.py, paper.py (i numeri dichiarati dal paper)
scripts/    check_env, prepare_data, evaluate, run_seeds, verify_runs,
            benchmark_models, plot_results, plot_sampling, cost_ablation,
            region_geometry
tests/      50 test in pochi secondi, girano SENZA il dataset scaricato
results/    metriche, configurazioni e report di ogni run
```

Ogni test difende una decisione o custodisce un numero riportato; nessuno
verifica che PyTorch funzioni. Coprono le invarianti delle due augmentation,
l'equivalenza dell'encoder con `nn.MultiheadAttention`, la convenzione del
contatore di FLOPs, l'EMA, la geometria delle regioni nei suoi tre regimi
(apprese, congelate, vincolate), il raggruppamento dei run e le guardie sugli
override da riga di comando. Quattro sono guardie di regressione sui numeri di
questo README: il costo del denso adottato, il budget dichiarato dello sparso, i
parametri contro tutte e nove le configurazioni della Tabella 3, e l'indipendenza
della pendenza per token dalle assunzioni sullo spettrogramma.

```bash
pytest -q
```

---

## 12. Codice di terze parti e dati

Nessun codice copiato. Il repository ufficiale di EAT
(`github.com/Alibaba-MIIL/AudioClassfication`) è stato consultato solo per
sciogliere ambiguità su augmentation e parametri dell'ottimizzatore. Il codice
ufficiale di SparseFormer (`github.com/showlab/sparseformer`) è stato consultato
per la sequenza della early convolution e per la formula della normalizzazione a
tre deviazioni standard, che il paper nomina senza scriverla. I punti in cui una
delle due fonti ha deciso una scelta sono annotati nel codice e nei config.

Dati non inclusi. Google Speech Commands V2:
`http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
