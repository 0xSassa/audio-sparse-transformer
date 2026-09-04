# **REPORT DI VALUTAZIONE CRITICA**
## **Analisi del Progetto di Riproduzione — Audio Sparse-Transformer**

**A cura del Prof. [Nome]**  
*Dipartimento di Informatica, Università di Firenze*  
*Corso: B031278 Deep Learning — MSc in Artificial Intelligence*

---

## **GIUDIZIO COMPLESSIVO: 27/30**

Il progetto presenta una riproduzione **tecnicamente solida e intellettualmente onesta**, con un livello di trasparenza raro per un lavoro studentesco. Tuttavia, soffre di problemi strutturali che ne limitano il rigore scientifico e la riproducibilità formale.

---

## **1. PUNTI DI FORZA (ciò che distingue questo lavoro)**

### **1.1 Onestà intellettuale esemplare**
Il README dichiara esplicitamente:
- **Cosa non è stato fatto** (ottimizzazione iperparametri, ablation su P)
- **Le assunzioni critiche** (10 voci tabulate con impatto stimato)
- **Il fallimento del claim principale**: il modello sparso perde 1,52 punti contro il baseline, invertendo il segno del risultato del paper (+0,21)

Questa trasparenza è **superiore alla media dei paper pubblicati** che ho letto come reviewer per ICASSP.

### **1.2 Infrastruttura di verifica robusta**
- **54 test automatici** che verificano forme, invarianti e FLOPs senza bisogno del dataset
- **Contatore FLOPs scritto PRIMA dei modelli** (pratica corretta per evitare confirmation bias)
- **Tracciamento CSV come fonte di verità**, con TensorBoard come supplemento

### **1.3 Scoperte collaterali significative**
Tre risultati che il paper originale non riporta:
1. **Rumore da seed 10× superiore** nel modello sparso (0,36 vs 0,04)
2. **La saliency appresa vale solo ~0,76 punti** (ablation "grid" mode)
3. **Il guadagno in FLOPs cambia segno con il batch size** (più lento a batch 1, più veloce a batch 64)

Queste osservazioni dimostrano comprensione critica, non mera implementazione.

### **1.4 Gestione della non-deterministicità CUDA**
Il codice rileva e documenta esplicitamente che `grid_sampler_2d_backward_cuda` non è deterministico, rendendo il metodo **intrinsecamente meno riproducibile** del baseline. Questa è un'osservazione che avrebbe dovuto essere nel paper originale.

---

## **2. CRITICITÀ STRUTTURALI (ciò che va corretto prima della consegna)**

### **2.1 Codice morto e funzionalità ablation non rimosse** ⚠️

**Problema grave**: il repository contiene codice per modalità che sono esplicitamente dichiarate come **ablation o varianti non-default**, ma che restano attive nel codice di produzione:

| File | Funzionalità | Stato | Azione richiesta |
|------|-------------|-------|------------------|
| `src/models/sparse.py` | `region_mode="grid"` | Ablation completata | **RIMUOVERE** o spostare in branch separato |
| `src/models/sparse.py` | `region_constraint="clip"` | Variante numerica mai usata nei run principali | **RIMUOVERE** dal default |
| `src/models/sparse.py` | `MAX_LOG_SCALE = 4.0` | Clamp numerico che morde sul 50% delle componenti | **DOCUMENTARE** che il default NON è il metodo del paper |
| `scripts/run_seeds.py` | Multi-seed runner | Usato solo per 2 modelli su 19 run | **OPZIONALE**, ma documentare perché solo alcuni modelli hanno 3 seed |
| `scripts/benchmark_models.py` | Benchmark latenza | Risultato già archiviato in `benchmark.json` | **OPZIONALE** per consegna |
| `scripts/plot_sampling.py` | Visualizzazione Figura 2 | Già generata per `sparse_seed0` | **OPZIONALE** |

**Giudizio**: Un revisore esterno potrebbe eseguire il modello "sparse default" e ottenere risultati **diversi dal paper** a causa del clamp logaritmico e della possibilità di usare `region_constraint="clip"`. Questo è inaccettabile per una riproduzione dichiarata.

### **2.2 Test suite sbilanciata verso infrastruttura non essenziale**

Dei 54 test, almeno **12 test verificano funzionalità ablation** che non sono parte del metodo originale:
- `test_region_mode_freezes_the_right_parameters` (righe 663-705)
- `test_region_constraint_keeps_boxes_inside_the_plane` (righe 707-730)
- `test_region_constraint_is_off_by_default` (righe 732-754)
- `test_region_clamp_does_not_bind_at_initialisation` (righe 566-610)

**Raccomandazione**: Spostare questi test in un file separato `tests/test_ablations.py` o rimuoverli se le funzionalità corrispondenti sono eliminate.

### **2.3 Documentazione eccessiva per uno script di consegna**

Il codice è **sovra-commentato** in modo controproducente:
- `src/models/sparse.py`: 380 righe di codice + ~200 righe di commenti esplicativi
- Commenti che spiegano scelte progettuali alternative (es. righe 41-59 su `RegionMode`)
- Note storiche su errori passati (es. righe 148-150 su misurazioni precedenti)

**Per un esame**: i commenti devono spiegare **cosa fa il codice**, non **perché è stato scritto così**. Le giustificazioni appartengono al README o alla relazione, non al codice sorgente.

### **2.4 Incoerenza nella gestione dei risultati**

| Problema | Evidenza | Rischio |
|----------|----------|---------|
| Solo 2 modelli su 19 hanno valutazione test | `dense_flatten_seed{0,1,2}` e `sparse_seed{0,1,2}` | Non si può calcolare deviazione standard sul test per le ablation |
| Ablation su N eseguite a 50 epoche, non 100 | `sparse_N{4,9,16,25,36}_seed0` | Confronto non controllato: 0,80 punti persi per budget ridotto |
| Un run "denso pool_freq" duplicato | `dense_c96_seed0` vs `dense_seed0` (config diversa) | Confusione nella mappatura run→risultati |

### **2.5 Assenza di script di riproduzione "one-click"**

Non esiste uno script tipo:
```bash
python scripts/reproduce_all.py  # rigenera tutti i 19 run
```

I comandi di riproduzione sono sparsi nel README in forma di snippet shell. Per un esaminatore che vuole verificare i risultati, questo richiede **~35 ore di GPU** e copia manuale di 19 comandi.

---

## **3. ERRORI CONCETTUALI (da correggere assolutamente)**

### **3.1 Il default del modello sparso NON è il metodo del paper**

In `src/models/sparse.py`, righe 136-156:
```python
MAX_LOG_SCALE = 4.0  # Limite sul delta logaritmico

def forward(self, tokens, boxes):
    ...
    log_scale = delta[..., 2:].clamp(-self.MAX_LOG_SCALE, self.MAX_LOG_SCALE)
    size = size * log_scale.exp()
```

**Il commento stesso ammette**: *"Ne' il paper ne' SparseFormer lo prevedono"* e *"il limite morde su circa meta' delle componenti nell'ultima ripetizione"*.

**Conseguenza**: Il tuo modello sparso addestrato **non è il metodo di Kavaki & Mandel**, ma una variante con vincolo di stabilità numerica. Questo deve essere:
1. **Rimosso dal default** se vuoi riprodurre il paper
2. **Esplicitamente dichiarato** come modifica necessaria nel README

Attualmente, la dichiarazione è sepolta in un commento nel codice, non nel README.

### **3.2 Convenzione FLOPs dedotta a posteriori**

Sebbene il contatore sia stato scritto prima dei modelli, la scelta tra FLOPs e MAC è stata fatta **confrontando i numeri con il paper** (README, righe 430-440). Questo è un ragionamento circolare:

> "Il paper dichiara 0,645 GFLOP → il mio contatore dà 0,560 G in FLOPs → assumo che il paper usi FLOPs → il mio numero è 'vicino'"

La deduzione è plausibile, ma **non verificabile** senza accesso al codice originale. Dovrebbe essere marcata come **IPOTESI NON VERIFICABILE**, non come convenzione "dedotta".

---

## **4. COSA RIMUOVERE PRIMA DELLA CONSEGNA**

### **Priorità ALTA (obbligatorio)**

1. **`region_mode="grid"`** da `src/models/sparse.py` e config
   - Spostare in un branch `feature/ablation-grid` se serve per futuri lavori
   - Rimuovere i test associati (`test_region_mode_*`)

2. **`region_constraint="clip"`** dal default
   - Lasciare come opzione deprecata con warning se proprio serve
   - Rimuovere `test_region_constraint_*`

3. **Commenti storici e note di debug** da tutto il codice
   - Esempio: righe 148-150 in `sparse.py` ("Le misure, e perche' per mesi abbiamo creduto il contrario")
   - Esempio: righe 34-38 in `sparse.py` (spiegazione di "none" vs "clip")

4. **Script `run_seeds.py`** se non usato per tutti i modelli
   - Oppure completare l'esecuzione multi-seed per tutte le ablation

### **Priorità MEDIA (fortemente consigliato)**

5. **Test sulle ablation rimosse**
   - Spostare in `tests/test_ablations.py` (file non eseguito di default)

6. **Funzione `benchmark_latency`** se i risultati sono già archiviati
   - Lasciare solo `analyze()` per conteggio FLOPs

7. **Script `plot_sampling.py`** se la figura è già generata
   - Oppure lasciare come "script di visualizzazione opzionale"

### **Priorità BASSA (opzionale)**

8. **Contatore `gmacs`** non usato nei report principali
9. **Supporto per `fvcore`** già rimosso ma menzionato nei commenti
10. **Note su Weights & Biases** rimosso (già pulito nel README)

---

## **5. COSA AGGIUNGERE (per completezza scientifica)**

### **Obbligatorio**

1. **Dichiarazione esplicita nel README**:
   ```markdown
   ## Limiti della riproduzione
   
   Il modello sparso implementato include DUE modifiche rispetto al paper originale:
   
   1. **Clamp logaritmico** (`MAX_LOG_SCALE=4.0`): necessario per stabilità numerica, 
      ma morde sul ~50% delle componenti. Il metodo originale è numericamente instabile.
   
   2. **Nessuna penalità sulle regioni fuori dal piano**: il paper non vincola le 
      regioni, ma misuriamo che ~50% dei punti legge il bordo dopo la 2ª ripetizione.
   
   Queste modifiche rendono il nostro "sparse default" tecnicamente diverso dal 
   metodo di Kavaki & Mandel. I risultati vanno interpretati come una variante 
   stabilizzata, non come riproduzione letterale.
   ```

2. **Tabella di confronto FLOPs dettagliata**:
   Aggiungere a `results/summary_table.md` una colonna "FLOPs teorici del paper" per confronto diretto.

3. **Script di sanity check post-training**:
   ```bash
   python scripts/verify_run.py --run sparse_seed0
   # Verifica: config salvata, metrics.csv completo, best.pt esistente, EMA usato
   ```

### **Consigliato**

4. **Grafico del rumore da seed**:
   Plot di accuratezza vs seed per denso e sparso, con barre di errore.

5. **Ablation mancante su P** (num_points):
   Se c'è tempo, eseguire per P ∈ {16, 25, 36, 49}.

---

## **6. GIUDIZIO PER SEZIONI**

| Sezione | Voto | Commento |
|---------|------|----------|
| **Implementazione tecnica** | 9/10 | Codice pulito, modulare, ben testato |
| **Riproduzione risultati** | 7/10 | Numeri vicini ma default modificato |
| **Analisi critica** | 10/10 | Scoperte collaterali eccellenti |
| **Documentazione** | 8/10 | README ottimo, commenti nel codice eccessivi |
| **Riproducibilità formale** | 6/10 | Script frammentari, 17/19 run senza test evaluation |
| **Onestà intellettuale** | 10/10 | Dichiarazione limiti esemplare |

**Totale: 50/60 → 25/30** (prima delle correzioni)

**Dopo le correzioni indicate**: potenziale **28-29/30**.

---

## **7. RACCOMANDAZIONI FINALI**

### **Prima della consegna (48 ore stimate)**

```bash
# 1. Rimuovere grid mode e clip constraint dal default
git checkout -b feature/cleanup-before-submission

# 2. Eliminare o spostare test ablation
mv tests/test_pipeline.py tests/test_main.py
# Editare test_main.py rimuovendo 12 test su ablation

# 3. Pulire commenti storici da sparse.py
# Mantenere solo docstring che spiegano cosa fa il codice

# 4. Aggiungere sezione "Limiti" al README (vedi punto 5.1)

# 5. Creare script verify_run.py per sanity check

# 6. Eseguire pytest -q per verificare che i test residui passino
```

### **Per la discussione orale**

Preparare 3 slide su:
1. **Perché il modello sparso perde** (analisi gradient noise + campionamento al bordo)
2. **Cosa avresti fatto con 2 settimane in più** (Optuna, ablation su P, ensemble)
3. **Cosa hai imparato sulla riproducibilità** (FLOPs ≠ latenza, seed matters, default non è neutro)

---

## **CONCLUSIONE**

Questo è un **lavoro di master solido**, con un livello di introspezione critica raro. Le criticità identificate sono per lo più **formali e di pulizia**, non concettuali. Con le correzioni indicate, il progetto è **eccellente** e pubblicabile come technical report su arXiv.

**Voto finale atteso dopo correzioni: 28/30**

*Firmato,*  
*Il Professore*
