Ho provato a riprodurre un paper ICASSP 2025 senza avere accesso al codice originale.

Il paper propone un Audio Sparse-Transformer per la classificazione vocale. Invece di elaborare tutti i frame dello spettrogramma, il modello usa pochi token latenti che imparano quali regioni tempo-frequenza campionare.

Nella mia riproduzione, il modello sparso non supera il baseline denso.

Su Google Speech Commands V2, addestrando entrambi i modelli da zero su tre seed:

• Dense Transformer: 97,03% di accuracy, 5,20 M parametri, 0,5604 GFLOP
• Sparse Transformer: 95,52% di accuracy, 2,82 M parametri, 0,0485 GFLOP

Il modello sparso usa quindi il 45,7% di parametri in meno e richiede 11,6 volte meno FLOP, con una perdita di 1,51 punti percentuali di accuracy.

La parte più impegnativa è stata ricostruire ciò che il paper non specificava: frontend audio, tokenizzazione del baseline, condivisione dei pesi, loss e dettagli del training. Nel repository ho separato le scelte dichiarate dagli autori, quelle dedotte dalla letteratura e le assunzioni rimaste aperte.

Ho incluso anche i risultati che non seguono il paper. L'accuratezza cresce aumentando il numero di token, mentre l'ablation sul numero di punti campionati non riproduce l'andamento pubblicato.

Codice, configurazioni, ablation e risultati:
https://github.com/0xSassa/audio-sparse-transformer

Come valutereste questo compromesso tra accuratezza ed efficienza?

#DeepLearning #AudioAI #Transformers #ReproducibleResearch
