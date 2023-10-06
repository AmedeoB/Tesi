# Info
Questo branch offre una decomposizione del programma in classi e funzioni.

## Classi
Le classi sono implementate nel file fun_lib.py

### Proxytree
Questa classe sostituisce totalmente la generazione dell'albero e la definizione di tutte le sue costanti

### Proxymanager
Questa classe sostituisce totalmente le costanti di gestione come la scelta di stampa di debug, salvataggio, caricamento, moltiplicatori di tempo per i solver e i moltiplicatori di lagrange

## Funzioni
Le funzioni sostitutive sono implementate nel file models.py, qui vengono eseguiti tutti i cicli di operazione dei vari tipi di solver su entrambi i problemi di vm assignment e path planning. In generale prendono sempre come input un CQM, il proxytree e il proxymanager

### vm_model() & path_model()
Creano dei problemi CQM per i problemi di vm e path, il path model richiede la best solution del problema vm

### cqm_vm_solver() & bqm_vm_solver()
Risolvono il problema di vm assignment, il cqm ritorna una tupla contente la best solution e il tempo impiegato

### cqm_path_solver() & bqm_path_solver()
Risolvono il problema di path planning, il cqm ritorna il tempo impiegato