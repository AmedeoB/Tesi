# PROBLEMI
- L'obiettivo funziona davvero come indicato?
- Check obiettivo
  - manca la parte del consumo delle singole porte?
- funziona solo con 1 VM per server
- il CQM è molto più veloce del BQM nonostante dovrebbe essere il 
    (Video: Quantum Programming Tutorial | D-Wave Qubits 2021 \\ Time:58:30)
- il BQM dovrebbe fare più iterazioni per trovare il best result? 
  - {[Linea 292] sampler.sample(bqm, num_reads=1000)}
    - testato e non migliora
- on non è usato nell'obiettivo anche se è una variabile


# TO DO
- Create class
- Create main cycle
- Label constraints
- Change Dictionary to matrix? In NumPy vectors
- Check obj4, itera due volte sui nodi


# TEST
- senza condizioni di adiacenza 
    - CQM   t: 1.836      e: 191.0
    - BQM   t: 55.821     e: 70526.0
- con condizioni di adiacenza su on{}
    - CQM   t: 1.950      e: 190.0
    - BQM   t: 57.607     e: 74206.0


---------------------------------------------------------

# D-WAVE DECOMPOSER
https://docs.dwavesys.com/docs/latest/handbook_decomposing.html

Energy impact decomposing
- implemented by EnergyImpactDecomposer class ( https://docs.ocean.dwavesys.com/en/stable/docs_hybrid/reference/decomposers.html#hybrid.decomposers.EnergyImpactDecomposer )