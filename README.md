# CHECK
- L'obiettivo funziona davvero come indicato?
- Check obiettivo
  - manca la parte del consumo delle singole porte?

# TO DO
- Create class
- Create main cycle
- rimetti le condizioni di adiacenza


# PROBLEMI
- funziona solo con 1 VM per server


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