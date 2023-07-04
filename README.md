# PROBLEMI
- oldmain.py
  - codice pessimo
  - modificare dei calcoli di tempo
  - non funziona con server dispari
  - variabili non usate
  - codice commentato da pulire
  - codice ripetuto
  - da errori (leggi sotto)
- Dwave time limit ?

# D-WAVE DECOMPOSER
https://docs.dwavesys.com/docs/latest/handbook_decomposing.html

Energy impact decomposing
- implemented by EnergyImpactDecomposer class ( https://docs.ocean.dwavesys.com/en/stable/docs_hybrid/reference/decomposers.html#hybrid.decomposers.EnergyImpactDecomposer )

# ERROR 
PROFONDITA': 3
###########################################
Paths
[[5, 4], [6, 2], [0, 1], [7, 3]]
time: 8.666970491409302
Gli indici da 0 a 7 sono i server
Gli indici da 8 a 14 sono gli switch
Collegamenti attivi: 
on0-11
on1-11
on11-0
on11-1
on12-2
on12-3
on13-4
on13-5
on14-6
on14-7
on2-12
on3-12
on4-13
on5-13
on6-14
on7-14
rho[f, [n1, n2]] = 1 se parte del flusso f-esimo va da n1 ad n2
rho0-13-5
rho0-4-13
rho1-1-11
rho1-11-0
rho2-12-3
rho2-2-12
rho3-14-7
rho3-6-14
Switch/server attivi
s0
s1
s2
s3
s4
s5
s6
s7
sw3
sw4
sw5
sw6
v[j, i]: la j-esima macchina virtuale sul i-esimo server
v0-2
v1-3
v2-0
v3-7
v4-5
v5-4
v6-1
v7-6
