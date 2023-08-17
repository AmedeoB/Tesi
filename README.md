# PROBLEMI
- funziona solo con 1 VM per server


# TO DO
- Add Label constraints
- Change Dictionary to matrix? In NumPy vectors
- Check obj4, itera due volte sui nodi


# TEST
- OBJ 1 (Fail runs: 0 / 10)
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 40.0     t: 0.0305
        - BQM   e: 40.0     t: 0.7944
- OBJ 2 (Fail: 4 / 10)
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 24.0     t: 0.0309
        - BQM   e: 27.0     t: 0.7806
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 26.0     t: 0.0321
        - BQM   e: 27.0     t: 0.7702
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 27.0     t: 0.0300
        - BQM   e: 28.0     t: 0.8655
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 27.0     t: 0.0299
        - BQM   e: 30.0     t: 0.8430
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 25.0     t: 0.0306
        - BQM   e: 25.0     t: 1.6003
    - ////// Runs: 2 / 10 ///////
        - CQM   e: 26.0     t: 0.0294
        - BQM   e: 26.0     t: 0.7814
    - ////// Runs: 2 / 10 ///////
        - CQM   e: 27.0     t: 0.0300
        - BQM   e: 27.0     t: 0.8655
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 28.0     t: 0.0347
        - BQM   e: 28.0     t: 0.9656
- OBJ 3 (Fail: 8 / 10)
    - ////// Runs: 8 / 10 ///////
        - CQM   e: 20.0     t: 0.0308
        - BQM   e: 30.0     t: 0.7598
    - ////// Runs: 2 / 10 ///////
        - CQM   e: 20.0     t: 0.0297
        - BQM   e: 20.0     t: 0.8195
- OBJ 4 (Fail: 5 / 10)
    - ////// Runs: 4 / 10 ///////
        - CQM   e: 8.0     t: 0.0309
        - BQM   e: 12.0    t: 1.0969
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 8.0     t: 0.0301
        - BQM   e: 32.0    t: 0.8825
    - ////// Runs: 5 / 10 ///////
        - CQM   e: 8.0     t: 0.0309
        - BQM   e: 8.0     t: 0.7716 (3.3145)
- OBJ 4B (Fail: 10 / 10)
    - ////// Runs: 2 / 10 ///////
        - CQM   e: 4.0     t: 0.0318
        - BQM   e: 8.0     t: 0.7549
    - ////// Runs: 5 / 10 ///////
        - CQM   e: 4.0     t: 0.0342
        - BQM   e: 12.0    t: 0.7676
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 4.0     t: 0.0313
        - BQM   e: 28.0    t: 0.7988
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 4.0     t: 0.0324
        - BQM   e: 32.0    t: 0.8738
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 4.0     t: 0.0321
        - BQM   e: 46.0    t: 0.8739
- OBJ 4B 2.0 (Fail: 7 / 10)
    - ////// Runs: 5 / 10 ///////
        - CQM   e: 4.0     t: 0.0318
        - BQM   e: 8.0     t: 0.7549
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 4.0     t: 0.0342
        - BQM   e: 12.0    t: 0.7676
    - ////// Runs: 1 / 10 ///////
        - CQM   e: 4.0     t: 0.0305
        - BQM   e: 24.0    t: 0.9067
    - ////// Runs: 3 / 10 ///////
        - CQM   e: 4.0     t: 0.0301
        - BQM   e: 4.0     t: 0.7889
 

---------------------------------------------------------

# D-WAVE DECOMPOSER
https://docs.dwavesys.com/docs/latest/handbook_decomposing.html

Energy impact decomposing
- implemented by EnergyImpactDecomposer class ( https://docs.ocean.dwavesys.com/en/stable/docs_hybrid/reference/decomposers.html#hybrid.decomposers.EnergyImpactDecomposer )
