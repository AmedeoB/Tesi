### INSIEMI ###
set SERVERS;
set VMS;
set SWITCHES;
set FLOWS;
set NODES;


### PARAMETRI ###


### VARIABILI ###
var switch_status {sw in SWITCHES} binary, default 0;
var flow_path {f in FLOWS, n1 in NODES, n2 in NODES} binary, default 0;
var on {n1 in NODES, n2 in NODES} binary, default 0;


### VINCOLI ###


### OBIETTIVO ###

