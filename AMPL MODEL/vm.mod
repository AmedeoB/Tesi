### INSIEMI ###
set SERVERS;
set VMS;


### PARAMETRI ###
param server_capacity{SERVERS};
param idle_powcons{SERVERS};
param dyn_powcons{SERVERS};
param cpu_util{VMS};


### VARIABILI ###
var server_status{SERVERS} binary, default 0;
var vm_status {SERVERS, VMS} binary, default 0;


### VINCOLI ###
subject to c11 {s in SERVERS}:
    sum{v in VMS} cpu_util[v]* vm_status[s][v] <= server_capacity[s] * server_status[s]
;

subject to c12 {v in VMS}:
    sum{s in SERVERS} vm_status[s][v] == 1
;


### OBIETTIVO ###
minimize energy:
    sum{s in SERVERS} server_status[s] * idle_powcons[s] + 
    sum{s in SERVERS, v in VMS} vm_status[s][v] * cpu_util[v] * dyn_powcons[s]
;