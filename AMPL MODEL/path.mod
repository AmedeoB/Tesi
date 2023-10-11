# TO DO
# capire la conversione da link a nodi collegati

### INSIEMI ###
set SERVERS;
set VMS;
set SWITCHES;
set FLOWS;
set NODES;
set ENTER_EXIT;
set LINKS;


### PARAMETRI ###
param vm_status{VMS, SERVERS};
param src_dst{FLOWS, ENTER_EXIT};
param link_capacity{LINKS}


### VARIABILI ###
var switch_status {sw in SWITCHES} binary, default 0;
var flow_path {f in FLOWS, n1 in NODES, n2 in NODES} binary, default 0;
var on {n1 in NODES, n2 in NODES} binary, default 0;


### VINCOLI ###
subject to C13 {f in FLOWS, s in SERVERS}:
    sum{sw in SWITCHES} flow_path[f][s][sw] <= vm_status[src_dst[f][0]][s]
;

subject to C14 {f in FLOWS, s in SERVERS}:
    sum{sw in SWITCHES} flow_path[f][sw][s] <= vm_status[src_dst[f][1]][s]
;

subject to C15 {f in FLOWS, s in SERVERS}:
    vm_status[src_dst[f][0]][s] + vm_status[src_dst[f][1]][s] ==
    sum{sw in SWITCHES} flow_path[f][s][sw] - flow_path[f][sw][s]
;

subject to C16 {sw in SWITCHES, f in FLOWS}:
    sum{n in NODES} flow_path[f][n][sw] - flow_path[f][sw][n] == 0
;

subject to C17 {l in LINKS}:
    sum{f in FLOWS} data_rate[f] * (flow_path[f][n1][n2] + flow_path[f][n2][n1])
    <= link_capacity[l] * on[n1][n2]
;

subject to C18e19 {l in LINKS}:
    on[n1][n2] <= ????
;

### OBIETTIVO ###
minimize energy:
    sum{sw in SWITCHES} switch_status[sw] * idle_powcons[sw] + 
    sum{sw in SWITCHES, f in FLOWS, n in NODES} dyn_powcons[sw] * (flow_path[f][n][sw] + flow_path[f][sw][n])
;
