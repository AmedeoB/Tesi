"""
################################### MODEL VARIABLES DICTIONARY ######################################################
_____________________________________________________________________________________________________________________
Variable         |  Type         |  Model        |  Description
_________________|_______________|_______________|___________________________________________________________________
SERVERS             int             M               number of servers
VMS                 int             N               number of VM
SWITCHES            int             K               number of switches
FLOWS               int             F               number of flows (server-server paths, = M/2)
LINKS               int             L               number of links (graph links)
idle_powcons        1D list         p(idle_i/k)     idle power consumption fo each node (i server, k switch)
dyn_powcons         1D list         p(dyn_i/k)      maximum dynamic power of each node (i server, k switch) 
adjancy_list        2D list         ---             node's adjancy list
link_capacity       1D list         C(l)            capacity of each link
server_capacity     1D list         C(s)            capacity of each server
src_dst             2D list         ---             list of vms communicating through a path, identifies flows
server_status       1D list         s(i)            server (i) status, 1 on, 0 off
switch_status       1D list         sw(k)           switch (k) status, 1 on, 0 off
vm_status           2D list         v(ji)           VM (j) status per server (i), 1 on, 0 off
cpu_util            1D array        u(v(ji))        CPU utilization of each VM 
data_rate           2D array        d(fl)           data rate of flow (f) on link (l)
flow_path           bin dictionary  ρ(f,(k,i))      se parte del flow (f) va da k a i (nodi), allora 1, 0 altrimenti
on                  bin dictionary  on(n1, n2)      link between node n1 and n2 is ON                
#####################################################################################################################
"""

""" IMPORTS """
# D_WAVE
import dimod
import dwave.system

# OTHERS
import numpy as np
import random

"""--------------------------------------------------"""

DEBUG = True            # Debug boolean
ITERATIONS = 1          # Total problem iterations
TIME_MULT = 1           # CQM Time multiplier fro BQM

################# TREE DEFINITION PARAMETERS ###################
BINARY = False
DEPTH = 3               # Tree depth
SERVER_C = 10           # Server capacity
LINK_C = 10             # Link capacity
IDLE_PC = 10            # Idle power consumption
DYN_PC = 1              # Dynamic power consumption
REQ_AVG = 8             # Average flow request
DATAR_AVG = 4           # Average data rate per flow
#################################################################

SERVERS = pow(2, DEPTH)                                 # Server number
SWITCHES = sum(pow(2,i) for i in range(DEPTH))          # Switch number
VMS = SERVERS                                           # VM number per server
FLOWS = VMS//2 if VMS%2==0 else VMS//2+1                # Flow number
NODES = SERVERS + SWITCHES                              # Total Nodes
if(BINARY):
    LINKS = 2*SWITCHES                                      # Link number
else:
    LINKS = 0
    for i in range(DEPTH-1):
        LINKS += 2**i * 2**(i+1)
    LINKS += 2*(2**(DEPTH-1))
if DEBUG:
    print("SWITCH Indexes: ", *[k for k in range(SWITCHES)])
    print("SERVER Indexes: ", *[s+SWITCHES for s in range(SERVERS)])

server_capacity = [SERVER_C for i in range(SERVERS)]           # Capacity of each server
link_capacity = [LINK_C for i in range (LINKS)]              # Capacity of each Link
idle_powcons = [IDLE_PC for i in range(NODES)]   # Idle power consumption of each node
dyn_powcons = [DYN_PC for i in range(NODES)]     # Maximum dynamic power of each node
cpu_util = (np.random.normal(REQ_AVG, 1, VMS)).astype(int)         # CPU utilization of each VM
data_rate = (np.random.normal(DATAR_AVG, 1, (FLOWS, LINKS))).astype(int)      # Data rate of flow f on link l 
if DEBUG:
	print("\n### CPU Utilization ###\n", cpu_util)
	print("\n### Flow Path Data Rate ###\n", data_rate)

# Create Tree:
adjancy_list = [[0 for j in range(NODES)] for i in range(NODES)]    # NxN Binary matrix of adjacent nodes
if(BINARY):
    for i in range(NODES):
        first_son = i*2 + 1
        second_son = first_son + 1
        if (second_son < NODES):
            adjancy_list[i][first_son] = 1
            adjancy_list[i][second_son] = 1
            adjancy_list[first_son][i] = 1
            adjancy_list[second_son][i] = 1
else:
    # Create all sw-sw links
    for lvl in range(DEPTH-1):
        if lvl == 0:
            for i in range(1,3):
                adjancy_list[0][i] = 1
                adjancy_list[i][0] = 1
        else:
            first_sw = 2**(lvl) - 1
            last_sw = first_sw * 2
            for father in range(first_sw, last_sw + 1):
                first_son = 2**(lvl+1) - 1
                last_son = first_son * 2
                for son in range(first_son, last_son + 1):
                    adjancy_list[father][son] = 1
                    adjancy_list[son][father] = 1
    
    # Last layer first and last switch
    ll_firstsw = 2**(DEPTH-1) - 1
    ll_lastsw = ll_firstsw * 2
    
    # Create all sw-s links
    for switch in range(ll_firstsw, ll_lastsw + 1):
        first_son = switch * 2 + 1
        second_son = first_son + 1
        adjancy_list[switch][first_son] = 1
        adjancy_list[switch][second_son] = 1
        adjancy_list[first_son][switch] = 1
        adjancy_list[second_son][switch] = 1

if DEBUG:
    for i in range(len(adjancy_list)):
        print("\nNodo ", i, " collegato ai nodi:", end="\t")
        for j in range(len(adjancy_list)):
            if adjancy_list[i][j] == 1:
                print(j, end=" ")
    print("\n")

# Creating communicating VMS
src_dst = [[0 for j in range(2)] for i in range(FLOWS)]
index_list = [i for i in range(VMS)]
random.shuffle(index_list)

for i in range(FLOWS):
    for j in range(2):
        src_dst[i][j] = index_list[i*2 + j]

print("VM Paths:", end=" ")
print(src_dst)
print("\n")


############### VM MODEL BINARY VARIABLES ###################################
server_status = [dimod.Binary("s" + str(i)) for i in range(SERVERS)]          # Binary value for each server, 1 ON, 0 OFF
vm_status = [[dimod.Binary("vm" + str(i) + "-s" + str(j)) 
        for i in range(VMS)] for j in range(SERVERS)]                   # Binary value for each VM on each server, 1 ON, 0 OFF



############### VM MODEL ########################################
# Create CQM
vm_cqm = dimod.ConstrainedQuadraticModel()

# Objective
# 1 - SUM of server pow cons
obj1 = dimod.quicksum(server_status[i] * idle_powcons[i+SWITCHES] for i in range(SERVERS))
# 2 - SUM of vm dyn pow cons
obj2 = dimod.quicksum(dyn_powcons[i+SWITCHES] * dimod.quicksum(cpu_util[j] * vm_status[i][j] for j in range(VMS)) for i in range(SERVERS))
# Total
vm_cqm.set_objective(obj1 + obj2)

# Constraints
# (11) For each server, the CPU utilization of each VM on that server must be less or equal than server's capacity       
for i in range(SERVERS):
    vm_cqm.add_constraint(dimod.quicksum(cpu_util[j] * vm_status[i][j] for j in range(VMS)) - server_capacity[i]*server_status[i] <= 0)

# (12) For each VM, it can only be active on one server
for j in range(VMS):
    vm_cqm.add_constraint(dimod.quicksum(vm_status[i][j] for i in range(SERVERS)) == 1)


print("\n\n\n")
print("####################### CQM VM Model ###########################")
print("\n")

# Create Sampler
cqm_sampler = dwave.system.LeapHybridCQMSampler()

# Sample results
cqm_samples = cqm_sampler.sample_cqm(vm_cqm)

# Exec time
print("CQM TIME: ", cqm_samples.info.get('run_time')," micros")

# Extract feasible solution
cqm_feasibles = cqm_samples.filter(lambda sample: sample.is_feasible)

# Extract best solution
cqm_best = cqm_feasibles.first

# Energy
print("CQM ENERGY: ", str(cqm_best[1]))

# Extract variables
for i in cqm_best[0]:
    if cqm_best[0].get(i) != 0.0:
        print(i, cqm_best[0].get(i),sep = ": ",end= " | ")



print("\n\n\n")
print("####################### BQM VM Model ###########################")
print("\n")

# From CQM to BQM
vm_bqm, _ = dimod.cqm_to_bqm(vm_cqm)

# Create Sampler
bqm_sampler = dwave.system.LeapHybridSampler()

# Sample Results
bqm_samples = bqm_sampler.sample(vm_bqm)

# Exec Time
print("CQM TIME: ", bqm_samples.info.get('run_time')," micros")

# Extract feasible solution
# bqm_feasibles = bqm_samples.filter(lambda sample: sample.is_feasible)

# Extract best solution
# bqm_best = bqm_feasibles.first
bqm_best = bqm_samples.first

# Energy
print("BQM ENERGY: ", str(bqm_best[1]))

# Extract variables
for i in bqm_best[0]:
    if bqm_best[0].get(i) != 0.0:
        print(i, bqm_best[0].get(i),sep = ": ",end= " | ")




############### PATH MODEL BINARY VARIABLES ###################################
switch_status = [dimod.Binary("sw" + str(i)) for i in range(SWITCHES)]        # Binary value for each switch, 1 ON, 0 OFF

# Initialize flow_path dictionary for each possible combination of flow and adjacent node
# flow_path[f, [n1,n2]] = 1 if part of flow f goes from n1 to n2
flow_path = {}
for f in range(FLOWS):
    for i in range(SWITCHES + SERVERS):
        for k in range(SWITCHES + SERVERS):
            if adjancy_list[i][k]:     # Adjancy Condition (lowers variable number)
                flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(k)] = dimod.Binary("f" + str(f) + "-n" + str(i) + "-n" + str(k))

# Initialize dictionary for each adjacent node
# on[n1,n2] = 1 if the link between n1 and n2 is ON
on = {}
for i in range(NODES):
    for j in range(NODES):
        if adjancy_list[i][j]:         # Adjancy Condition (lowers variable number)
            on["on" + str(i) + "-" + str(j)] = dimod.Binary("on" + str(i) + "-" + str(j))


############### PATH MODEL ########################################
# Create CQM
path_cqm = dimod.ConstrainedQuadraticModel()

# Objective
# 3 - SUM of switch idle pow cons
obj3 = dimod.quicksum(switch_status[i] * idle_powcons[i] for i in range(SWITCHES))
# 4 - SUM of flow path
obj4 = dimod.quicksum(flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(j)] + flow_path['f' + str(f) + '-n' + str(j) + '-n' + str(i)]
                    for i in range(NODES) for j in range (NODES) for f in range(FLOWS) if adjancy_list[i][j] == 1)
# Total
path_cqm.set_objective(obj3 + obj4)

# Constraints
# (13) For each flow and server, the sum of exiting flow from the server to all adj switch is <= than vms part of that flow     
for f in range(FLOWS):
    for i in range(SWITCHES, SWITCHES + SERVERS):           # Start from switches cause nodes are numerated in order -> all switches -> all servers
        path_cqm.add_constraint( dimod.quicksum( flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(k)] for k in range(SWITCHES) if adjancy_list[i][k] == 1) 
                - cqm_best[0].get("vm"+str(i-SWITCHES)+"-s"+str(src_dst[f][0])) <= 0)

# (14) For each flow and server, the sum of entering flow from the server to all adj switch is <= than vms part of that flow     
for f in range(FLOWS):
    for i in range(SWITCHES, SWITCHES + SERVERS):
        path_cqm.add_constraint( dimod.quicksum( flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(i)] for k in range(SWITCHES) if adjancy_list[k][i] == 1) 
                - cqm_best[0].get("vm"+str(i-SWITCHES)+"-s"+str(src_dst[f][1])) <= 0) 

# (15) For each flow and server, force allocation of all flows     
for f in range(FLOWS):
    for i in range(SWITCHES, SWITCHES + SERVERS):
        path_cqm.add_constraint( cqm_best[0].get("vm"+str(i-SWITCHES)+"-s"+str(src_dst[f][0])) - cqm_best[0].get("vm"+str(i-SWITCHES)+"-s"+str(src_dst[f][1]))  
                - ( dimod.quicksum(flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(k)] for k in range(SWITCHES) if adjancy_list[i][k] == 1) - 
                dimod.quicksum( flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(i)] for k in range(SWITCHES) if adjancy_list[k][i] == 1)) == 0)

# (16) For each switch and flow, entering and exiting flow from the switch are equal
for k in range(SWITCHES):
    for f in range(FLOWS):
        path_cqm.add_constraint( dimod.quicksum( flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(k)]  for n in range(NODES) if adjancy_list[n][k] == 1) - 
                dimod.quicksum( flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(n)] for n in range(NODES) if adjancy_list[k][n] == 1) == 0 )

# (17) For each link, the data rate on a it is less or equal than its capacity      
for l in range(LINKS):
    father = l//2       # Father node of the link
    son = l+1           # Son node of the link
    path_cqm.add_constraint( dimod.quicksum( data_rate[f][l] * (flow_path['f' + str(f) + '-n' + str(father) + '-n' + str(son)] + 
            flow_path['f' + str(f) + '-n' + str(son) + '-n' + str(father)]) for f in range(FLOWS)) - 
            link_capacity[l] * on["on" + str(father) + "-" + str(son)] <= 0)

# (18)(19) For each link, the link is ON only if both nodes are ON       
for l in range(LINKS):
    father = l//2           # Father node of the link
    son = l+1               # Son node of the link
    if son >= SWITCHES:     # Son is a Server
        path_cqm.add_constraint(on["on" + str(father) + "-" + str(son)] - switch_status[father] <= 0)
        path_cqm.add_constraint(on["on" + str(father) + "-" + str(son)] - cqm_best[0].get("s"+str(son-SWITCHES)) <= 0)
    else:                   # Both nodes are switches
        path_cqm.add_constraint(on["on" + str(father) + "-" + str(son)] - switch_status[father] <= 0)
        path_cqm.add_constraint(on["on" + str(father) + "-" + str(son)] - switch_status[son] <= 0)


print("\n\n\n")
print("####################### CQM Path Model ###########################")
print("\n")

# Sample results
cqm_samples = cqm_sampler.sample_cqm(path_cqm)

# Exec time
print("CQM TIME: ", cqm_samples.info.get('run_time')," micros")

# Extract feasible solution
cqm_feasibles = cqm_samples.filter(lambda sample: sample.is_feasible)

# Extract best solution
cqm_best = cqm_feasibles.first

# Energy
print("CQM ENERGY: ", str(cqm_best[1]))

# Extract variables
for i in cqm_best[0]:
    if cqm_best[0].get(i) != 0.0:
        print(i, cqm_best[0].get(i),sep = ": ",end= " | ")



print("\n\n\n")
print("####################### BQM Path Model ###########################")
print("\n")

# From CQM to BQM
path_bqm, _ = dimod.cqm_to_bqm(path_cqm)

# Create Sampler
bqm_sampler = dwave.system.LeapHybridSampler()

# Sample Results
bqm_samples = bqm_sampler.sample(path_bqm)

# Exec Time
print("CQM TIME: ", bqm_samples.info.get('run_time')," micros")

# Extract feasible solution
# bqm_feasibles = bqm_samples.filter(lambda sample: sample.is_feasible)

# Extract best solution
# bqm_best = bqm_feasibles.first
bqm_best = bqm_samples.first

# Energy
print("BQM ENERGY: ", str(bqm_best[1]))

# Extract variables
for i in bqm_best[0]:
    if bqm_best[0].get(i) != 0.0:
        print(i, bqm_best[0].get(i),sep = ": ",end= " | ")
