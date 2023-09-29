"""
TODO
    
"""
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
_____________________________________________________________________________________________________________________
#####################################################################################################################
"""

""" IMPORTS """
# D_WAVE
import dimod
import dwave.system
import dwave.inspector
import dwave.preprocessing

# OTHERS
import numpy as np
import random
import fun_lib as fn

"""--------------------------------------------------"""

DEBUG = True            # Debug boolean
ITERATIONS = 1          # Total problem iterations
TIME_MULT1 = 1           # CQM Time multiplier 1 for BQM in VM problem
TIME_MULT2 = 1           # CQM Time multiplier 2 for BQM in path problem

################# TREE DEFINITION PARAMETERS ###################
BINARY = False
DEPTH = 3                   # Tree depth
SERVER_C = 10               # Server capacity
LINK_C = 5*DEPTH            # Link capacity
LINK_C_DECREASE = 2
IDLE_PC = 10*DEPTH          # Idle power consumption
IDLE_PC_DECREASE = 5        # Idle power consumption
DYN_PC = 2*DEPTH            # Dynamic power consumption
DYN_PC_DECREASE = 1         # Dynamic power consumption
REQ_AVG = 8                 # Average flow request
DATAR_AVG = 4               # Average data rate per flow
LAGRANGE_MUL = IDLE_PC*10   # Lagrange multiplier for cqm -> bqm conversion
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

server_capacity = [SERVER_C for i in range(SERVERS)]           # Capacity of each server

link_capacity = []
# Switch links
for lvl in range(DEPTH-1):
    for link in range(2**(2*lvl+1)):
        link_capacity.append(LINK_C)
    LINK_C -= LINK_C_DECREASE
# Server links
for link in range(2**DEPTH):
    link_capacity.append(LINK_C)

idle_powcons = []           # Idle power consumption of each node
dyn_powcons = []            # Dynamic power of each node
for lvl in range(DEPTH+1):
    for node in range(2**lvl):
        idle_powcons.append(IDLE_PC)
        dyn_powcons.append(DYN_PC)
    IDLE_PC -= IDLE_PC_DECREASE
    DYN_PC -= DYN_PC_DECREASE

cpu_util = (np.random.normal(REQ_AVG, 1, VMS)).astype(int)         # CPU utilization of each VM
data_rate = (np.random.normal(DATAR_AVG, 1, (FLOWS, LINKS))).astype(int)      # Data rate of flow f on link l 

if DEBUG:
    print("SWITCH Indexes: ", *[k for k in range(SWITCHES)])
    print("SERVER Indexes: ", *[s+SWITCHES for s in range(SERVERS)])
    print("SERVER Capacity: ", *[s for s in server_capacity])
    print("LINK Capacity: ", *[s for s in link_capacity])
    print("IDLE Power Consumption: ", *[s for s in idle_powcons])
    print("DYNAMIC Power Consumption: ", *[s for s in dyn_powcons])
    print("VM's CPU Utilization: ", *[s for s in cpu_util])
    print("\n### Flow Path Data Rate ###")
    for links in data_rate:
        print("Flow ", *np.where(np.all(data_rate == links, axis=1))[0], ": ", *[s for s in links])
    print("\n\n")


# Create Tree:
link_dict = {}
link_counter = 0
adjancy_list = [[0 for j in range(NODES)] for i in range(NODES)]    # NxN Binary matrix of adjacent nodes
if(BINARY):
    for father in range(NODES):
        for i in range(2):
            son = father * 2 + 1 + i
            if (son < NODES):

                adjancy_list[father][son] = 1
                adjancy_list[son][father] = 1
                link_dict[str((father,son))] = link_counter
                link_dict[str((son,father))] = link_counter
                link_counter += 1
else:
    # Create all sw-sw links
    for lvl in range(DEPTH-1):
        if lvl == 0:
            for i in range(1,3):
                adjancy_list[0][i] = 1
                adjancy_list[i][0] = 1
                link_dict[str((0,i))] = link_counter
                link_dict[str((i,0))] = link_counter
                link_counter += 1
        else:
            first_sw = 2**(lvl) - 1
            last_sw = first_sw * 2
            for father in range(first_sw, last_sw + 1):
                first_son = 2**(lvl+1) - 1
                last_son = first_son * 2
                for son in range(first_son, last_son + 1):
                    adjancy_list[father][son] = 1
                    adjancy_list[son][father] = 1
                    link_dict[str((father,son))] = link_counter
                    link_dict[str((son,father))] = link_counter
                    link_counter += 1
    
    # Last layer first and last switch
    ll_firstsw = 2**(DEPTH-1) - 1
    ll_lastsw = ll_firstsw * 2
    
    # Create all sw-s links
    for father in range(ll_firstsw, ll_lastsw + 1):
        for i in range(2):
            son = father * 2 + 1 + i

            adjancy_list[father][son] = 1
            adjancy_list[son][father] = 1
            link_dict[str((father,son))] = link_counter
            link_dict[str((son,father))] = link_counter
            link_counter += 1

if DEBUG:
    print("### Tree Structure ###")
    for i in range(len(adjancy_list)):
        print("\nNodo ", i, " collegato ai nodi:", end="\t")
        for j in range(len(adjancy_list)):
            if adjancy_list[i][j] == 1:
                print(j, " (link ", link_dict.get(str((i,j))) ,")", sep="", end="\t")
    print("\n\n")


# Creating communicating VMS
src_dst = [[0 for j in range(2)] for i in range(FLOWS)]
index_list = [i for i in range(VMS)]
random.shuffle(index_list)
for i in range(FLOWS):
    for j in range(2):
        src_dst[i][j] = index_list[i*2 + j]

if DEBUG:
    print("### VM Paths ###")
    for path in src_dst:
        print("Path ", src_dst.index(path), ": ", end="\t")
        print( *[s for s in path], sep="  -  ")
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
obj1 = dimod.quicksum(server_status[s] * idle_powcons[s+SWITCHES] for s in range(SERVERS))
# 2 - SUM of vm dyn pow cons
obj2 = dimod.quicksum(dyn_powcons[s+SWITCHES] * dimod.quicksum(cpu_util[vm] * vm_status[s][vm] for vm in range(VMS)) for s in range(SERVERS))
# Total
vm_cqm.set_objective(obj1 + obj2)

# Constraints
# (11) For each server, the CPU utilization of each VM on that server must be less or equal than server's capacity       
for s in range(SERVERS):
    vm_cqm.add_constraint(
        dimod.quicksum(
            cpu_util[vm] * vm_status[s][vm] for vm in range(VMS)) 
        - server_capacity[s]*server_status[s] 
        <= 0)

# (12) For each VM, it must be active on one and only one server
for vm in range(VMS):
    vm_cqm.add_constraint(
        dimod.quicksum(
            vm_status[s][vm] for s in range(SERVERS)) 
        == 1)



print("\n\n\n")
print("####################### CQM VM Model ###########################")
print("\n")

# Create Sampler
cqm_sampler = dwave.system.LeapHybridCQMSampler()

# Sample results
cqm_samples = cqm_sampler.sample_cqm(vm_cqm)

# Exec time
cqm_time = cqm_samples.info.get('run_time')
print("CQM TIME: ", cqm_time, " micros")

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
vm_bqm, _ = dimod.cqm_to_bqm(vm_cqm, lagrange_multiplier = LAGRANGE_MUL)

# Roof Duality
rf_energy, _ = dwave.preprocessing.roof_duality(vm_bqm)

# Create Sampler
bqm_sampler = dwave.system.LeapHybridSampler()

# Sample Results
bqm_samples = bqm_sampler.sample(vm_bqm, cqm_time // 10**6 * TIME_MULT1)

# Exec Time
bqm_time = bqm_samples.info.get('run_time')
print("BQM TIME: ", bqm_time, " micros")

# Extract feasible solution
# bqm_feasibles = bqm_samples.filter(lambda sample: sample.is_feasible)

# Extract best solution
# bqm_best = bqm_feasibles.first
bqm_best = bqm_samples.first

# Energy
print("BQM ENERGY: ", str(bqm_best[1]))
print("BQM Roof Duality: ", rf_energy)

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
    for n1 in range(NODES):
        for n2 in range(NODES):
            if adjancy_list[n1][n2]:     # Adjancy Condition (lowers variable number)
                flow_path['f' + str(f) + '-n' + str(n1) + '-n' + str(n2)] = dimod.Binary("f" + str(f) + "-n" + str(n1) + "-n" + str(n2))

# Initialize dictionary for each adjacent node
# on[n1,n2] = 1 if the link between n1 and n2 is ON
on = {}
for n1 in range(NODES):
    for n2 in range(n1, NODES):
        if adjancy_list[n1][n2]:         # Adjancy Condition (lowers variable number)
            on["on" + str(n1) + "-" + str(n2)] = dimod.Binary("on" + str(n1) + "-" + str(n2))



############### PATH MODEL ########################################
# Create CQM
path_cqm = dimod.ConstrainedQuadraticModel()

# Objective
# 3 - SUM of switch idle pow cons
obj3 = dimod.quicksum(switch_status[sw] * idle_powcons[sw] for sw in range(SWITCHES))
# 4 - SUM of flow path
# obj4 = dimod.quicksum(flow_path['f' + str(f) + '-n' + str(n1) + '-n' + str(n2)] + flow_path['f' + str(f) + '-n' + str(n2) + '-n' + str(n1)]
                    # for n1 in range(NODES) for n2 in range(NODES) for f in range(FLOWS) if adjancy_list[n1][n2] == 1)
obj4 = dimod.quicksum( dyn_powcons[sw] * flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(sw)] + flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(n)]
                    for n in range(NODES) for f in range(FLOWS) for sw in range(SWITCHES) if adjancy_list[n][sw] == 1)
# Total
path_cqm.set_objective(obj3 + obj4)


# Constraints
# (13) For each flow and server, the sum of exiting flow from the server to all adj switch is <= than vms part of that flow     
for f in range(FLOWS):
    for s in range(SWITCHES, SWITCHES + SERVERS):           # Start from switches cause nodes are numerated in order -> all switches -> all servers
        path_cqm.add_constraint( 
            dimod.quicksum( 
                flow_path['f' + str(f) + '-n' + str(s) + '-n' + str(sw)] for sw in range(SWITCHES) if adjancy_list[s][sw] == 1) 
                - cqm_best[0].get("vm" + str(src_dst[f][0]) + "-s" + str(s-SWITCHES)) 
            <= 0)

# (14) For each flow and server, the sum of entering flow from the server to all adj switch is <= than vms part of that flow     
for f in range(FLOWS):
    for s in range(SWITCHES, SWITCHES + SERVERS):
        path_cqm.add_constraint( 
            dimod.quicksum( 
                flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(s)] for sw in range(SWITCHES) if adjancy_list[sw][s] == 1) 
                - cqm_best[0].get("vm" + str(src_dst[f][1]) + "-s" + str(s-SWITCHES)) 
            <= 0) 

# (15) For each flow and server, force allocation of all flows     
for f in range(FLOWS):
    for s in range(SWITCHES, SWITCHES + SERVERS):
        path_cqm.add_constraint( 
            cqm_best[0].get("vm" + str(src_dst[f][0]) + "-s" + str(s-SWITCHES)) - cqm_best[0].get("vm" + str(src_dst[f][1]) + "-s" + str(s-SWITCHES))  
            - ( 
                dimod.quicksum( 
                    flow_path['f' + str(f) + '-n' + str(s) + '-n' + str(sw)] for sw in range(SWITCHES) if adjancy_list[s][sw] == 1) 
                - dimod.quicksum( 
                    flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(s)] for sw in range(SWITCHES) if adjancy_list[sw][s] == 1)) 
            == 0)

# (16) For each switch and flow, entering and exiting flow from the switch are equal
for sw in range(SWITCHES):
    for f in range(FLOWS):
        path_cqm.add_constraint( 
            dimod.quicksum( 
                flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(sw)]  for n in range(NODES) if adjancy_list[n][sw] == 1) 
            - dimod.quicksum( 
                flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(n)] for n in range(NODES) if adjancy_list[sw][n] == 1) 
            == 0)

# (17) For each link, the data rate on it is less or equal than its capacity      
for l in range(LINKS):
    n1,n2 = fn.get_nodes(l, link_dict)
    path_cqm.add_constraint( 
        dimod.quicksum( 
            data_rate[f][l] * (
                flow_path['f' + str(f) + '-n' + str(n1) + '-n' + str(n2)] 
                + flow_path['f' + str(f) + '-n' + str(n2) + '-n' + str(n1)]) for f in range(FLOWS)) 
        - link_capacity[l] * on["on" + str(n1) + "-" + str(n2)] 
        <= 0)

# (18)(19) For each link, the link is ON only if both nodes are ON       
for l in range(LINKS):
    n1,n2 = fn.get_nodes(l, link_dict)
    if n1 < SWITCHES:
        path_cqm.add_constraint(
            on["on" + str(n1) + "-" + str(n2)] 
            - switch_status[n1] 
            <= 0)
    else:
        path_cqm.add_constraint(
            on["on" + str(n1) + "-" + str(n2)] 
            - cqm_best[0].get("s"+str(n1-SWITCHES)) 
            <= 0)
    if n2 < SWITCHES:
        path_cqm.add_constraint(
            on["on" + str(n1) + "-" + str(n2)] 
            - switch_status[n2] 
            <= 0)
    else:
        path_cqm.add_constraint(
            on["on" + str(n1) + "-" + str(n2)] 
            - cqm_best[0].get("s"+str(n2-SWITCHES)) 
            <= 0)



print("\n\n\n")
print("####################### CQM Path Model ###########################")
print("\n")

# Sample results
cqm_samples = cqm_sampler.sample_cqm(path_cqm)

# Exec time
cqm_time = cqm_samples.info.get('run_time')
print("CQM TIME: ", cqm_time, " micros")

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
path_bqm, _ = dimod.cqm_to_bqm(path_cqm, lagrange_multiplier = LAGRANGE_MUL)

# Roof Duality
rf_energy, _ = dwave.preprocessing.roof_duality(path_bqm)

# Sample Results
bqm_samples = bqm_sampler.sample(path_bqm, cqm_time // 10**6 * TIME_MULT2)

# Exec Time
bqm_time = bqm_samples.info.get('run_time')
print("BQM TIME: ", bqm_time, " micros")

# Extract feasible solution
# bqm_feasibles = bqm_samples.filter(lambda sample: sample.is_feasible)

# Extract best solution
# bqm_best = bqm_feasibles.first
bqm_best = bqm_samples.first

# Energy
print("BQM ENERGY: ", str(bqm_best[1]))
print("BQM Roof Duality: ", rf_energy)

# Extract variables
for i in bqm_best[0]:
    if bqm_best[0].get(i) != 0.0:
        print(i, bqm_best[0].get(i),sep = ": ",end= " | ")



# print("\n\n\n")
# print("####################### BQM Full Quantum Path Model ###########################")
# print("\n")

# # Create Sampler
# fq_bqm_sampler = dwave.system.EmbeddingComposite(dwave.system.DWaveSampler())

# # Sample Results
# fq_bqm_samples = fq_bqm_sampler.sample(path_bqm, num_reads=100)

# # Exec Time
# fq_bqm_time = fq_bqm_samples.info.get('run_time')
# print("BQM TIME: ", fq_bqm_time, " micros")

# # Extract feasible solution
# # bqm_feasibles = bqm_samples.filter(lambda sample: sample.is_feasible)

# # Extract best solution
# # bqm_best = bqm_feasibles.first
# fq_bqm_best = fq_bqm_samples.first

# # Energy
# print("Full Quantum BQM ENERGY: ", str(fq_bqm_best[1]))

# # Extract variables
# for i in fq_bqm_best[0]:
#     if fq_bqm_best[0].get(i) != 0.0:
#         print(i, fq_bqm_best[0].get(i),sep = ": ",end= " | ")
