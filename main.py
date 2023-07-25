"""
####### MODEL VARIABLES DICTIONARY #######
__________________________________________________________________________________________________
Variable         |  Type         |  Model        |  Description
_________________|_______________|_______________|_________________________________________________
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
cpu_util            2D array        u(v(ji))        CPU utilization of each VM v(ji)
data_rate           2D array        d(fl)           data rate of flow (f) on link (l)

flow_path           bin dictionary  ρ(f,(k,i))      se parte del flow (f) va da k a i (nodi), allora 1, 0 altrimenti
on                  bin dictionary  on(n1, n2)      link between node n1 and n2 is ON                
"""

# IMPORTS D-WAVE
from dimod import ConstrainedQuadraticModel, Integer, QuadraticModel, Binary, quicksum, cqm_to_bqm 
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridCQMSampler
import dwave.inspector
from dwave.preprocessing import roof_duality

# IMPORTS OTHERS
from scipy.stats import norm
import numpy as np
from random import seed, randint
import time
import re

################# CONSTANTS #################
seed()                  # Init random seed
DEBUG = True            # Debug bool
DEPTH = 2               # Tree depth
#############################################



SERVERS = pow(2, DEPTH)                                 # Server number
VMS = SERVERS                                           # VM number per server
SWITCHES = sum(pow(2,i) for i in range(DEPTH))          # Switch number
FLOWS = VMS//2 if VMS%2==0 else VMS//2+1                # Flow number
LINKS = 2*SWITCHES                                      # Link number
NODES = SERVERS + SWITCHES                              # Total Nodes

server_capacity = [10 for i in range(SERVERS)]           # Capacity of each server
link_capacity = [10 for i in range (LINKS)]              # Capacity of each Link
idle_powcons = [10 for i in range(NODES)]   # Idle power consumption of each node
dyn_powcons = [1 for i in range(NODES)]     # Maximum dynamic power of each node


adjancy_list = [[0 for j in range(NODES)] 
        for i in range(NODES)]              # Binary list of adjacent nodes (0 non-andj, 1 adj)

server_status = [Binary("s" + str(i)) for i in range(SERVERS)]          # Binary value for each server, 1 ON, 0 OFF
switch_status = [Binary("sw" + str(i)) for i in range(SWITCHES)]        # Binary value for each switch, 1 ON, 0 OFF
vm_status = [[Binary("vm" + str(i) + "-s" + str(j)) 
        for i in range(VMS)] for j in range(SERVERS)]                   # Binary value for each VM on each server, 1 ON, 0 OFF

cpu_util = (np.random.normal(8, 1, (SERVERS, VMS))).astype(int)         # CPU utilization of each VM on each server
data_rate = (np.random.normal(4, 1, (FLOWS, LINKS))).astype(int)        # Data rate of flow f on link l 

# Calculate adjancy list:
for i in range(NODES):
    first_son = i*2 + 1
    second_son = first_son + 1
    if (second_son < NODES):
        adjancy_list[i][first_son] = 1
        adjancy_list[i][second_son] = 1
        adjancy_list[first_son][i] = 1
        adjancy_list[second_son][i] = 1

if DEBUG:
    for i in range(len(adjancy_list)):
        print()
        print("Nodo ", i, " collegato ai nodi:")
        for j in range(len(adjancy_list)):
            if adjancy_list[i][j] == 1:
                print(j, end=" ")
    print("\n")


src_dst = [[0 for j in range(2)] for i in range(FLOWS)]     # list of commmunicating servers
randoms = []                                                # list of generated random values
for i in range(FLOWS):
    for j in range(2):
        while True:            
            ran = randint(0, VMS-1)
            
            if not ran in randoms:
                # Fill list
                src_dst[i][j] = ran                
                # Keep number recorded
                randoms.append(ran)                
                break
print("Paths:", end=" ")
print(src_dst)
print("\n")


# Initialize flow_path dictionary for each possible combination of flow and adjacent node
# flow_path[f, [n1,n2]] = 1 if part of flow f goes from n1 to n2
flow_path = {}
for f in range(FLOWS):
    for i in range(SWITCHES + SERVERS):
        for k in range(SWITCHES + SERVERS):
            #if adjancy_list[i][k]:     # Adjancy Condition (unnecessary)
                flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(k)] = Binary("f" + str(f) + "-n" + str(i) + "-n" + str(k))

# Initialize dictionary for each adjacent node
# on[n1,n2] = 1 if the link between n1 and n2 is ON
on = {}
for i in range(NODES):
    for j in range(NODES):
        #if adjancy_list[i][j]:         # Adjancy Condition (unnecessary)
            on["on" + str(i) + "-" + str(j)] = Binary("on" + str(i) + "-" + str(j))



######################### CQM MODEL #########################
# Create CQM
cqm = ConstrainedQuadraticModel()


# OBJECTIVE
# Define Subobjectives
obj1 = quicksum(server_status[i] * idle_powcons[i+SWITCHES] for i in range(SERVERS))
obj2 = quicksum(dyn_powcons[i+SWITCHES] * quicksum(cpu_util[i][j] * vm_status[i][j] for j in range(VMS)) for i in range(SERVERS))
obj3 = quicksum(switch_status[i] * idle_powcons[i] for i in range(SWITCHES))

obj4 = quicksum(flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(j)] + flow_path['f' + str(f) + '-n' + str(j) + '-n' + str(i)]
                    for i in range(NODES) for j in range (NODES) for f in range(FLOWS) if adjancy_list[i][j] == 1)
"""
obj4 = quicksum(flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(n)] + flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(k)] 
                    for k in range(SWITCHES) for n in range(NODES) for f in range(FLOWS))
"""
# Set Objective
cqm.set_objective(obj1 + obj2 + obj3 + obj4)


# CONSTRAINTS
# (11) For each server, the CPU utilization of each VM on that server must be less or equal than server's capacity       
for i in range(SERVERS):
    cqm.add_constraint(quicksum(cpu_util[i][j] * vm_status[i][j] for j in range(VMS)) - server_capacity[i]*server_status[i] <= 0)

# (12) For each VM, it can only be active on one server
for j in range(VMS):
    cqm.add_constraint(quicksum(vm_status[i][j] for i in range(SERVERS)) == 1)

# (13) For each flow and server, the sum of exiting flow from the server to all adj switch is <= than vms part of that flow     
for f in range(FLOWS):
    for i in range(SWITCHES, SWITCHES + SERVERS):           # Start from switches cause nodes are numerated in order -> all switches -> all servers
        cqm.add_constraint(quicksum(flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(k)] for k in range(SWITCHES) if adjancy_list[i][k] == 1) - vm_status[i-SWITCHES][src_dst[f][0]] <= 0)

# (14) For each flow and server, the sum of entering flow from the server to all adj switch is <= than vms part of that flow     
for f in range(FLOWS):
    for i in range(SWITCHES, SWITCHES + SERVERS):
        cqm.add_constraint(quicksum(flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(i)] for k in range(SWITCHES) if adjancy_list[k][i] == 1) - vm_status[i-SWITCHES][src_dst[f][1]] <= 0) 

# (15) For each flow and server, force allocation of all flows     
for f in range(FLOWS):
    for i in range(SWITCHES, SWITCHES + SERVERS):
        cqm.add_constraint(vm_status[i-SWITCHES][src_dst[f][0]] - vm_status[i-SWITCHES][src_dst[f][1]]  -
                (quicksum(flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(k)] for k in range(SWITCHES) if adjancy_list[i][k] == 1) - 
                quicksum(flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(i)] for k in range(SWITCHES) if adjancy_list[k][i] == 1)) == 0)

# (16) For each switch and flow, entering and exiting flow from the switch are equal
for k in range(SWITCHES):
    for f in range(FLOWS):
        cqm.add_constraint(quicksum(flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(k)]  for n in range(NODES) if adjancy_list[n][k] == 1) - 
                quicksum(flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(n)] for n in range(NODES) if adjancy_list[k][n] == 1) == 0 )

# (17) For each link, the data rate on a it is less or equal than its capacity      
for l in range(LINKS):
    father = l//2       # Father node of the link
    son = l+1           # Son node of the link
    cqm.add_constraint(quicksum( data_rate[f][l] * (flow_path['f' + str(f) + '-n' + str(father) + '-n' + str(son)] + 
            flow_path['f' + str(f) + '-n' + str(son) + '-n' + str(father)]) for f in range(FLOWS)) - 
            link_capacity[l] * on["on" + str(father) + "-" + str(son)] <= 0)

# (18)(19) For each link, the link is ON only if both nodes are ON       
for l in range(LINKS):
    father = l//2           # Father node of the link
    son = l+1               # Son node of the link
    if son >= SWITCHES:     # Son is a Server
        cqm.add_constraint(on["on" + str(father) + "-" + str(son)] - switch_status[father] <= 0)
        cqm.add_constraint(on["on" + str(father) + "-" + str(son)] - server_status[son-SWITCHES] <= 0)
    else:                   # Both nodes are switches
        cqm.add_constraint(on["on" + str(father) + "-" + str(son)] - switch_status[father] <= 0)
        cqm.add_constraint(on["on" + str(father) + "-" + str(son)] - switch_status[son] <= 0)





print("SWITCH Indexes: ", *[k for k in range(SWITCHES)])
print("SERVER Indexes: ", *[s+SWITCHES for s in range(SERVERS)])

print("\n\n\n")
print("####################### CQM Model ###########################")
print("\n")

# Start execution timer
start_time = time.time()

# Create sampler
sampler = LeapHybridCQMSampler()

# Resolve problem, output (numpy array):
#   variable values
#   solution cost (energy consumption)
#   satisfied and unsatisfied constraints
#   if the solution is feasible
res = sampler.sample_cqm(cqm)

# Print execution time  |   TODO: dovrebbe essere dopo il sampler non dopo il filtraggio
print("Execution Time: %s" %(time.time() - start_time))


# Extract only solution that satisfy all constraints
feasible_sampleset = res.filter(lambda data_rate: data_rate.is_feasible)

# Extract best solution (minimal energy consumption)
best_sol = feasible_sampleset.first

# Extract variables values
dict = best_sol[0]
count = 0

# Iterate through variables set
for i in dict:
    if dict[i] > 0:
        # Data is flow_path
        if count == 0 and re.search("f.*", i) is not None:
            print("\n")
            print("--- Assegnamento Flow ---")
            count += 1

        # Data is on
        elif count == 1 and re.search("on.*", i) is not None:
            print("\n")
            print("--- Link attivi ---")
            count += 1
        
        # Data is active switches/servers
        elif count == 2 and re.search("s.*", i) is not None:
            print("\n")
            print("--- Switch / Server attivi ---")
            count += 1

        # Data is VMs distribution over servers
        elif count == 3 and re.search("v.*", i) is not None:
            print("\n")
            print("--- Assegnamento VM ---")
            count += 1
        
        # General printer
        print(i, end= " | ")

# Print Energy consumption 
print()
print()
print("ENERGY: " + str(best_sol[1]))



print("\n\n\n")
print("####################### BQM Model ###########################")
print("\n")
# Convert model from CQM to BQM
bqm, invert = cqm_to_bqm(cqm)

# Pre-processing to improve performance
roof_duality(bqm)

# Start Exection timer
start_time = time.time()

# Create sampler    | TODO: spostare sopra lo start time
sampler = EmbeddingComposite(DWaveSampler())

# Solve problem
sampleset = sampler.sample(bqm)

# Print execution time
print("Execution Time: %s" %(time.time() - start_time))

# Plotting
# dwave.inspector.show(sampleset)

# Extract best solution
best_sol = sampleset.first
print("ENERGY: " + str(best_sol[1]))

# Extract embedding info & print logic variables and qubit
embedding = sampleset.info['embedding_context']['embedding']
print(f"Numero di variabili logiche: {len(embedding.keys())}")
print(f"Numero di qubit fisici usati nell'embedding: {sum(len(chain) for chain in embedding.values())}")




print("\n\n\n")
print("####################### Ising Model ###########################")
print("\n")
# Convert from BQM to ising
h, j, offset = bqm.to_ising()

# Create sampler
sampler = EmbeddingComposite(DWaveSampler())

# Start Execution timer
start_time = time.time()

# Solve problem
res = sampler.sample_ising(h, j)

# Print Execution timer
print("Execution Time: %s" %(time.time() - start_time))

# Plotting
# dwave.inspector.show(res)
