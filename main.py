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
server_capacity     1D list         C(s)            capacity of each server
idle_powcons        1D list         p(idle_i)       idle power consumption fo each node
dyn_powcons         1D list         p(dyn_i)        maximum dynamic power of each node 
adjancy_list        2D list         ---             node's adjancy list
link_capacity       1D list         C(l)            capacity of each link
src_dst             2D list         ---             list of server communicating through a path
server_status       1D list         ---             server status, 1 on, 0 off
switch_status       1D list         ---             switch status, 1 on, 0 off
vm_status           2D list         v(ji)           VM (j) status per server (i), 1 on, 0 off
cpu_util            2D array        u(v(ji)         CPU utilization of each VM v(ji)
data_rate           2D array        d(fl)           data rate of flow (f) on link (l)

flow_path           bin dictionary  ρ(f,(k,i))      se parte del flow (f) va da k a i (nodi), allora 1, 0 altrimenti
on                  bin dictionary
"""

"""
TODO List
> Create class
> Create main cycle
> Fix non even server problem and different-children problem
> Change variables name (paper like or program like)
"""

# IMPORTS
from dimod import ConstrainedQuadraticModel, Integer, QuadraticModel, Binary, quicksum, cqm_to_bqm 
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridCQMSampler
import dwave.inspector
from dwave.preprocessing import roof_duality
from scipy.stats import norm
import numpy as np
from random import seed
from random import randint
from datetime import datetime
import re


# Init random seed
seed()
DEBUG = True
DEPTH = 3   # Define constant depth




SERVERS = pow(2, DEPTH)                                 # Server number
VMS = SERVERS                                           # VM number per server
SWITCHES = sum(pow(2,i) for i in range(DEPTH))          # Switch number
FLOWS = SERVERS//2 if SERVERS%2==0 else SERVERS//2+1    # Flow number
LINKS = 2*SWITCHES                                      # Link number

server_capacity = [10 for i in range(SERVERS)]           # Capacity of each server
link_capacity = [10 for i in range (LINKS)]              # Capacity of each Link
idle_powcons = [10 for i in range(SERVERS + SWITCHES)]   # Idle power consumption of each node
dyn_powcons = [1 for i in range(SERVERS + SWITCHES)]     # Maximum dynamic power of each node


adjancy_list = [[0 for j in range(SERVERS + SWITCHES)] 
        for i in range(SERVERS + SWITCHES)]              # Binary list of adjacent nodes (0 non-andj, 1 adj)

server_status = [Binary("s" + str(i)) for i in range(SERVERS)]          # Binary value for each server, 1 ON, 0 OFF
switch_status = [Binary("sw" + str(i)) for i in range(SWITCHES)]        # Binary value for each switch, 1 ON, 0 OFF
vm_status = [[Binary("v" + str(j) + "-" + str(i)) 
        for i in range(SERVERS)] for j in range(VMS)]                   # Binary value for each VM on each server, 1 ON, 0 OFF

cpu_util = (np.random.normal(8, 1, (SERVERS, VMS))).astype(int)         # CPU utilization of each VM on each server
data_rate = (np.random.normal(4, 1, (FLOWS, LINKS))).astype(int)        # Data rate of flow f on link l 

# Calculate adjancy list:
#   > M                     index 0 of switches
#   > i*2 + 1               first son
#   > tmp & tmp + 1         both sons
#   > if                    switch - switch adjancy
#   > else                  switch - server adjancy
# for i in range(SWITCHES):
#     tmp = i*2 + 1 + SERVERS
#     if tmp < (SERVERS + SWITCHES):
#         adjancy_list[i + SERVERS][tmp] = 1
#         adjancy_list[tmp][i + SERVERS] = 1
#         adjancy_list[i + SERVERS][tmp + 1] = 1
#         adjancy_list[tmp + 1][i + SERVERS] = 1
#     else:
#         adjancy_list[i + SERVERS][tmp - (SERVERS + SWITCHES)] = 1
#         adjancy_list[tmp - (SERVERS + SWITCHES)][i + SERVERS] = 1
#         adjancy_list[i + SERVERS][tmp + 1 - (SERVERS + SWITCHES)] = 1
#         adjancy_list[tmp + 1 - (SERVERS + SWITCHES)][i + SERVERS] = 1

# Calculate adjancy list:
for i in range(SERVERS + SWITCHES):
    first_son = i*2 + 1
    second_son = first_son + 1
    if (second_son < SERVERS + SWITCHES):
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


src_dst = [[0 for j in range(2)] for i in range(FLOWS)]     # list of commmunicating servers
randoms = []                                                # list of generated random values
for i in range(FLOWS):
    for j in range(2):
        while True:            
            ran = randint(0, SERVERS-1)
            
            if not ran in randoms:
                # Fill list
                src_dst[i][j] = ran                
                # Keep number recorded
                randoms.append(ran)                
                break
print("Paths:", end=" ")
print(src_dst)


# Initialize flow_path dictionary for each possible combination of flow and adjacent node
# flow_path[f, [n1,n2]] = 1 if part of flow f goes from n1 to n2
flow_path = {}
for f in range(FLOWS):
    for i in range(SWITCHES + SERVERS):
        for k in range(SWITCHES + SERVERS):
            if adjancy_list[i][k]:
                flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(k)] = Binary("f" + str(f) + "-n" + str(i) + "-n" + str(k))

# Initialize dictionary for each adjacent node
on = {}
for i in range(SERVERS + SWITCHES):
    for j in range(SERVERS + SWITCHES):
        if adjancy_list[i][j]:
            on["on" + str(i) + "-" + str(j)] = Binary("on" + str(i) + "-" + str(j))



######################### CQM MODEL #########################
# Create CQM
cqm = ConstrainedQuadraticModel()


# OBJECTIVE
# Define Subobjectives
obj1 = quicksum(idle_powcons[i] * server_status[i] for i in range(SERVERS))
obj2 = quicksum(dyn_powcons[i] * quicksum(cpu_util[j][i] * vm_status[j][i] for j in range(VMS)) for i in range(SERVERS))
obj3 = quicksum(idle_powcons[i] * switch_status[i - SERVERS] for i in range(SERVERS, SERVERS + SWITCHES))
obj4 = quicksum(flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(j)] + flow_path['f' + str(f) + '-n' + str(j) + '-n' + str(i)]
                    for i in range(SERVERS + SWITCHES) for j in range (SERVERS + SWITCHES) for f in range(FLOWS) if adjancy_list[i][j] == 1)

# Set Objective
cqm.set_objective(obj1 + obj2 + obj3 + obj4)


# CONSTRAINTS
# For each server, the CPU utilization of each VM on that server must be less or equal than server's capacity       (11)
for i in range(SERVERS):
    cqm.add_constraint(quicksum(cpu_util[j][i] * vm_status[j][i] for j in range(VMS)) - server_capacity[i] * server_status[i] <= 0)

# For each VM, it can only be active on one server      (12)
for j in range(VMS):
    cqm.add_constraint(quicksum(vm_status[j][i] for i in range(SERVERS)) == 1)

# For each flow and server, ???     (13)
for f in range(FLOWS):
    for i in range(SERVERS):
        cqm.add_constraint(quicksum(flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(k)] for k in range(SERVERS, SERVERS + SWITCHES) if adjancy_list[i][k] == 1)  - vm_status[src_dst[f][0]][i] <= 0)

# For each flow and server, ???     (14)
for f in range(FLOWS):
    for i in range(SERVERS):
        cqm.add_constraint(quicksum(flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(i)] for k in range(SERVERS, SERVERS + SWITCHES) if adjancy_list[k][i] == 1) - vm_status[src_dst[f][1]][i] <= 0) 

# For each flow and server, ???     (15)
for f in range(FLOWS):
    for i in range(SERVERS):
        cqm.add_constraint(vm_status[src_dst[f][0]][i] - vm_status[src_dst[f][1]][i]  - (quicksum(flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(k)] for k in range(SERVERS, SWITCHES + SERVERS) if adjancy_list[i][k] == 1) - quicksum(flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(i)] for k in range(SERVERS, SWITCHES + SERVERS) if adjancy_list[k][i] == 1)) == 0)

# For each switch, ???      (16)
for k in range(SERVERS, SERVERS + SWITCHES):
    for f in range(FLOWS):
        cqm.add_constraint(quicksum(flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(k)]  for n in range(SERVERS + SWITCHES) if adjancy_list[n][k] == 1) - quicksum(flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(n)] for n in range(SERVERS + SWITCHES) if adjancy_list[k][n] == 1) == 0)

# For each node couple, the data rate on a path is less or equal than its capacity      (17)
count = 0
for i in range(SERVERS + SWITCHES):
    for j in range(SERVERS + SWITCHES):
        if adjancy_list[i][j] == 1 and j > i:
            cqm.add_constraint(quicksum(data_rate[f][count] * (flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(j)] + flow_path['f' + str(f) + '-n' + str(j) + '-n' + str(i)]) for f in range(FLOWS)) - link_capacity[count] * on["on" + str(i) + "-" + str(j)] <= 0)
            count += 1

# For each node couple, the data rate on a path is less or equal than its capacity      (17)
count = 0
for j in range(SERVERS + SWITCHES):
    for i in range(SERVERS + SWITCHES):
        if adjancy_list[i][j] == 1 and i > j:
            cqm.add_constraint(quicksum(data_rate[f][count] * (flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(j)] + flow_path['f' + str(f) + '-n' + str(j) + '-n' + str(i)]) for f in range(FLOWS)) - link_capacity[count] * on["on" + str(i) + "-" + str(j)] <= 0)
            count += 1

# For each node couple, if they are adjacent, ???       (18) (19)
for i in range(SERVERS + SWITCHES):
    for j in range(SERVERS + SWITCHES):
        if adjancy_list[i][j] == 1:
            if i < SERVERS:
                cqm.add_constraint(on["on" + str(i) + "-" + str(j)] - server_status[i] <= 0)
            else:
                cqm.add_constraint(on["on" + str(i) + "-" + str(j)] - switch_status[i - SERVERS] <= 0)
            if j < SERVERS:
                cqm.add_constraint(on["on" + str(i) + "-" + str(j)] - server_status[j] <= 0)
            else:
                cqm.add_constraint(on["on" + str(i) + "-" + str(j)] - switch_status[j - SERVERS] <= 0)


print("\n\n\n")
print("####################### CQM Model ###########################")
print("\n")
# Start execution timer
import time
start_time = time.time()

# Create sampler
sampler = LeapHybridCQMSampler()

# Resolve problem, output (numpy array):
#   variable values
#   solution cost (energy consumption)
#   satisfied and unsatisfied constraints
#   if the solution is feasible
res = sampler.sample_cqm(cqm)

# Extract only solution that satisfy all constraints
feasible_sampleset = res.filter(lambda data_rate: data_rate.is_feasible)

# Extract best solution (minimal energy consumption)
best_sol = feasible_sampleset.first

# Print execution time  |   TODO: dovrebbe essere dopo il sampler non dopo il filtraggio
print("time: %s" %(time.time() - start_time))

# Extract variables values
dict = best_sol[0]
count = 0
print("Gli indici da 0 a " + str(SERVERS - 1) + " sono i server")
print("Gli indici da " + str(SERVERS) + " a " + str(SERVERS + SWITCHES - 1) + " sono gli switch")

# Iterate through variables set
for i in dict:
    if dict[i] > 0:
        # Data is on
        if count == 0 and re.search("on.*", i) is not None:
            print("Collegamenti attivi: ")
            count += 1
        
        # Data is flow_path
        elif count == 1 and re.search("flow_path.*", i) is not None:
            print("flow_path[f, [n1, n2]] = 1 se parte del flusso f-esimo va da n1 ad n2")
            count += 1
        
        # Data is active switches/servers
        elif count == 2 and re.search("s.*", i) is not None:
            print("Switch/server attivi")
            count += 1

        # Data is VMs distribution over servers
        elif count == 3 and re.search("v.*", i) is not None:
            print("v[j, i]: la j-esima macchina virtuale sul i-esimo server")
            count += 1
        
        # General printer
        print(i)

# Print Energy consumption 
print("Energia: " + str(best_sol[1]))



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

# Plotting
# dwave.inspector.show(sampleset)

# Extract best solution
best_sol = sampleset.first


# Print execution time
print("time: %s" %(time.time() - start_time))

# Extract embedding info
embedding = sampleset.info['embedding_context']['embedding']
# Print num of logic variables and qubit used in the embedding
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
print("time: %s" %(time.time() - start_time))

# Plotting
# dwave.inspector.show(res)
