"""
MODEL VARIABLES DICTIONARY
    M           int         number of servers
    N           int         number of VM
    K           int         number of switches
    F           int         number of flows (server-server paths, = M/2)
    L           int         number of links (graph links)

    Cs          1D list     capacity of each server
    pi_idle     1D list     idle power consumption fo each node
    pi_dyn      1D list     maximum dynamic power of each node 
    adj_node    2D list     node's adjancy list
    C           1D list     capacity of each link
    src_dst     2D list     list of server communicating through a path

    si          1D list     server status, 1 on, 0 off
    swk         1D list     switch status, 1 on, 0 off
    v           2D list     VM status per server, 1 on, 0 off
    u_v         2D array    CPU utilization of each VM
    d           2D array    data rate of flow f on link l

    rho         dictionary  
    on          dictionary
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

u = []      # CPU utilization of VM j on server i
d = []      # Data rate of flow f on link l

depth = 3   # Define constant depth


M = pow(2, depth)                       # Server number
N = M                                   # VM number per server
K = sum(pow(2,i) for i in range(depth)) # Switch number
F = M//2 if M%2==0 else M//2+1          # Flow number

Cs = [10 for i in range(M)]             # Capacity of each server
pi_idle = [10 for i in range(M + K)]    # Idle power consumption of each node
pi_dyn = [1 for i in range(M + K)]      # Maximum dynamic power of each node
L = 2*K                                 # Link number

# Binary list of adjacent nodes (0 non-andj, 1 adj)
adj_node = [[0 for j in range(M + K)] for i in range(M + K)]    
# Calculate adjancy list:
#   > M                     index 0 of switches
#   > i*2 + 1               first son
#   > tmp & tmp + 1         both sons
#   > if                    switch - switch adjancy
#   > else                  switch - server adjancy
for i in range(K):
    tmp = i*2 + 1 + M
    if tmp < (M + K):
        adj_node[i + M][tmp] = 1
        adj_node[tmp][i + M] = 1
        adj_node[i + M][tmp + 1] = 1
        adj_node[tmp + 1][i + M] = 1
    else:
        adj_node[i + M][tmp - (M + K)] = 1
        adj_node[tmp - (M + K)][i + M] = 1
        adj_node[i + M][tmp + 1 - (M + K)] = 1
        adj_node[tmp + 1 - (M + K)][i + M] = 1

Cl = [10 for i in range (L)]                     # Capacity of each Link


# Fill src_dst with all possible server ids to couple
src_dst = [[0 for j in range(2)] for i in range(F)] # list of commmunicating servers
randoms = []                                                # list of generated random values
for i in range(F):
    for j in range(2):
        while True:
            
            ran = randint(0, M-1)
            
            if not ran in randoms:
                # Fill list
                src_dst[i][j] = ran
                
                # Keep number recorded
                randoms.append(ran)
                
                break
# Print Paths
print("Paths")
print(src_dst)


si = [Binary("s" + str(i)) for i in range(M)]       # Binary value for each server, 1 means on, 0 off
swk = [Binary("sw" + str(i)) for i in range(K)]     # Binary value for each switch, 1 means on, 0 off

v = [[Binary("v" + str(j) + "-" + str(i)) for i in range(M)] for j in range(N)]  # Binary value for each VM on each server, 1 means on, 0 off
u_v = np.random.normal(8, 1, (M, N))                # CPU utilization of each VM on each server
u_v = u_v.astype(int)

# Initialize rho dictionary for each adjacent node
# rho[f, [n1,n2]] = 1 if part of flow f goes from n1 to n2
rho = {}
for f in range(F):
    for i in range(K + M):
        for k in range(K + M):
            if adj_node[i][k]:
                rho['rho' + str(f) + '-' + str(i) + '-' + str(k)] = Binary("rho" + str(f) + "-" + str(i) + "-" + str(k))


d = np.random.normal(4, 1, (F, L))          # Data rate of flow f on link l 
d = d.astype(int)

# Initialize dictionary for each adjacent node
on = {}
for i in range(M + K):
    for j in range(M + K):
        if adj_node[i][j]:
            on["on" + str(i) + "-" + str(j)] = Binary("on" + str(i) + "-" + str(j))



######################### CQM MODEL #########################
# Create CQM
cqm = ConstrainedQuadraticModel()


# OBJECTIVE
# Define Subobjectives
obj1 = quicksum(pi_idle[i] * si[i] for i in range(M))
obj2 = quicksum(pi_dyn[i] * quicksum(u_v[j][i] * v[j][i] for j in range(N)) for i in range(M))
obj3 = quicksum(pi_idle[i] * swk[i - M] for i in range(M, M + K))
obj4 = quicksum(rho['rho' + str(f) + '-' + str(i) + '-' + str(j)] + rho['rho' + str(f) + '-' + str(j) + '-' + str(i)]
                    for i in range(M + K) for j in range (M + K) for f in range(F) if adj_node[i][j] == 1)

# Set Objective
cqm.set_objective(obj1 + obj2 + obj3 + obj4)


# CONSTRAINTS
# For each server, the CPU utilization of each VM on that server must be less or equal than server's capacity       (11)
for i in range(M):
    cqm.add_constraint(quicksum(u_v[j][i] * v[j][i] for j in range(N)) - Cs[i] * si[i] <= 0)

# For each VM, it can only be active on one server      (12)
for j in range(N):
    cqm.add_constraint(quicksum(v[j][i] for i in range(M)) == 1)

# For each flow and server, ???     (13)
for f in range(F):
    for i in range(M):
        cqm.add_constraint(quicksum(rho['rho' + str(f) + '-' + str(i) + '-' + str(k)] for k in range(M, M + K) if adj_node[i][k] == 1)  - v[src_dst[f][0]][i] <= 0)

# For each flow and server, ???     (14)
for f in range(F):
    for i in range(M):
        cqm.add_constraint(quicksum(rho['rho' + str(f) + '-' + str(k) + '-' + str(i)] for k in range(M, M + K) if adj_node[k][i] == 1) - v[src_dst[f][1]][i] <= 0) 

# For each flow and server, ???     (15)
for f in range(F):
    for i in range(M):
        cqm.add_constraint(v[src_dst[f][0]][i] - v[src_dst[f][1]][i]  - (quicksum(rho['rho' + str(f) + '-' + str(i) + '-' + str(k)] for k in range(M, K + M) if adj_node[i][k] == 1) - quicksum(rho['rho' + str(f) + '-' + str(k) + '-' + str(i)] for k in range(M, K + M) if adj_node[k][i] == 1)) == 0)

# For each switch, ???      (16)
for k in range(M, M + K):
    for f in range(F):
        cqm.add_constraint(quicksum(rho['rho' + str(f) + '-' + str(n) + '-' + str(k)]  for n in range(M + K) if adj_node[n][k] == 1) - quicksum(rho['rho' + str(f) + '-' + str(k) + '-' + str(n)] for n in range(M + K) if adj_node[k][n] == 1) == 0)

# For each node couple, the data rate on a path is less or equal than its capacity      (17)
count = 0
for i in range(M + K):
    for j in range(M + K):
        if adj_node[i][j] == 1 and j > i:
            cqm.add_constraint(quicksum(d[f][count] * (rho['rho' + str(f) + '-' + str(i) + '-' + str(j)] + rho['rho' + str(f) + '-' + str(j) + '-' + str(i)]) for f in range(F)) - Cl[count] * on["on" + str(i) + "-" + str(j)] <= 0)
            count += 1

# For each node couple, the data rate on a path is less or equal than its capacity      (17)
count = 0
for j in range(M + K):
    for i in range(M + K):
        if adj_node[i][j] == 1 and i > j:
            cqm.add_constraint(quicksum(d[f][count] * (rho['rho' + str(f) + '-' + str(i) + '-' + str(j)] + rho['rho' + str(f) + '-' + str(j) + '-' + str(i)]) for f in range(F)) - Cl[count] * on["on" + str(i) + "-" + str(j)] <= 0)
            count += 1

# For each node couple, if they are adjacent, ???       (18) (19)
for i in range(M + K):
    for j in range(M + K):
        if adj_node[i][j] == 1:
            if i < M:
                cqm.add_constraint(on["on" + str(i) + "-" + str(j)] - si[i] <= 0)
            else:
                cqm.add_constraint(on["on" + str(i) + "-" + str(j)] - swk[i - M] <= 0)
            if j < M:
                cqm.add_constraint(on["on" + str(i) + "-" + str(j)] - si[j] <= 0)
            else:
                cqm.add_constraint(on["on" + str(i) + "-" + str(j)] - swk[j - M] <= 0)


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
feasible_sampleset = res.filter(lambda d: d.is_feasible)

# Extract best solution (minimal energy consumption)
best_sol = feasible_sampleset.first

# Print execution time  |   TODO: dovrebbe essere dopo il sampler non dopo il filtraggio
print("time: %s" %(time.time() - start_time))

# Extract variables values
dict = best_sol[0]
count = 0
print("Gli indici da 0 a " + str(M - 1) + " sono i server")
print("Gli indici da " + str(M) + " a " + str(M + K - 1) + " sono gli switch")

# Iterate through variables set
for i in dict:
    if dict[i] > 0:
        # Data is on
        if count == 0 and re.search("on.*", i) is not None:
            print("Collegamenti attivi: ")
            count += 1
        
        # Data is rho
        elif count == 1 and re.search("rho.*", i) is not None:
            print("rho[f, [n1, n2]] = 1 se parte del flusso f-esimo va da n1 ad n2")
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
