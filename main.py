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


# IMPORTS D-WAVE
from dimod import ConstrainedQuadraticModel, Integer, QuadraticModel, Binary, quicksum, cqm_to_bqm 
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridCQMSampler, LeapHybridBQMSampler
import dwave.inspector
from dwave.preprocessing import roof_duality

# IMPORTS OTHERS
from scipy.stats import norm
import numpy as np
from random import seed, randint
import time
import re
import json

seed()                  # Init random seed
DEBUG = True            # DEBUG BOOLEAN
SAVE_DICT = True
LOAD_DICT = True
ITERATIONS = 1          # Iterations to compare results
TIME_MULT = 1

################# TREE DEFINITION PARAMETERS ###################
DEPTH = 3               # Tree depth
SERVER_C = 10           # Server capacity
LINK_C = 10             # Link capacity
IDLE_PC = 10            # Idle power consumption
DYN_PC = 1              # Dynamic power consumption
REQ_AVG = 8             # Average flow request
DATAR_AVG = 4           # Average data rate per flow
###############################################################


############### TREE CONSTRUCTION ####################################################################

SERVERS = pow(2, DEPTH)                                 # Server number
VMS = SERVERS                                           # VM number per server
SWITCHES = sum(pow(2,i) for i in range(DEPTH))          # Switch number
FLOWS = VMS//2 if VMS%2==0 else VMS//2+1                # Flow number
LINKS = 2*SWITCHES                                      # Link number
NODES = SERVERS + SWITCHES                              # Total Nodes

server_capacity = [SERVER_C for i in range(SERVERS)]           # Capacity of each server
link_capacity = [LINK_C for i in range (LINKS)]              # Capacity of each Link
idle_powcons = [IDLE_PC for i in range(NODES)]   # Idle power consumption of each node
dyn_powcons = [DYN_PC for i in range(NODES)]     # Maximum dynamic power of each node


adjancy_list = [[0 for j in range(NODES)] 
        for i in range(NODES)]              # Binary list of adjacent nodes (0 non-andj, 1 adj)


cpu_util = (np.random.normal(REQ_AVG, 1, VMS)).astype(int)         # CPU utilization of each VM
data_rate = (np.random.normal(DATAR_AVG, 1, (FLOWS, LINKS))).astype(int)      # Data rate of flow f on link l 
if DEBUG:
	print("\n### CPU Utilization ###\n", cpu_util)
	print("\n### Data Rate ###\n", data_rate)

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



############### BINARY VARIABLES ################################################################################################

server_status = [Binary("s" + str(i)) for i in range(SERVERS)]          # Binary value for each server, 1 ON, 0 OFF
switch_status = [Binary("sw" + str(i)) for i in range(SWITCHES)]        # Binary value for each switch, 1 ON, 0 OFF
vm_status = [[Binary("vm" + str(i) + "-s" + str(j)) 
        for i in range(VMS)] for j in range(SERVERS)]                   # Binary value for each VM on each server, 1 ON, 0 OFF

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



######################### MODEL (CQM) ################################################################################################
# Create CQM
cqm = ConstrainedQuadraticModel()


# OBJECTIVE
# Define Subobjectives
obj1 = quicksum(server_status[i] * idle_powcons[i+SWITCHES] for i in range(SERVERS))
obj2 = quicksum(dyn_powcons[i+SWITCHES] * quicksum(cpu_util[j] * vm_status[i][j] for j in range(VMS)) for i in range(SERVERS))
obj3 = quicksum(switch_status[i] * idle_powcons[i] for i in range(SWITCHES))

obj4 = quicksum(flow_path['f' + str(f) + '-n' + str(i) + '-n' + str(j)] + flow_path['f' + str(f) + '-n' + str(j) + '-n' + str(i)]
                    for i in range(NODES) for j in range (NODES) for f in range(FLOWS) if adjancy_list[i][j] == 1)
# obj4 = quicksum(flow_path['f' + str(f) + '-n' + str(k) + '-n' + str(n)] + flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(k)] 
                    # for k in range(SWITCHES) for n in range(NODES) for f in range(FLOWS) if adjancy_list[k][n] == 1)

# Set Objective
cqm.set_objective(obj1 + obj2 + obj3 + obj4)
#cqm.set_objective(obj4)


# CONSTRAINTS
# (11) For each server, the CPU utilization of each VM on that server must be less or equal than server's capacity       
for i in range(SERVERS):
    cqm.add_constraint(quicksum(cpu_util[j] * vm_status[i][j] for j in range(VMS)) - server_capacity[i]*server_status[i] <= 0)

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

avg_time = 0
avg_energy = 0
for i in range(ITERATIONS):

    print("\n\n\n")
    print("####################### CQM Model ###########################")
    print("\n")

    # Create sampler
    cqm_sampler = LeapHybridCQMSampler()

    # Resolve problem, output (numpy array):
    #   variable values
    #   solution cost (energy consumption)
    #   satisfied and unsatisfied constraints
    #   if the solution is feasible
    cqm_res = cqm_sampler.sample_cqm(cqm)


    # Print execution time
    cqm_time = cqm_res.info.get('run_time')
    print("CQM Execution Time: ", cqm_time, " micros")


    # Extract only solution that satisfy all constraints
    cqm_feasible_sampleset = cqm_res.filter(lambda data_rate: data_rate.is_feasible)

    # Extract best solution (minimal energy consumption)
    cqm_best_sol = cqm_feasible_sampleset.first

    # Print Energy consumption 
    print("ENERGY: " + str(cqm_best_sol[1]))

    # Extract variables values
    cqm_dict = cqm_best_sol[0]
    count = 0

    if SAVE_DICT:
        with open("cqm_dict.txt", "w") as fp:
            json.dump(cqm_dict, fp)
            print("CQM Dictionary updated!")

    # Iterate through variables set
    for i in cqm_dict:
        if cqm_dict[i] > 0:
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
            print(i, cqm_dict.get(i),sep = ": ",end= " | ")



    print("\n\n\n")
    print("####################### BQM Model ###########################")
    print("\n")
    # Convert model from CQM to BQM
    #   inverter is a function that converts samples over the binary quadratic model back into samples 
    #   for the constrained quadratic model.
    bqm, inverter = cqm_to_bqm(cqm)

    # Pre-processing to improve performance
    roof_duality(bqm)

    # Create sampler
    bqm_sampler = LeapHybridBQMSampler()

    # Solve problem
    bqm_res = bqm_sampler.sample(bqm, time_limit = (cqm_time // (10**6))*TIME_MULT)

    # Print execution time
    bqm_time = bqm_res.info.get('run_time')
    print("BQM Execution Time: ", bqm_time, " micros")

    # Extract only solution that satisfy all constraints
    # bqm_feasible_sampleset = bqm_res.filter(lambda data_rate: data_rate.is_feasible)

    # Extract best solution
    bqm_best_sol = bqm_res.first
    # bqm_best_sol = bqm_feasible_sampleset.first
    print("ENERGY: " + str(bqm_best_sol[1]))

    # Extract variables values
    bqm_dict = bqm_best_sol[0]
    count = 0

    # if SAVE_DICT:
    #     with open("bqm_dict.txt", "w") as fp:
    #         json.dump(bqm_dict, fp)
    #         print("BQM Dictionary updated!")

    # Iterate through variables set
    for i in bqm_dict:
        if bqm_dict[i] > 0:
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
            print(i, bqm_dict.get(i),sep = ": ",end= " | ")



    # Single Time & Energy Difference
    avg_time += cqm_time - bqm_time
    avg_energy += cqm_best_sol[1] - bqm_best_sol[1]
    print("\n\n\n")
    print("####################### Time & Energy Difference (CQM - BQM) ###########################")
    print("Time difference: ", cqm_time - bqm_time, " micros")
    print("Energy difference: ", cqm_best_sol[1] - bqm_best_sol[1])


# Average Time & Energy Difference
print("\n\n\n")
print("####################### AVERAGE Time & Energy Difference (CQM - BQM) ###########################")
print("Average Time difference: ", avg_time / ITERATIONS, " micros")
print("Average Energy difference: ", avg_energy / ITERATIONS)


# print("\n\n\n")
# print("####################### Ising Model ###########################")
# print("\n")
# # Convert from BQM to ising
# h, j, offset = bqm.to_ising()

# # Create sampler
# isi_sampler = EmbeddingComposite(DWaveSampler())

# # Start Execution timer
# start_time = time.time()

# # Solve problem
# res = isi_sampler.sample_ising(h, j)

# # Print Execution timer
# print("Execution Time: %s" %(time.time() - start_time))

# Plotting
# dwave.inspector.show(res)
