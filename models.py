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
# D_WAVE
import dimod
import dwave.system
import dwave.inspector
import dwave.preprocessing

# OTHERS
import numpy as np
import random
import fun_lib as fn
import json

def vm_model(proxytree: fn.Proxytree, proxymanager: fn.Proxymanager, vm_cqm: dimod.ConstrainedQuadraticModel):
    '''
    Creates the vm assignment model as a Constrained Quadratic Model
    '''

    # Variables
    server_status = [dimod.Binary("s" + str(i)) for i in range(proxytree.SERVERS)]          # Binary value for each server, 1 ON, 0 OFF
    vm_status = [[dimod.Binary("vm" + str(i) + "-s" + str(j)) 
        for i in range(proxytree.VMS)] for j in range(proxytree.SERVERS)] 

    # Objective
    # 1 - SUM of server pow cons
    obj1 = dimod.quicksum(server_status[s] * proxytree.idle_powcons[s+proxytree.SWITCHES] for s in range(proxytree.SERVERS))
    # 2 - SUM of vm dyn pow cons
    obj2 = dimod.quicksum(proxytree.dyn_powcons[s+proxytree.SWITCHES] * dimod.quicksum(proxytree.cpu_util[vm] * vm_status[s][vm] for vm in range(proxytree.VMS)) for s in range(proxytree.SERVERS))
    # Total
    vm_cqm.set_objective(obj1 + obj2)

    # Constraints
    # (11) For each server, the CPU utilization of each VM on that server must be less or equal than server's capacity       
    for s in range(proxytree.SERVERS):
        vm_cqm.add_constraint(
            dimod.quicksum(
                proxytree.cpu_util[vm] * vm_status[s][vm] for vm in range(proxytree.VMS)) 
            - proxytree.server_capacity[s]*server_status[s] 
            <= 0, label="C11-N"+str(s))

    # (12) For each VM, it must be active on one and only one server
    for vm in range(proxytree.VMS):
        vm_cqm.add_constraint(
            dimod.quicksum(
                vm_status[s][vm] for s in range(proxytree.SERVERS)) 
            == 1, label="C12-N"+str(vm))

    return


def cqm_vm_solver(proxytree: fn.Proxytree, proxymanager: fn.Proxymanager, vm_cqm: dimod.ConstrainedQuadraticModel):
    '''
    Solves the vm assignment problem using a CQM Hybrid Solver.
    Returns a tuple containing the variable dictionary and the execution time
    '''
    
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

    # Save best solution
    if proxymanager.SAVE_DICT:
            with open("cqm_dict.txt", "w") as fp:
                json.dump(cqm_best[0], fp)
                print("CQM Dictionary updated!")

    return (cqm_best[0], cqm_time)


def bqm_vm_solver(proxytree: fn.Proxytree, proxymanager: fn.Proxymanager, vm_cqm: dimod.ConstrainedQuadraticModel, cqm_time):
    '''
    Solves the vm assignment problem using a CQM Hybrid Solver.
    Returns a tuple containing the variable dictionary and the execution time
    '''
    # From CQM to BQM
    vm_bqm, inverter = dimod.cqm_to_bqm(vm_cqm, lagrange_multiplier = proxymanager.LAGRANGE_MUL)

    # Roof Duality
    # rf_energy, rf_variables = dwave.preprocessing.roof_duality(vm_bqm)
    # print("Roof Duality variables: ", rf_variables)

    # Create Sampler
    bqm_sampler = dwave.system.LeapHybridSampler()

    # Sample Results
    bqm_samples = bqm_sampler.sample(vm_bqm, cqm_time // 10**6 * proxymanager.TIME_MULT1)

    # Exec Time
    bqm_time = bqm_samples.info.get('run_time')
    print("BQM TIME: ", bqm_time, " micros")

    # Extract feasible solution
    # bqm_feasibles = bqm_samples.filter(lambda sample: sample.is_feasible)

    # Extract best solution
    # bqm_best = bqm_feasibles.first
    bqm_best = bqm_samples.first
    inverted_sample = inverter(bqm_best.sample)

    # Energy
    print("BQM ENERGY: ", str(bqm_best[1]))
    # print("Roof Duality Energy: ", rf_energy)

    # Extract variables
    print("\n## BQM Variables ##")
    for i in bqm_best[0]:
        if bqm_best[0].get(i) != 0.0:
            print(i, bqm_best[0].get(i),sep = ": ",end= " | ")
    print("\n\n## Converted Variables ##")
    for i in inverted_sample:
        if inverted_sample.get(i) != 0.0:
            print(i, inverted_sample.get(i),sep = ": ",end= " | ")
    print("\n\nFeasible: ", vm_cqm.check_feasible(inverted_sample))



def path_model(proxytree: fn.Proxytree, proxymanager: fn.Proxymanager, path_cqm: dimod.ConstrainedQuadraticModel, cqm_solution):
    '''
    Creates the path planning model as a Constrained Quadratic Model
    '''

    switch_status = [dimod.Binary("sw" + str(i)) for i in range(proxytree.SWITCHES)]        # Binary value for each switch, 1 ON, 0 OFF

    # Initialize flow_path dictionary for each possible combination of flow and adjacent node
    # flow_path[f, [n1,n2]] = 1 if part of flow f goes from n1 to n2
    flow_path = {}
    for f in range(proxytree.FLOWS):
        for n1 in range(proxytree.NODES):
            for n2 in range(proxytree.NODES):
                if proxytree.adjancy_list[n1][n2]:     # Adjancy Condition (lowers variable number)
                    flow_path['f' + str(f) + '-n' + str(n1) + '-n' + str(n2)] = dimod.Binary("f" + str(f) + "-n" + str(n1) + "-n" + str(n2))

    # Initialize dictionary for each adjacent node
    # on[n1,n2] = 1 if the link between n1 and n2 is ON
    on = {}
    for n1 in range(proxytree.NODES):
        for n2 in range(n1, proxytree.NODES):
            if proxytree.adjancy_list[n1][n2]:         # Adjancy Condition (lowers variable number)
                on["on" + str(n1) + "-" + str(n2)] = dimod.Binary("on" + str(n1) + "-" + str(n2))


    # Load best solution from vm model
    if proxymanager.LOAD_DICT:
            with open("cqm_dict.txt") as fp:
                # for line in fp:
                    # cqm_best[0] = json.loads(fp)
                    # command, description = line.strip().split(None, 1)
                    # cqm_best[0][command] = description.strip()
                cqm_dict = json.loads(fp.read())
                print("CQM Dictionary loaded!")
                print(cqm_dict)
            cqm_best=(cqm_dict, cqm_best[1])
    else:
        cqm_best = cqm_solution

    # Objective
    # 3 - SUM of switch idle pow cons
    obj3 = dimod.quicksum(switch_status[sw] * proxytree.idle_powcons[sw] for sw in range(proxytree.SWITCHES))
    # 4 - SUM of flow path
    # obj4 = dimod.quicksum(flow_path['f' + str(f) + '-n' + str(n1) + '-n' + str(n2)] + flow_path['f' + str(f) + '-n' + str(n2) + '-n' + str(n1)]
                        # for n1 in range(NODES) for n2 in range(NODES) for f in range(FLOWS) if adjancy_list[n1][n2] == 1)
    obj4 = dimod.quicksum( proxytree.dyn_powcons[sw] * flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(sw)] + flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(n)]
                        for n in range(proxytree.NODES) for f in range(proxytree.FLOWS) for sw in range(proxytree.SWITCHES) if proxytree.adjancy_list[n][sw] == 1)
    # Total
    path_cqm.set_objective(obj3 + obj4)


    # Constraints
    # (13) For each flow and server, the sum of exiting flow from the server to all adj switch is <= than vms part of that flow     
    # for f in range(FLOWS):
    #     for s in range(SWITCHES, SWITCHES + SERVERS):           # Start from switches cause nodes are numerated in order -> all switches -> all servers
    #         path_cqm.add_constraint( 
    #             dimod.quicksum( 
    #                 flow_path['f' + str(f) + '-n' + str(s) + '-n' + str(sw)] for sw in range(SWITCHES) if adjancy_list[s][sw] == 1) 
    #                 - cqm_best[0].get("vm" + str(src_dst[f][0]) + "-s" + str(s-SWITCHES)) 
    #             <= 0, label="C13-N"+str(f*SERVERS+s))

    # # (14) For each flow and server, the sum of entering flow from the server to all adj switch is <= than vms part of that flow     
    # for f in range(FLOWS):
    #     for s in range(SWITCHES, SWITCHES + SERVERS):
    #         path_cqm.add_constraint( 
    #             dimod.quicksum( 
    #                 flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(s)] for sw in range(SWITCHES) if adjancy_list[sw][s] == 1) 
    #                 - cqm_best[0].get("vm" + str(src_dst[f][1]) + "-s" + str(s-SWITCHES)) 
    #             <= 0, label="C14-N"+str(f*SERVERS+s)) 

    # (15) For each flow and server, force allocation of all flows     
    for f in range(proxytree.FLOWS):
        for s in range(proxytree.SWITCHES, proxytree.SWITCHES + proxytree.SERVERS):
            path_cqm.add_constraint( 
                cqm_best[0].get("vm" + str(proxytree.src_dst[f][0]) + "-s" + str(s-proxytree.SWITCHES)) - cqm_best[0].get("vm" + str(proxytree.src_dst[f][1]) + "-s" + str(s-proxytree.SWITCHES))  
                - ( 
                    dimod.quicksum( 
                        flow_path['f' + str(f) + '-n' + str(s) + '-n' + str(sw)] for sw in range(proxytree.SWITCHES) if proxytree.adjancy_list[s][sw] == 1) 
                    - dimod.quicksum( 
                        flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(s)] for sw in range(proxytree.SWITCHES) if proxytree.adjancy_list[sw][s] == 1)) 
                == 0, label="C15-N"+str(f*proxytree.SERVERS+s))

    # (16) For each switch and flow, entering and exiting flow from the switch are equal
    for sw in range(proxytree.SWITCHES):
        for f in range(proxytree.FLOWS):
            path_cqm.add_constraint( 
                dimod.quicksum( 
                    flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(sw)]  for n in range(proxytree.NODES) if proxytree.adjancy_list[n][sw] == 1) 
                - dimod.quicksum( 
                    flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(n)] for n in range(proxytree.NODES) if proxytree.adjancy_list[sw][n] == 1) 
                == 0, label="C16-N"+str(sw*proxytree.FLOWS+f))

    # (17) For each link, the data rate on it is less or equal than its capacity      
    for l in range(proxytree.LINKS):
        n1,n2 = fn.get_nodes(l, proxytree.link_dict)
        path_cqm.add_constraint( 
            dimod.quicksum( 
                proxytree.data_rate[f] * (
                    flow_path['f' + str(f) + '-n' + str(n1) + '-n' + str(n2)] 
                    + flow_path['f' + str(f) + '-n' + str(n2) + '-n' + str(n1)]) for f in range(proxytree.FLOWS)) 
            - proxytree.link_capacity[l] * on["on" + str(n1) + "-" + str(n2)] 
            <= 0, label="C17-N"+str(l))

    # (18)(19) For each link, the link is ON only if both nodes are ON       
    for l in range(proxytree.LINKS):
        n1,n2 = fn.get_nodes(l, proxytree.link_dict)
        if n1 < proxytree.SWITCHES:
            path_cqm.add_constraint(
                on["on" + str(n1) + "-" + str(n2)] 
                - switch_status[n1] 
                <= 0, label="C18-N"+str(l))
        # else:
        #     path_cqm.add_constraint(
        #         on["on" + str(n1) + "-" + str(n2)] 
        #         - cqm_best[0].get("s"+str(n1-SWITCHES)) 
        #         <= 0, label="C18-N"+str(l))
        if n2 < proxytree.SWITCHES:
            path_cqm.add_constraint(
                on["on" + str(n1) + "-" + str(n2)] 
                - switch_status[n2] 
                <= 0, label="C19-N"+str(l))
        # else:
        #     path_cqm.add_constraint(
        #         on["on" + str(n1) + "-" + str(n2)] 
        #         - cqm_best[0].get("s"+str(n2-SWITCHES)) 
        #         <= 0, label="C19-N"+str(l))
    
    return


def cqm_path_solver(proxytree: fn.Proxytree, proxymanager: fn.Proxymanager, path_cqm: dimod.ConstrainedQuadraticModel):
    '''
    Solves the path planning problem using a CQM Hybrid Solver.
    Returns the time needed to solve the problem
    '''
    # Create Sampler
    cqm_sampler = dwave.system.LeapHybridCQMSampler()

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

    return cqm_time


def bqm_path_solver(proxytree: fn.Proxytree, proxymanager: fn.Proxymanager, path_cqm: dimod.BinaryQuadraticModel, cqm_time):
    '''
    Solves the path planning problem using a BQM Hybrid Solver.
    '''
    # From CQM to BQM
    path_bqm, inverter = dimod.cqm_to_bqm(path_cqm, lagrange_multiplier = proxymanager.LAGRANGE_MUL)

    # Roof Duality
    # rf_energy, rf_variables = dwave.preprocessing.roof_duality(vm_bqm)
    # print("Roof Duality variables: ", rf_variables)

    # Create Sampler
    bqm_sampler = dwave.system.LeapHybridSampler()
    
    # Sample Results
    bqm_samples = bqm_sampler.sample(path_bqm, cqm_time // 10**6 * proxymanager.TIME_MULT2)

    # Exec Time
    bqm_time = bqm_samples.info.get('run_time')
    print("BQM TIME: ", bqm_time, " micros")

    # Extract best solution
    bqm_best = bqm_samples.first
    inverted_sample = inverter(bqm_best.sample)

    # Energy
    print("BQM ENERGY: ", str(bqm_best[1]))
    # print("Roof Duality Energy: ", rf_energy)

    # Extract variables
    print("\n## BQM Variables ##")
    for i in bqm_best[0]:
        if bqm_best[0].get(i) != 0.0:
            print(i, bqm_best[0].get(i),sep = ": ",end= " | ")
    print("\n\n## Converted Variables ##")
    for i in inverted_sample:
        if inverted_sample.get(i) != 0.0:
            print(i, inverted_sample.get(i),sep = ": ",end= " | ")
    print("\n\nFeasible: ", path_cqm.check_feasible(inverted_sample))
    
    return