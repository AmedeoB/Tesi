DEBUG = False

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
import hybrid
# import dwave.inspector
# import dwave.preprocessing

# OTHERS
import fun_lib as fn
import json


def check_bqm_feasible(bqm_solution: dict, cqm_model: dimod.ConstrainedQuadraticModel, 
            inverter: dimod.constrained.CQMToBQMInverter):
    '''
    Checks if given sampleset is feasible for the given CQM.

    Args:
        - bqm_solution: the sampleset to check 
            type: sampleset
        - cqm_model: the CQM for checking 
            type: ConstrainedQuadraticModel
        - inverter: the converter from the BQM to the CQM 
            type: CQMToBQMInverter
    
    Returns:
        - null
    '''
    inverted_sample = inverter(bqm_solution)
    
    print("\n\n## Converted Variables ##")

    last_char = ""
    for var, value in inverted_sample.items():
        if last_char != var[0]:         # Var separator
            print(end="\n")
        if value != 0.0:                # Nonzero var printer
            print(var, value, sep = ": ",end= " | ")
        last_char = var[0]          # Update last char to separate vars
    
    print("\n\nFeasible: ", cqm_model.check_feasible(inverted_sample))



def cqm_solver(cqm_problem: dimod.ConstrainedQuadraticModel, problem_label: str, 
            save = False):
    '''
    Solves the CQM problem using a CQM Hybrid Solver and returns
    the results.

    Args:
        - cqm_problem (ConstrainedQuadraticModel): the BQM to 
        solve
        - problem_label (str): the problem label
        - save (bool, optional, default=False): save option
        for the dictionary

    Returns:
        - Tuple: containing the solution's dictionary
        and its execution time
            type: tuple(dict(), int)
    '''
    
    # Create Sampler
    sampler = dwave.system.LeapHybridCQMSampler()

    # Sample results
    sampleset = sampler.sample_cqm(cqm_problem, label = problem_label)

    # Exec time
    exec_time = sampleset.info.get('run_time')
    print("CQM TIME: ", exec_time, " micros")

    # Extract feasible solution
    feasible_sampleset = sampleset.filter(lambda sample: sample.is_feasible)

    # Extract best solution and energy
    best_solution = feasible_sampleset.first[0]
    energy = feasible_sampleset.first[1]

    # Energy
    print("CQM ENERGY: ", str(energy))

    # Extract variables
    print("\n## CQM Variables ##")
    last_char = ""
    for var, value in best_solution.items():
        if last_char != var[0]:         # Var separator
            print(end="\n")
        if value != 0.0:                # Nonzero var printer
            print(var, value, sep = ": ",end= " | ")
        last_char = var[0]          # Update last char to separate vars
    print(end= "\n")
    # Save best solution
    if save:
        with open(("cqm_dict_"+problem_label+".txt"), "w") as fp:
            json.dump(best_solution, fp)
            print(problem_label+" dictionary updated!")

    return (best_solution, exec_time)



def bqm_solver(bqm_problem: dimod.BinaryQuadraticModel, problem_label: str, 
            cqm_time = 0, time_mult = 1):
    '''
    Solves the BQM problem using decomposition and returns
    the result.

    Args:
        - bqm_problem (BinaryQuadraticModel): the BQM to 
        solve
        - problem_label (str): the problem label
        - cqm_time (int, optional, default=0): cqm time
        to compute custom resolve time
        - time_mult (int, optional, default=1): custom
        time multiplier for resolve time

    Returns:
        - best_solution: the solution's dictionary
            type: dict()
    '''   
    # Roof Duality
    # rf_energy, rf_variables = dwave.preprocessing.roof_duality(vm_bqm)
    # print("Roof Duality variables: ", rf_variables)

    # Create Sampler
    sampler = dwave.system.LeapHybridSampler()

    # Sample Results
    if cqm_time:
        sampleset = sampler.sample(bqm_problem, cqm_time//10**6 *time_mult, label = problem_label)
    else:
        sampleset = sampler.sample(bqm_problem, label = problem_label)

    # Exec Time
    exec_time = sampleset.info.get('run_time')
    print("BQM TIME: ", exec_time, " micros")

    # Extract best solution & energy
    best_solution = sampleset.first[0]
    energy = sampleset.first[1]

    # Energy
    print("BQM ENERGY: ", energy)
    # print("Roof Duality Energy: ", rf_energy)

    # Extract variables
    print("\n## BQM Variables ##")
    last_char = ""
    for var, value in best_solution.items():
        if last_char != var[0]:         # Var separator
            print(end="\n")
        if value != 0.0:                # Nonzero var printer
            print(var, value, sep = ": ",end= " | ")
        last_char = var[0]          # Update last char to separate vars
    print(end= "\n")
    
    return best_solution



def merge_substates(_, substates):
    '''
    Minimal function to merge substates in a multiple
    substates per cycle environment
    '''

    a, b = substates
    return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))



def decomposed_solver(bqm_problem: dimod.BinaryQuadraticModel, problem_label: str):
    '''
    Solves the BQM problem using decomposition and returns
    the result.

    Args:
        - bqm_problem (BinaryQuadraticModel): the BQM to 
        solve
        - problem_label (str): the problem label
        - cqm_time (int, optional, default=0): cqm time
        to compute custom resolve time
        - time_mult (int, optional, default=1): custom
        time multiplier for resolve time

    Returns:
        - best_solution: the solution's dictionary
            type: dict()
    '''
    # Decomposer
    decomposer = hybrid.ComponentDecomposer()
    decomposer_random = hybrid.RandomSubproblemDecomposer(size= 10)
    # decomposer = hybrid.Unwind( 
    #                 hybrid.SublatticeDecomposer()
    #             )

    # Subsampler
    qpu = dwave.system.DWaveSampler()
    subsampler = hybrid.QPUSubproblemAutoEmbeddingSampler(
                    qpu_sampler=qpu
    )
    # subsampler = hybrid.Map(
    #                     hybrid.QPUSubproblemAutoEmbeddingSampler(
    #                         qpu_sampler= qpu,
    #                     )
    #             ) | hybrid.Reduce (
    #                     hybrid.Lambda(merge_substates)
    #             )
    
    # Composer
    composer = hybrid.SplatComposer()
    
    # Parallel solvers
    classic_branch = hybrid.InterruptableTabuSampler() 
    
    # Merger
    merger = hybrid.ArgMin()
    # merger = hybrid.GreedyPathMerge()    

    # Branch
    qpu_branch = (decomposer | subsampler | composer) | hybrid.TrackMin(output= True)
    random_branch = (decomposer_random | subsampler | composer) | hybrid.TrackMin(output= True)
    parallel_branches = hybrid.RacingBranches(
                    classic_branch, 
                    qpu_branch,
                    random_branch,
                    ) | merger

    # Define workflow
    workflow = hybrid.LoopUntilNoImprovement(
                        parallel_branches, 
                        # convergence= 3, 
                        # max_iter= 5, 
                        max_time= 3,
                        )

    # Solve
    origin_embeddings = hybrid.make_origin_embeddings(qpu, )
    init_state = hybrid.State.from_sample(
                        hybrid.random_sample(bqm_problem), 
                        bqm_problem,
                        origin_embeddings= origin_embeddings)
    solution = workflow.run(init_state).result()

    # Print timers
    # hybrid.print_counters(workflow)

    # Extract best solution & energy
    best_solution = solution.samples.first[0]
    energy = solution.samples.first[1]

    # Energy
    print("Decomposer BQM ENERGY: ", energy)

    # Extract variables
    print("\n## Decomposer BQM Variables ##")
    last_char = ""
    for var, value in best_solution.items():
        if last_char != var[0]:         # Var separator
            print(end="\n")
        if value != 0.0:                # Nonzero var printer
            print(var, value, sep = ": ",end= " | ")
        last_char = var[0]          # Update last char to separate vars
    print(end= "\n")
    
    # Extract infos
    print("\n\n## Decomposer BQM Extra Info ##")
    print(solution.info)
    # print(solution)
    

    return best_solution
    


def vm_model(proxytree: fn.Proxytree, cqm: dimod.ConstrainedQuadraticModel):
    '''
    Creates the vm assignment model as a Constrained Quadratic Model

    Args:
        - proxytree: the tree structure to generate the model
            type: fn.Proxytree
        - vm_cqm: the CQM to fill
            type: dimod.ConstrainedQuadraticModel
    
    Returns:
        - null
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
    cqm.set_objective(obj1 + obj2)

    # Constraints
    # (11) For each server, the CPU utilization of each VM on that server must be less or equal than server's capacity       
    for s in range(proxytree.SERVERS):
        cqm.add_constraint(
            dimod.quicksum(
                proxytree.cpu_util[vm] * vm_status[s][vm] for vm in range(proxytree.VMS)) 
            - proxytree.server_capacity[s]*server_status[s] 
            <= 0, label="C11-N"+str(s))

    # (12) For each VM, it must be active on one and only one server
    for vm in range(proxytree.VMS):
        cqm.add_constraint(
            dimod.quicksum(
                vm_status[s][vm] for s in range(proxytree.SERVERS)) 
            == 1, label="C12-N"+str(vm))
    



def path_model(proxytree: fn.Proxytree, cqm: dimod.ConstrainedQuadraticModel, 
            vm_solution = {}, load = False):
    '''
    Creates the path planning model as a Constrained Quadratic Model
    
    Args:
        - proxytree: the tree structure to generate the model
            type: fn.Proxytree
        - path_cqm: the CQM to fill
            type: dimod.ConstrainedQuadraticModel
        - cqm_solution: the previous VM assignment solution
            type: tuple(dict, int)
        - load: boolean var for loading saved results in
        cqm_dict_vm_model.txt
            type: bool
    
    Returns:
        - null
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


    # Load best solution from file
    if load:
            with open("cqm_dict_vm_model.txt") as fp:
                vm_solution = json.loads(fp.read())
                print("VM-CQM Dictionary loaded!")
                print(vm_solution)
                print("\n\n")


    # Objective
    # 3 - SUM of switch idle pow cons
    obj3 = dimod.quicksum(switch_status[sw] * proxytree.idle_powcons[sw] for sw in range(proxytree.SWITCHES))
    # 4 - SUM of flow path
    # obj4 = dimod.quicksum(flow_path['f' + str(f) + '-n' + str(n1) + '-n' + str(n2)] + flow_path['f' + str(f) + '-n' + str(n2) + '-n' + str(n1)]
                        # for n1 in range(NODES) for n2 in range(NODES) for f in range(FLOWS) if adjancy_list[n1][n2] == 1)
    obj4 = dimod.quicksum( proxytree.dyn_powcons[sw] * flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(sw)] + flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(n)]
                        for n in range(proxytree.NODES) for f in range(proxytree.FLOWS) for sw in range(proxytree.SWITCHES) if proxytree.adjancy_list[n][sw] == 1)
    # Total
    cqm.set_objective(obj3 + obj4)


    # Constraints
    # (13) For each flow and server, the sum of exiting flow from the server to all adj switch is = than vms part of that flow
    # This is the splitted version, the combined one should 
    #   > remove the if condition 
    #   > change to <= type constraint
    #   > add the code line "- cqm_solution.get("vm" + str(proxytree.src_dst[f][0]) + "-s" + str(s-proxytree.SWITCHES))"
    for f in range(proxytree.FLOWS):
        for s in range(proxytree.SWITCHES, proxytree.SWITCHES + proxytree.SERVERS):           # Start from switches cause nodes are numerated in order -> all switches -> all servers
            if vm_solution.get("vm" + str(proxytree.src_dst[f][0]) + "-s" + str(s-proxytree.SWITCHES)) == 0:
                cqm.add_constraint( 
                    dimod.quicksum( 
                        flow_path['f' + str(f) + '-n' + str(s) + '-n' + str(sw)] for sw in range(proxytree.SWITCHES) if proxytree.adjancy_list[s][sw] == 1)
                    == 0, label="C13-N"+str(f*proxytree.SERVERS+s))

    # (14) For each flow and server, the sum of entering flow from the server to all adj switch is = than vms part of that flow
    # This is the splitted version, the combined one should 
    #   > remove the if condition 
    #   > change to <= type constraint
    #   > add the code line "- cqm_solution.get("vm" + str(proxytree.src_dst[f][1]) + "-s" + str(s-proxytree.SWITCHES))"  
    for f in range(proxytree.FLOWS):
        for s in range(proxytree.SWITCHES, proxytree.SWITCHES + proxytree.SERVERS):
            if vm_solution.get("vm" + str(proxytree.src_dst[f][1]) + "-s" + str(s-proxytree.SWITCHES)) == 0:
                cqm.add_constraint( 
                    dimod.quicksum( 
                        flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(s)] for sw in range(proxytree.SWITCHES) if proxytree.adjancy_list[sw][s] == 1) 
                    == 0, label="C14-N"+str(f*proxytree.SERVERS+s)) 

    # (15) For each flow and server, force allocation of all flows     
    for f in range(proxytree.FLOWS):
        for s in range(proxytree.SWITCHES, proxytree.SWITCHES + proxytree.SERVERS):
            cqm.add_constraint( 
                vm_solution.get("vm" + str(proxytree.src_dst[f][0]) + "-s" + str(s-proxytree.SWITCHES)) - vm_solution.get("vm" + str(proxytree.src_dst[f][1]) + "-s" + str(s-proxytree.SWITCHES))  
                - ( 
                    dimod.quicksum( 
                        flow_path['f' + str(f) + '-n' + str(s) + '-n' + str(sw)] for sw in range(proxytree.SWITCHES) if proxytree.adjancy_list[s][sw] == 1) 
                    - dimod.quicksum( 
                        flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(s)] for sw in range(proxytree.SWITCHES) if proxytree.adjancy_list[sw][s] == 1)) 
                == 0, label="C15-N"+str(f*proxytree.SERVERS+s))

    # (16) For each switch and flow, entering and exiting flow from the switch are equal
    for sw in range(proxytree.SWITCHES):
        for f in range(proxytree.FLOWS):
            cqm.add_constraint( 
                dimod.quicksum( 
                    flow_path['f' + str(f) + '-n' + str(n) + '-n' + str(sw)]  for n in range(proxytree.NODES) if proxytree.adjancy_list[n][sw] == 1) 
                - dimod.quicksum( 
                    flow_path['f' + str(f) + '-n' + str(sw) + '-n' + str(n)] for n in range(proxytree.NODES) if proxytree.adjancy_list[sw][n] == 1) 
                == 0, label="C16-N"+str(sw*proxytree.FLOWS+f))

    # (17) For each link, the data rate on it is less or equal than its capacity      
    for l in range(proxytree.LINKS):
        n1,n2 = fn.get_nodes(l, proxytree.link_dict)
        cqm.add_constraint( 
            dimod.quicksum( 
                proxytree.data_rate[f] * (
                    flow_path['f' + str(f) + '-n' + str(n1) + '-n' + str(n2)] 
                    + flow_path['f' + str(f) + '-n' + str(n2) + '-n' + str(n1)]) for f in range(proxytree.FLOWS)) 
            - proxytree.link_capacity[l] * on["on" + str(n1) + "-" + str(n2)] 
            <= 0, label="C17-N"+str(l))

    # (18)(19) For each link, the link is ON only if both nodes are ON       
    for l in range(proxytree.LINKS):
        n1,n2 = fn.get_nodes(l, proxytree.link_dict)

        cqm.add_constraint(
            on["on" + str(n1) + "-" + str(n2)] 
            - switch_status[n1] 
            <= 0, label="C18-N"+str(l))

        if n2 < proxytree.SWITCHES:     # If the second node is a switch
            cqm.add_constraint(
                on["on" + str(n1) + "-" + str(n2)] 
                - switch_status[n2] 
                <= 0, label="C19-N"+str(l))
        else:                           # If it's a server
            cqm.add_constraint(
                on["on" + str(n1) + "-" + str(n2)] 
                - vm_solution.get("s"+str(n2-proxytree.SWITCHES)) 
                == 0, label="C19-N"+str(l))


        # -------------------------------------------------
        # Based on structure formation, n1 will always be a switch
        # if structure changes, this will be the new conditions
        # structure. On top of that, this is the decomposed version,
        # the full one should modify the == to <= in the else.
        # -------------------------------------------------
        # if n1 < proxytree.SWITCHES:
        #     path_cqm.add_constraint(
        #         on["on" + str(n1) + "-" + str(n2)] 
        #         - switch_status[n1] 
        #         <= 0, label="C18-N"+str(l))
        # else:
        #     path_cqm.add_constraint(
        #         on["on" + str(n1) + "-" + str(n2)] 
        #         - cqm_best[0].get("s"+str(n1-SWITCHES)) 
        #         <= 0, label="C18-N"+str(l))
        # if n2 < proxytree.SWITCHES:
        #     path_cqm.add_constraint(
        #         on["on" + str(n1) + "-" + str(n2)] 
        #         - switch_status[n2] 
        #         <= 0, label="C19-N"+str(l))
        # else:
        #     path_cqm.add_constraint(
        #         on["on" + str(n1) + "-" + str(n2)] 
        #         - cqm_best[0].get("s"+str(n2-SWITCHES)) 
        #         == 0, label="C19-N"+str(l))






# ###################################################################################
#                         QUANTUM SAMPLER CODE
# ###################################################################################
# print("\n\n\n")
# print("####################### BQM Full Quantum Path Model ###########################")
# print("\n")
# Create Sampler
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