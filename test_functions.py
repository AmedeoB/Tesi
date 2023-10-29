# import dimod
# import hybrid
# import dwave.system as system
import json
from os.path import exists

def dict_filter(pair):
    key, _ = pair
    return "time" in key 

def info_writer(dictionary: dict, path: str):
   
    writeheads = False
    if not exists(path): writeheads = True
    
    with open(path,"a") as file:
        # Keys
        if writeheads:
            for k in dictionary.keys():
                file.write(f"{k}\t")
            file.write(f"\n")
        
        # Values
        for v in dictionary.values():
            file.write(f"{v}\t")
        file.write(f"\n")



# def print_decomposition(decomposer_name: str, subproblem):
#     print("\n\n")
#     print("####################### " + decomposer_name.upper() + " ###########################")
#     # print(subproblem)
#     try:
#         # print("Subset: ", subproblem)
#         print("Num Variables: ", subproblem.num_variables)
#         # print("Variables: ", list(subproblem.variables))
#         print("Num Interactions: ", subproblem.num_interactions)
#         # print("Interactions: ", list(subproblem.quadratic))
#         print("\n")
#     except:
#         print("Error")

# def component_custom_key(item):
#     if  "slack" in str(item):
#         return 0
#     elif "sw" in str(item):
#         return 1
#     elif "-n" in str(item):
#         return 2
#     elif "on" in str(item):
#         return 3
#     else:
#         return 4


# def decomposer_test(bqm: dimod.BinaryQuadraticModel):
    
#     print("BQM INFOS")
#     print("Variables: ", bqm.num_variables)
#     print("Interactions: ", bqm.num_interactions)

#     # quads = list(bqm.quadratic)
#     # cons = []
#     # for i in quads:
#     #     if "slack" in str(i): print(str(i))
#     #     cons.append(set(i))
#     # # print(cons)

#     init_state = hybrid.State.from_sample(
#                         hybrid.min_sample(bqm),
#                         bqm,
#                         # embedding
#                     )
#     subproblem_size = 50

#     # ERROR: produce un sottoinsieme uguale all'insieme iniziale
#     # decomposer = hybrid.ComponentDecomposer(key=component_custom_key, reverse=False)
#     # decomposition = decomposer.next(init_state).result().subproblem
#     # print_decomposition("Component Decomposer", decomposition)
    
#     decomposer = hybrid.EnergyImpactDecomposer(subproblem_size)
#     decomposition = decomposer.next(init_state).result().subproblem
#     print_decomposition("Energy Decomposer", decomposition)

#     # ERROR: has no attribute 'constraint_graph'
#     # decomposer = hybrid.RandomConstraintDecomposer(subproblem_size, cons)
#     # decomposition = decomposer.next(init_state).result().subproblem
#     # print_decomposition("Constraint Decomposer", decomposition)

#     decomposer = hybrid.RandomSubproblemDecomposer(subproblem_size)
#     decomposition = decomposer.next(init_state).result().subproblem
#     print_decomposition("Random Decomposer", decomposition)

#     # ERROR: non produce sottinsiemi
#     # decomposer = hybrid.SublatticeDecomposer()
#     # decomposition = decomposer.next(init_state).result().subproblem
#     # print_decomposition("Sublattice Decomposer", decomposition)
