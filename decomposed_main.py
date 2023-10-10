"""
TODO
"""
"""------------------- IMPORTS ---------------------"""
# D_WAVE
import dimod
import dwave.system
import dwave.inspector
import dwave.preprocessing

# CUSTOM
from Converter.converter import cqm_to_bqm
import fun_lib as fn
import models

# OTHERS
import numpy as np
import random
import json

"""--------------------------------------------------"""

proxytree = fn.Proxytree(depth = 3, server_c = 10, link_c = 5, idle_pc = 10
            , dyn_pc = 2, datar_avg = 4)
proxymanager = fn.Proxymanager(proxytree, save = False, load = False, 
            lag_mul_vm = 10, lag_mul_path = 10)

# Problem structure debugger
if proxymanager.DEBUG:
    print("SWITCH Indexes: ", *[k for k in range(proxytree.SWITCHES)])
    print("SERVER Indexes: ", *[s+proxytree.SWITCHES for s in range(proxytree.SERVERS)])
    print("SERVER Capacity: ", *[s for s in proxytree.server_capacity])
    print("LINK Capacity: ", *[s for s in proxytree.link_capacity])
    print("IDLE Power Consumption: ", *[s for s in proxytree.idle_powcons])
    print("DYNAMIC Power Consumption: ", *[s for s in proxytree.dyn_powcons])
    print("VM's CPU Utilization: ", *[s for s in proxytree.cpu_util])
    print("Flow Path Data Rate: ", *[s for s in proxytree.data_rate])
    print("\n\n")

    print("### Tree Structure ###")
    for i in range(len(proxytree.adjancy_list)):
        print("\nNodo ", i, " collegato ai nodi:", end="\t")
        for j in range(len(proxytree.adjancy_list)):
            if proxytree.adjancy_list[i][j] == 1:
                print(j, " (link ", proxytree.link_dict.get(str((i,j))) ,")", sep="", end="\t")
    print("\n\n")
    
    print("### VM Paths ###")
    for path in proxytree.src_dst:
        print("Path ", proxytree.src_dst.index(path), ": ", end="\t")
        print( *[s for s in path], sep="  -  ")
    print("\n")




############### VM MODEL ########################################
# Create problem
vm_cqm = dimod.ConstrainedQuadraticModel()
# Variables & Constraints
models.vm_model(proxytree, proxymanager, vm_cqm)


print("\n\n\n")
print("####################### CQM VM Model ###########################")
print("\n")
# Solve
vm_cqm_solution = models.cqm_solver(proxytree, proxymanager, vm_cqm, problem_label = "vm_model")

print("\n\n\n")
print("####################### BQM VM Model ###########################")
print("\n")
# Convert
# [LEGACY] # vm_bqm, vm_inverter = dimod.cqm_to_bqm(vm_cqm, lagrange_multiplier = proxymanager.LAGRANGE_MUL1)
vm_bqm, vm_inverter = cqm_to_bqm(vm_cqm, lagrange_multiplier = proxymanager.LAGRANGE_MUL1)
# Solve
vm_bqm_best = models.bqm_solver(proxytree, proxymanager, vm_bqm, cqm_time = vm_cqm_solution[1]
        , problem_label = "bqm_vm_model", time_mult = proxymanager.TIME_MULT1)
# Check
models.check_bqm_feasible(bqm_best = vm_bqm_best, cqm_model = vm_cqm, inverter = vm_inverter)



############### PATH MODEL ########################################
# Create problem
path_cqm = dimod.ConstrainedQuadraticModel()
# Variables & Constraints
models.path_model(proxytree, proxymanager, path_cqm, vm_cqm_solution)


print("\n\n\n")
print("####################### CQM Path Model ###########################")
print("\n")
# Solve
path_cqm_solution = models.cqm_solver(proxytree, proxymanager, path_cqm, "path_model")


print("\n\n\n")
print("####################### BQM Path Model ###########################")
print("\n")
# Convert
# [LEGACY] # path_bqm, path_inverter = dimod.cqm_to_bqm(path_cqm, lagrange_multiplier = proxymanager.LAGRANGE_MUL2)
path_bqm, path_inverter = cqm_to_bqm(path_cqm, lagrange_multiplier = proxymanager.LAGRANGE_MUL2)
# Solve
path_bqm_best = models.bqm_solver(proxytree, proxymanager, path_bqm, cqm_time = path_cqm_solution[1]
        , problem_label = "bqm_path_model", time_mult = proxymanager.TIME_MULT2)
# Check
models.check_bqm_feasible(bqm_best = path_bqm_best, cqm_model = path_cqm, inverter = path_inverter)





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
