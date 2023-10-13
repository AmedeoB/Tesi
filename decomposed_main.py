DEBUG = True

"""
TODO
- trovare un modo per rendere opzionali da input i seguenti parametri:
    > salvataggio del dizionario
    > load del dizionario
    > moltiplicatori di lagrange
    > moltiplicatori di tempo
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
from fun_lib import printSection

# OTHERS
import numpy as np
import random
import json

"""--------------------------------------------------"""

proxytree = fn.Proxytree(
                depth = 3, 
                server_c = 10, 
                link_c = 5, 
                idle_pc = 10, 
                dyn_pc = 2, 
                datar_avg = 4
            )

proxymanager = fn.Proxymanager(
                proxytree, 
                # save_cqm_dict = False, 
                # load_cqm_dict = False, 
                # time_mul_vm = 1,
                # time_mul_path = 1,
                # lag_mul_vm = 10, 
                # lag_mul_path = 10
            )


# TREE STRUCTURE
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
models.vm_model(proxytree, vm_cqm)



#### CQM Solver ####
printSection("CQM VM Model")

# Solve
if DEBUG:   print("VM Save dictionary: ", proxymanager.SAVE_DICT)
if proxymanager.SAVE_DICT:
    vm_cqm_solution, vm_cqm_time = models.cqm_solver(vm_cqm, "vm_model", save = True)
else:
    vm_cqm_solution, vm_cqm_time = models.cqm_solver(vm_cqm, "vm_model")



#### BQM Solver ####
printSection("BQM VM Model")

# Convert
if DEBUG:   print("VM Custom Lagrange: ", proxymanager.VM_CUSTOM_LAGRANGE)
if proxymanager.VM_CUSTOM_LAGRANGE:
    vm_bqm, vm_inverter = dimod.cqm_to_bqm(vm_cqm, lagrange_multiplier = proxymanager.VM_LAGRANGE_MUL)
else:
    vm_bqm, vm_inverter = dimod.cqm_to_bqm(vm_cqm)

# Solve
if DEBUG:   print("VM Custom Time: ", proxymanager.VM_CUSTOM_TIME)
if proxymanager.VM_CUSTOM_TIME:  
    vm_bqm_solution = models.bqm_solver(vm_bqm, problem_label = "bqm_vm_model", 
                cqm_time = vm_cqm_time, time_mult = proxymanager.VM_TIME_MULT)
else:
    vm_bqm_solution = models.bqm_solver(vm_bqm, problem_label = "bqm_vm_model")

# Check
models.check_bqm_feasible(bqm_solution = vm_bqm_solution, cqm_model = vm_cqm, 
            inverter = vm_inverter)



############### PATH MODEL ########################################

# Create problem
path_cqm = dimod.ConstrainedQuadraticModel()
# Variables & Constraints
if DEBUG:   print("PATH Load Dictionary: ", proxymanager.LOAD_DICT)
if proxymanager.LOAD_DICT:
    models.path_model(proxytree, path_cqm, load = True)
else:
    models.path_model(proxytree, path_cqm, cqm_solution = vm_cqm_solution)



#### CQM Solver ####
printSection("CQM Path Model")

# Solve
if DEBUG:   print("PATH Save Dictionary: ", proxymanager.SAVE_DICT)
if proxymanager.SAVE_DICT:
    path_cqm_solution, path_cqm_time = models.cqm_solver(path_cqm, "path_model", save = True)
else:
    path_cqm_solution, path_cqm_time = models.cqm_solver(path_cqm, "path_model")



#### BQM Solver ####
printSection("BQM Path Model")

# Convert
# [LEGACY] # path_bqm, path_inverter = cqm_to_bqm(path_cqm, lagrange_multiplier = proxymanager.LAGRANGE_MUL2)
if DEBUG:   print("PATH Custom Lagrange: ", proxymanager.PATH_CUSTOM_LAGRANGE)
if proxymanager.PATH_CUSTOM_LAGRANGE:
    path_bqm, path_inverter = dimod.cqm_to_bqm(path_cqm, lagrange_multiplier = proxymanager.PATH_LAGRANGE_MUL)
else:
    path_bqm, path_inverter = dimod.cqm_to_bqm(path_cqm)

# Solve
if DEBUG:   print("PATH Custom Time: ", proxymanager.VM_CUSTOM_TIME)
if proxymanager.VM_CUSTOM_TIME:  
    path_bqm_solution = models.bqm_solver(path_bqm, problem_label = "bqm_path_model", 
            cqm_time = path_cqm_time, time_mult = proxymanager.PATH_TIME_MULT)
else:
    path_bqm_solution = models.bqm_solver(path_bqm, problem_label = "bqm_path_model")


# Check
models.check_bqm_feasible(bqm_solution = path_bqm_solution, cqm_model = path_cqm, 
            inverter = path_inverter)





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
