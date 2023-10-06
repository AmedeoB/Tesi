"""
TODO
    
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
import models
import json

"""--------------------------------------------------"""
proxytree = fn.Proxytree()
proxymanager = fn.Proxymanager(proxytree.IDLE_PC)


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
vm_cqm = dimod.ConstrainedQuadraticModel()
models.vm_model(proxytree, proxymanager, vm_cqm)


print("\n\n\n")
print("####################### CQM VM Model ###########################")
print("\n")

cqm_best = models.cqm_vm_solver(proxytree, proxymanager, vm_cqm)


# print("\n\n\n")
# print("####################### BQM VM Model ###########################")
# print("\n")

# models.bqm_vm_solver(proxytree, proxymanager, vm_cqm, cqm_best[1])




############### PATH MODEL ########################################
path_cqm = dimod.ConstrainedQuadraticModel()
models.path_model(proxytree, proxymanager, path_cqm, cqm_best)


print("\n\n\n")
print("####################### CQM Path Model ###########################")
print("\n")

cqm_time = models.cqm_path_solver(proxytree, proxymanager, path_cqm)


print("\n\n\n")
print("####################### BQM Path Model ###########################")
print("\n")

models.bqm_path_solver(proxytree, proxymanager, path_cqm, cqm_time)





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
