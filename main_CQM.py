DEBUG = True

'''
TODO
    - nuovo proxymanager solo per cqm
'''


"""------------------- IMPORTS ---------------------"""
# D_WAVE
import dimod

# CUSTOM
import models
from fun_lib import print_section, Proxytree, Proxymanager

# TESTING
from test_functions import *

"""--------------------------------------------------"""

proxytree = Proxytree(
                depth = 3, 
                server_c = 10, 
                link_c = 5, 
                idle_pc = 10, 
                dyn_pc = 2, 
                datar_avg = 4,
                # random_tree = True
            )

proxymanager = Proxymanager(
                proxytree, 
                # save_cqm_dict = True, 
                load_cqm_dict = True, 
                # time_mul_vm = 1,
                # time_mul_path = 1,
                # lag_mul_vm = 10, 
                # lag_mul_path = 10
            )

# Print Tree Structure
proxytree.print_tree()


# ###################################################################
# |                       VM MODEL                                  |
# ###################################################################

# Create problem
vm_cqm = dimod.ConstrainedQuadraticModel()
# Variables & Constraints
models.vm_model(proxytree, vm_cqm)

if DEBUG:
    print_model_structure("vm model", vm_cqm)
exit()

####################
#    CQM Solver    #
####################
print_section("CQM VM Model")

# Solve
if DEBUG:   print("VM Save dictionary: ", proxymanager.SAVE_DICT)
if proxymanager.SAVE_DICT:
    vm_cqm_solution, vm_cqm_time = models.cqm_solver(vm_cqm, "vm_model", save = True)
else:
    vm_cqm_solution, vm_cqm_time = models.cqm_solver(vm_cqm, "vm_model")



# ###################################################################
# |                       PATH MODEL                                |
# ###################################################################

# Create problem
path_cqm = dimod.ConstrainedQuadraticModel()
# Variables & Constraints
if DEBUG:   print("PATH Load Dictionary: ", proxymanager.LOAD_DICT)
if proxymanager.LOAD_DICT:
    models.path_model(proxytree, path_cqm, load = True)
else:
    models.path_model(proxytree, path_cqm, vm_solution = vm_cqm_solution)




####################
#    CQM Solver    #
####################
print_section("CQM Path Model")

# Solve
if DEBUG:   print("PATH Save Dictionary: ", proxymanager.SAVE_DICT)
if proxymanager.SAVE_DICT:
    path_cqm_solution, path_cqm_time = models.cqm_solver(path_cqm, "path_model", save = True)
else:
    path_cqm_solution, path_cqm_time = models.cqm_solver(path_cqm, "path_model")
