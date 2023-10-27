DEBUG = False

'''
TODO
    - remove full import from fun_lib and make single imports
'''


"""------------------- IMPORTS ---------------------"""
# D_WAVE
import dimod

# CUSTOM
import models
from fun_lib import *

# TESTING
from test_functions import *

"""--------------------------------------------------"""

proxytree = Proxytree(
                depth = 7, 
                server_c = 10, 
                link_c = 5, 
                idle_pc = 10, 
                dyn_pc = 2, 
                datar_avg = 4,
                # random_tree = True
            )

manager = CQMmanager(
                save_solution_vm = True, 
                save_info_vm = True, 
                # load_solution = True, 
                save_solution_path = True, 
                save_info_path = True
            )

# Print Tree Structure
proxytree.print_tree()
if DEBUG:   manager.print_manager()


# ###################################################################
# |                       VM MODEL                                  |
# ###################################################################
print_section("VM Model")

# Create problem
vm_cqm = dimod.ConstrainedQuadraticModel()
# Variables & Constraints
models.vm_model(proxytree, vm_cqm)

if DEBUG: print_model_structure("vm model", vm_cqm)

####################
#    CQM Solver    #
####################

# Solve
vm_cqm_solution, vm_cqm_info = models.detailed_cqm_solver(vm_cqm, "vm_model", 
                    proxytree.DEPTH, save_solution = manager.SAVE_VM_SOL,
                    save_info= manager.SAVE_VM_INFO)

if DEBUG:   print_cqm_extrainfo(vm_cqm_solution, vm_cqm_info, "vm_model")



# ###################################################################
# |                       PATH MODEL                                |
# ###################################################################
print_section("CQM Path Model")

# Create problem
path_cqm = dimod.ConstrainedQuadraticModel()
# Variables & Constraints
models.path_model(proxytree, path_cqm, vm_solution = vm_cqm_solution, 
            load = manager.LOAD_SOL)

if DEBUG: print_model_structure("path model", path_cqm)


####################
#    CQM Solver    #
####################

# Solve
path_cqm_solution, path_cqm_info = models.detailed_cqm_solver(path_cqm, "path_model", 
                    proxytree.DEPTH, save_solution = manager.SAVE_PATH_SOL,
                    save_info= manager.SAVE_PATH_INFO)

if DEBUG:   print_cqm_extrainfo(path_cqm_solution, path_cqm_info, "path_model")
