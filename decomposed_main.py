DEBUG = True

"""
TODO
"""

"""------------------- IMPORTS ---------------------"""
# D_WAVE
import dimod

# CUSTOM
import models
from fun_lib import print_section, Proxytree, Proxymanager

"""--------------------------------------------------"""

proxytree = Proxytree(
                depth = 3, 
                server_c = 10, 
                link_c = 5, 
                idle_pc = 10, 
                dyn_pc = 2, 
                datar_avg = 4
            )

proxymanager = Proxymanager(
                proxytree, 
                # save_cqm_dict = False, 
                # load_cqm_dict = False, 
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



####################
#    BQM Solver    #
####################
print_section("BQM VM Model")

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
    models.path_model(proxytree, path_cqm, cqm_solution = vm_cqm_solution)



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



####################
#    BQM Solver    #
####################
print_section("BQM Path Model")

# Convert
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





