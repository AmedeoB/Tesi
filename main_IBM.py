'''
TODO
    - remove full import from fun_lib and make single imports
    - academia version for depth 5+
    - move save function from test lib to funlib
    - find way to set custom automatic timer and others
        stoppers
    - install python 3.10
'''
"""--------------------------------- IMPORTS ------------------------------------"""
# DOCPLEX
from docplex.cp.model import CpoModel, minimize

# CUSTOM
from fun_lib import Proxytree
from fun_lib_IBM import *

"""------------------------------------------------------------------------------"""

proxytree = Proxytree(
                depth = 12, 
                server_c = 10, 
                link_c = 5, 
                idle_pc = 10, 
                dyn_pc = 2, 
                datar_avg = 4,
                # random_tree = True
            )
ITERATIONS = 10




############
# VM MODEL #
#####################################################################################################

# Create
vm_model = CpoModel()
vm_cplex_model(vm_model, proxytree)        

# Solve
for _ in range(ITERATIONS):
    vm_solution = cplex_solver(vm_model, proxytree.DEPTH, "vm_model", save_solution=True)

#####################################################################################################


##############
# PATH MODEL #
#####################################################################################################

# Create
path_model = CpoModel()
path_cplex_model(path_model, proxytree, vm_solution)        

# Solve
for _ in range(ITERATIONS):
    path_solution = cplex_solver(path_model, proxytree.DEPTH, "path_model", save_solution=True)
