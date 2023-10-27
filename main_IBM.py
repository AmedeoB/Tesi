'''
TODO
    - remove full import from fun_lib and make single imports
'''
"""--------------------------------- IMPORTS ------------------------------------"""
# DOCPLEX
from docplex.cp.model import CpoModel, minimize

# CUSTOM
from fun_lib import Proxytree
from fun_lib_IBM import *

"""------------------------------------------------------------------------------"""

proxytree = Proxytree(
                depth = 3, 
                server_c = 10, 
                link_c = 5, 
                idle_pc = 10, 
                dyn_pc = 2, 
                datar_avg = 4,
                # random_tree = True
            )





############
# VM MODEL #
#####################################################################################################

# Create
vm_model = CpoModel()
vm_cplex_model(vm_model, proxytree)        

# Solve
vm_solution = cplex_solver(vm_model)

#####################################################################################################


##############
# PATH MODEL #
#####################################################################################################

# Create
path_model = CpoModel()
path_cplex_model(path_model, proxytree, vm_solution)        

# Solve
path_solution = cplex_solver(path_model)
