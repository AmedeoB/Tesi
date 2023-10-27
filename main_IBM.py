'''
TODO
    - remove full import from fun_lib and make single imports
'''
"""--------------------------------- IMPORTS ------------------------------------"""
# DOCPLEX
from docplex.cp.model import CpoModel

# CUSTOM
from fun_lib import Proxytree

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

# Create model
model = CpoModel()

# Create column index of each queen
server_status = [
        model.binary_var(name= "s{}".format(s))
        for s in range(proxytree.SERVERS)
        ]
vm_status = [
        model.binary_var(name= "vm{}-s{}".format(vm, s)) 
        for vm in range(proxytree.VMS) 
        for s in range(proxytree.SERVERS)
        ]