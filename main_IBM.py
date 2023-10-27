'''
TODO
    - remove full import from fun_lib and make single imports
'''
"""--------------------------------- IMPORTS ------------------------------------"""
# DOCPLEX
from docplex.cp.model import CpoModel, minimize

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
    model.binary_var(name= "s{}".format(s)) # pylint: disable=no-member
    for s in range(proxytree.SERVERS)
]
vm_status = [
    [
        model.binary_var(name= "vm{}-s{}".format(vm, s)) # pylint: disable=no-member
        for vm in range(proxytree.VMS) 
    ] for s in range(proxytree.SERVERS)
]

for s in range(proxytree.SERVERS):
    model.add_constraint(
        sum(
            (proxytree.cpu_util[vm] * vm_status[s][vm])
            for vm in range(proxytree.VMS)
        )
        - (proxytree.server_capacity[s] * server_status[s])
        <= 0
    )

for vm in range(proxytree.VMS):
    model.add_constraint(
        sum(
            vm_status[s][vm] 
            for s in range(proxytree.SERVERS)
        )
        == 1
    )

model.add(
    minimize(
        sum(
            (server_status[s] * proxytree.idle_powcons[s+proxytree.SWITCHES])
            for s in range(proxytree.SERVERS)
        ) 
        + sum(
            (proxytree.dyn_powcons[s+proxytree.SWITCHES]
            * sum(
                (vm_status[vm][s] * proxytree.cpu_util[vm])
                for vm in range(proxytree.VMS))
            ) 
            for s in range(proxytree.SERVERS)
        )
    )
)
        

print("Solving...")
solution = model.solve(TimeLimit= 10)
solution.print_solution()

print(
    f"\n# SOLUTION #\n"
    f"\nSolve Status: {solution.get_solve_status()}"
    f"\nEnergy: {solution.get_objective_value()}"
    f"\nSolve Time: {solution.get_solve_time()}"
)
solution.a
print("\n# VARIABLES \n")
for var in solution.get_all_var_solutions():
    if var.get_value() != 0:
        print(f"{var.get_name()}: {var.get_value()}")
