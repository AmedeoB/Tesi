import dimod
import hybrid
import dwave.system as system
import json
def testwriter():
    testset = {
        "brand": "Ford",
        "model": "Mustang",
        "year": 1964
    }
    depth=4
    name="test"
    with open((f"CQM LOGS/depth_{depth}/{name}.txt"), "w") as fp:
            json.dump(testset, fp)
    with open(f"CQM LOGS/depth_{depth}/{name}.txt") as fp:
                dictionary = json.loads(fp.read())
                print(dictionary)

def print_decomposition(decomposer_name: str, subproblem):
    print("\n\n")
    print("####################### " + decomposer_name.upper() + " ###########################")
    # print(subproblem)
    try:
        # print("Subset: ", subproblem)
        print("Num Variables: ", subproblem.num_variables)
        # print("Variables: ", list(subproblem.variables))
        print("Num Interactions: ", subproblem.num_interactions)
        # print("Interactions: ", list(subproblem.quadratic))
        print("\n")
    except:
        print("Error")

def component_custom_key(item):
    if  "slack" in str(item):
        return 0
    elif "sw" in str(item):
        return 1
    elif "-n" in str(item):
        return 2
    elif "on" in str(item):
        return 3
    else:
        return 4


def decomposer_test(bqm: dimod.BinaryQuadraticModel):
    
    print("BQM INFOS")
    print("Variables: ", bqm.num_variables)
    print("Interactions: ", bqm.num_interactions)

    # quads = list(bqm.quadratic)
    # cons = []
    # for i in quads:
    #     if "slack" in str(i): print(str(i))
    #     cons.append(set(i))
    # # print(cons)

    init_state = hybrid.State.from_sample(
                        hybrid.min_sample(bqm),
                        bqm,
                        # embedding
                    )
    subproblem_size = 50

    # ERROR: produce un sottoinsieme uguale all'insieme iniziale
    # decomposer = hybrid.ComponentDecomposer(key=component_custom_key, reverse=False)
    # decomposition = decomposer.next(init_state).result().subproblem
    # print_decomposition("Component Decomposer", decomposition)
    
    decomposer = hybrid.EnergyImpactDecomposer(subproblem_size)
    decomposition = decomposer.next(init_state).result().subproblem
    print_decomposition("Energy Decomposer", decomposition)

    # ERROR: has no attribute 'constraint_graph'
    # decomposer = hybrid.RandomConstraintDecomposer(subproblem_size, cons)
    # decomposition = decomposer.next(init_state).result().subproblem
    # print_decomposition("Constraint Decomposer", decomposition)

    decomposer = hybrid.RandomSubproblemDecomposer(subproblem_size)
    decomposition = decomposer.next(init_state).result().subproblem
    print_decomposition("Random Decomposer", decomposition)

    # ERROR: non produce sottinsiemi
    # decomposer = hybrid.SublatticeDecomposer()
    # decomposition = decomposer.next(init_state).result().subproblem
    # print_decomposition("Sublattice Decomposer", decomposition)

def test_decomposed_solver(bqm: dimod.BinaryQuadraticModel):

    subproblem_size = 30
    max_time = 30

    init_state = hybrid.State.from_sample(
                        hybrid.min_sample(bqm),
                        bqm
                )
    print("Init Energy: ", init_state.samples.first[1])


    decomposer = hybrid.EnergyImpactDecomposer(
                            size= subproblem_size,
                            rolling_history= 1.00,
                            # traversal= 'energy',
                ) 
    subsampler = hybrid.QPUSubproblemAutoEmbeddingSampler()
    composer = hybrid.SplatComposer()

    classic_branch = hybrid.InterruptableTabuSampler()

    merger = hybrid.GreedyPathMerge()

    qpu_branch = (decomposer | subsampler | composer) | hybrid.TrackMin()
    parallel_branches = hybrid.Race(
                            classic_branch,
                            qpu_branch
                        ) | merger
    
    workflow = hybrid.Loop(
                        qpu_branch, 
                        # convergence= 3, 
                        # max_iter= 5, 
                        max_time= max_time,
                        )

    final_state = workflow.run(init_state).result()

    best_solution = final_state.samples.first[0]
    energy = final_state.samples.first[1]

    # Energy
    print("Decomposer BQM ENERGY: ", energy)

    # Extract variables
    print("\n## Decomposer BQM Variables ##")
    last_char = ""
    for var, value in best_solution.items():
        if last_char != var[0]:         # Var separator
            print(end="\n")
        if value != 0.0:                # Nonzero var printer
            print(var, value, sep = ": ",end= " | ")
        last_char = var[0]          # Update last char to separate vars
    print(end= "\n")
    
    # Extract infos
    print("\n\n## Decomposer BQM Extra Info ##")
    print(final_state.info)

testwriter()