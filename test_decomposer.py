import dimod
import hybrid
import dwave.system as system


def print_decomposition(decomposer_name: str, decomposition):
    
    print("\n\n")
    print("####################### " + decomposer_name.upper() + " ###########################")
    # print(decomposition)
    print("Subset dimension: ", len(decomposition))
    print(decomposition.num_variables)
    print(decomposition.num_interactions)
    print("\n")



def decomposer_test(bqm: dimod.BinaryQuadraticModel):
    
    print("BQM INFOS")
    print("Variables: ", bqm.num_variables)
    print("Interactions: ", bqm.num_interactions)

    init_state = hybrid.State.from_sample(
                        hybrid.min_sample(bqm),
                        bqm,
                        # embedding
                    )
    subproblem_size = 10
    
    decomposer = hybrid.ComponentDecomposer()
    decomposition = decomposer.next(init_state).result().subproblem
    print_decomposition("Component Decomposer", decomposition)

    exit()

    decomposer = hybrid.EnergyImpactDecomposer(subproblem_size)
    decomposition = decomposer.next(init_state).result()
    print_decomposition("Energy Decomposer", decomposition)

    decomposer = hybrid.RandomConstraintDecomposer(subproblem_size, list(bqm.quadratic))
    decomposition = decomposer.next(init_state).result()
    print_decomposition("Constraint Decomposer", decomposition)

    decomposer = hybrid.RandomSubproblemDecomposer(subproblem_size)
    decomposition = decomposer.next(init_state).result()
    print_decomposition("Random Decomposer", decomposition)

    decomposer = hybrid.SublatticeDecomposer()
    decomposition = decomposer.next(init_state).result()
    print_decomposition("Sublattice Decomposer", decomposition)

