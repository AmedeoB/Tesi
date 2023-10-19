# | KERBEROS EXAMPLE |
'''
In questo esempio è usato Kerberos, un sistema parallelo già
strutturato
'''

import dimod
from hybrid.reference.kerberos import KerberosSampler
with open('/workspace/Tesi/Decomposer Tests/kerberos_example.qubo') as problem:  
    bqm = dimod.BinaryQuadraticModel.from_coo(problem)
len(bqm)          
solution = KerberosSampler().sample(bqm, max_iter=10, convergence=3)   
print(solution)
print(solution.first)
print(solution.first.energy)
print(solution.samples)
print(solution.first.sampleset)
# solution.first.energy     


############################################################################################
# | BUILDING BLOCKS EXAMPLE |
'''
In questo esempio viene creato un bqm e un tabu sampler.
Dal primo è poi ottenuto uno stato con sample di base con valori a 0.
Viene poi runnato e il risultato, il nuovo stato, è estratto.
Da questo è poi strapolato il sampleset
'''

import dimod
import hybrid
# Define a problem
bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': 0.5, 'bc': 0.5, 'ca': 0.5})
# Set up the sampler with an initial state
sampler = hybrid.TabuProblemSampler(tenure=2, timeout=5)
state = hybrid.State.from_sample({'a': 0, 'b': 0, 'c': 0}, bqm)
# Sample the problem
new_state = sampler.run(state).result()
print(new_state.samples)                     

############################################################################################
# | FLOW STRUCTURING EXAMPLE |
'''
In questo esempio viene creato un branch unico costituito da un energy decomposer
che divide il problema in sottoproblemi di dimensione 6, un tabu sampler che sampla
il problema due volte e uno splat composer.
Il problema inizia da uno stato minimo (-1 a tutte, in quanto variabili di spin)
'''

import dimod           # Create a binary quadratic model
from hybrid import EnergyImpactDecomposer, TabuSubproblemSampler, SplatComposer, State, min_sample
bqm = dimod.BinaryQuadraticModel({t: 0 for t in range(10)},
                                 {(t, (t+1) % 10): 1 for t in range(10)},
                                 0, 'SPIN')
branch = (EnergyImpactDecomposer(size=6, min_gain=-10) |
          TabuSubproblemSampler(num_reads=2) |
          SplatComposer())
new_state = branch.next(State.from_sample(min_sample(bqm), bqm))
print(new_state) 
print(new_state.subsamples) 
print(new_state.subsamples.info)  
print(new_state.subsamples.info.get('qpu_sampling_time'))  
print(new_state.subsamples.info.get('run_time'))      

############################################################################################
# | RACING BRANCHES EXAMPLE |
'''
In questo esempio ci sono 2 branches paralleli, dai loro risultati è selezionato
il migliore usando ArgMin e poi un loop crea un ciclo di risoluzione finché
l'output non rimane lo stesso per 3 iterazioni di fila
'''

import dimod
import hybrid

# Construct a problem
bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, dimod.SPIN)

# Define the workflow
iteration = hybrid.RacingBranches(
    hybrid.InterruptableTabuSampler(),
    hybrid.EnergyImpactDecomposer(size=2)
    | hybrid.QPUSubproblemAutoEmbeddingSampler()
    | hybrid.SplatComposer()
) | hybrid.ArgMin()
workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

# Solve the problem
init_state = hybrid.State.from_problem(bqm)
final_state = workflow.run(init_state).result()

time = final_state.subsamples#.info.get('qpu_sampling_time')

# Print results
print(final_state)
print("Solution: sample={.samples.first}".format(final_state))
print("Time: ", time)

############################################################################################
# | FLOW REFINING EXAMPLE |
'''
In questo esempio è configurata una finestra mobile di decomposizione che
decompone il problema in problemi di dimensione 50 fino a un totale del 
15% delle variabili. (In pratica a ogni iterazione usa 50 variabili diverse e
mai usate ma quando arriva al 15% si resetta)
'''


# Redefine the workflow: a rolling decomposition window
subproblem = hybrid.EnergyImpactDecomposer(size=50, rolling_history=0.15)
subsampler = hybrid.QPUSubproblemAutoEmbeddingSampler() | hybrid.SplatComposer()

iteration = hybrid.RacingBranches(
    hybrid.InterruptableTabuSampler(),
    subproblem | subsampler
) | hybrid.ArgMin()

workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

'''
In questo esempio invece che produrre un singolo sottoproblema per iterazione, ne 
vengono prodotti multipli e poi dati in pasto al smapler in parallelo.
Essendo insiemi di variabili disgiunte (subsample da 50, unici fino al 15%) si ricompongono
con semplicemente in un unico subsamples che poi viene ricomposto con l'input come di norma.

Da notare che qui non è runnato, la divisione è
    > Unwind, che suddivide tramite EnergyImpactDecomposer
    > Map, che mappa il sampler su tutti i subsamples
    > Reduce, che rimette insieme i subsamples
    > SplatComposer, che ricompone
'''
# Redefine the workflow: parallel subproblem solving for a single sample
subproblem = hybrid.Unwind(
    hybrid.EnergyImpactDecomposer(size=50, rolling_history=0.15)
)

# Helper function to merge subsamples in place
def merge_substates(_, substates):
    a, b = substates
    return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))

# Map QPU sampling over all subproblems, then reduce subsamples by merging in place
subsampler = hybrid.Map(
    hybrid.QPUSubproblemAutoEmbeddingSampler()
) | hybrid.Reduce(
    hybrid.Lambda(merge_substates)
) | hybrid.SplatComposer()

'''
In questo esempio la selezione delle variabili si basa sulla metodologia
Breadth First
'''
# Redefine the workflow: subproblem selection
subproblem = hybrid.Unwind(
    hybrid.EnergyImpactDecomposer(size=50, rolling_history=0.15, traversal='bfs'))

############################################################################################
# | TAILORING STATE SELECTION EXAMPLE |
'''
In questo esempio, ci sono 3 stati, uno dei quali ha flaggato un problema di Postprocessor.
Tramite un filtro appositamente costruito per l'argmin (la funzione preempt), possiamo quindi 
evitare di selezionare tale stato e di prendere invece il primo con valore ottimo non flaggato.

[{...,'samples': SampleSet(rec.array([([0, 1, 0], 0., 1)], ..., ['a', 'b', 'c'], {'Postprocessor': 'Excessive chain breaks'}, 'SPIN')},
{...,'samples': SampleSet(rec.array([([1, 1, 1], 1.5, 1)], ..., ['a', 'b', 'c'], {}, 'SPIN')},
{...,'samples': SampleSet(rec.array([([0, 0, 0], 0., 1)], ..., ['a', 'b', 'c'], {}, 'SPIN')}]
'''

def preempt(si):
    if 'Postprocessor' in si.samples.info:
        return(math.inf)
    else:
        return(si.samples.first.energy)

ArgMin(key=preempt).next(states)     


############################################################################################
# | -- EXAMPLE |



############################################################################################
# | -- EXAMPLE |



############################################################################################
# | -- EXAMPLE |
