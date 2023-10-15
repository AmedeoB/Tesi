############################################################################################

# import dimod
# import hybrid

# # Construct a problem
# bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, dimod.SPIN)

# # Define the workflow
# iteration = hybrid.RacingBranches(
#     hybrid.InterruptableTabuSampler(),
#     hybrid.EnergyImpactDecomposer(size=2)
#     | hybrid.QPUSubproblemAutoEmbeddingSampler()
#     | hybrid.SplatComposer()
# ) | hybrid.ArgMin()
# workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)

# # Solve the problem
# init_state = hybrid.State.from_problem(bqm)
# final_state = workflow.run(init_state).result()

# time = final_state.subsamples#.info.get('qpu_sampling_time')

# # Print results
# print(final_state)
# print("Solution: sample={.samples.first}".format(final_state))
# print("Time: ", time)

############################################################################################

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