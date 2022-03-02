# Compare cart pole performance RL vs NEAT

import numpy as np
import pickle
import plots as myplts

# Load RL
with open(f'K:\\EK School\\Postgraduate\\CISC856 Reinforcement Learning\\Project\\Deliverables\\Code\\results\\bestsim_ac_1_32', 'rb') as f:
    simRL = pickle.load(f)
# Load NEAT
with open(f'./Results/n0_sim', 'rb') as f:
    simNEAT = pickle.load(f)

myplts.states_compare(simRL,simNEAT)