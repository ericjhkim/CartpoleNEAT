"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

from __future__ import print_function
import os
import pickle
import model_ip as model
from movie import make_movie
import neat
import plots as myplts
import numpy as np
import matlab
import animate
import pickle

network = "n0"
savesim = 0
folder = "Results_IP"

# load the winner
with open(f'./{folder}/{network}', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
sim = model.CartPole()

# Run the given simulation for up to 60 seconds.
while sim.t < sim.simtime and not sim.crash and not sim.done:

    # Break if critical failure
    sim.check_done()
    if sim.crash or sim.done:
        break

    # Get cartpole states
    inputs = sim.get_cnstates()
    # Apply inputs to ANN
    action = net.activate(inputs)
    # Obtain control values
    control = sim.continuous_actuator_force(action)
    # Apply control to simulation
    sim.step(control)

print(f'Pendulum inverted in {round(sim.t,1)} seconds.')

if savesim:
    with open(f'./{folder}/{network}_sim', 'wb') as f:
        pickle.dump(sim, f)
        print("Saved")

sim.print_report()
myplts.ip_states(sim)
animate.animate(sim)

# matlab.py2mat(sim,'C:/EK_Projects/CP_NEAT/Matlab/cp_data.mat')
