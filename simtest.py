"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

from __future__ import print_function
import os
import pickle
import model
from movie import make_movie
import neat
import plots as myplts
import numpy as np
import matlab

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# load the winner
with open('C:\\EK_Projects\\CP_NEAT\\winner', 'rb') as f:
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

# print()
# print("Initial conditions:")
# print("        x = {0:.4f}".format(sim.x))
# print("    x_dot = {0:.4f}".format(sim.dx))
# print("    theta = {0:.4f}".format(sim.theta))
# print("theta_dot = {0:.4f}".format(sim.dtheta))
# print()

# Run the given simulation for up to 60 seconds.
testtime = 15
while sim.t < testtime:

    # Break if critical failure
    if abs(sim.x) >= sim.x_max or abs(sim.theta) >= sim.theta_max:
        sim.crash = True
        print('FAILED: Out of bounds.')
        break

    # Get cartpole states
    inputs = sim.get_states()
    # Apply inputs to ANN
    action = net.activate(inputs)
    # Obtain control values
    control = sim.discrete_actuator_force(action)
    # Apply control to simulation
    sim.step(control)

print('Pole balanced for ',round(sim.t,1),' of ',testtime,' seconds.')

myplts.states(sim)
matlab.py2mat(sim,'C:/EK_Projects/CP_NEAT/Matlab/cp_data.mat')

# print()
# print("Final conditions:")
# print("        x = {0:.4f}".format(sim.x))
# print("    x_dot = {0:.4f}".format(sim.dx))
# print("    theta = {0:.4f}".format(sim.theta))
# print("theta_dot = {0:.4f}".format(sim.dtheta))
# print()
# print("Making movie...")

# make_movie(net, model.discrete_actuator_force, 15.0, "movie.mp4")