"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

from __future__ import print_function
import os
import pickle
import model
from movie import make_movie
import neat

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# load the winner
with open('winner', 'rb') as f:
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
testtime = 600
while sim.t < testtime:
     # Get pole states
    inputs = sim.get_scaled_state()

    # Apply inputs to ANN
    action = net.activate(inputs)
    
    # Apply action to the simulated cart-pole
    force = model.discrete_actuator_force(action)
    sim.step(force)

print('Pole balanced for ',round(sim.t,1),' of ',testtime,' seconds.')

# print()
# print("Final conditions:")
# print("        x = {0:.4f}".format(sim.x))
# print("    x_dot = {0:.4f}".format(sim.dx))
# print("    theta = {0:.4f}".format(sim.theta))
# print("theta_dot = {0:.4f}".format(sim.dtheta))
# print()
# print("Making movie...")

# make_movie(net, model.discrete_actuator_force, 15.0, "movie.mp4")