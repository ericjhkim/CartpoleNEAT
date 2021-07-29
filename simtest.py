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
with open('winner', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

# Initial and final states (0 degrees is straight up)
init = [0,np.pi,0,0]
final = [1,0,0,0]
error = [0.1,np.deg2rad(2),0.1,np.deg2rad(1)]

maxforce = 2 # Control force

net = neat.nn.FeedForwardNetwork.create(c, config)
sim = model.CartPole(init,final,error)

# print()
# print("Initial conditions:")
# print("        x = {0:.4f}".format(sim.x))
# print("    x_dot = {0:.4f}".format(sim.dx))
# print("    theta = {0:.4f}".format(sim.theta))
# print("theta_dot = {0:.4f}".format(sim.dtheta))
# print()

# Run the given simulation for up to 60 seconds.
testtime = 10
while sim.t < testtime:

    # End if obective met:
    if (
        abs(sim.x) >= sim.x_max or 
        abs(sim.theta) >= sim.theta_max or 
        abs(sim.dx) >= sim.dx_max or 
        abs(sim.dtheta) >= sim.dtheta_max
        ):
        sim.crash = True
        print('Out of bounds')
        break
    elif (
        sim.x <= sim.x_1+sim.e_x and sim.x >= sim.x_1-sim.e_x and
        sim.theta <= sim.theta_1+sim.e_theta and sim.theta >= sim.theta_1-sim.e_theta and
        sim.dx <= sim.dx_1+sim.e_dx and sim.dx >= sim.dx_1-sim.e_dx and
        sim.dtheta <= sim.dtheta_1+sim.e_dtheta and sim.dtheta >= sim.dtheta_1-sim.e_dtheta                
        ):
        print('Objective reached at '+str(sim.t)+'s')
        break

     # Get pole states
    inputs = sim.get_scaled_state()

    # Apply inputs to ANN
    action = net.activate(inputs)
    
    # Apply action to the simulated cart-pole
    # print(sim.theta)
    # control = sim.actuator(action,maxforce)
    force = model.discrete_actuator_force(action,maxforce)
    # control=0
    sim.step(force)

print('Pole balanced for ',round(sim.t,1),' of ',testtime,' seconds.')

myplts.states(sim)
matlab.py2mat(sim,'C:/EK_Projects/CartPole_NEAT/Matlab/cp_data.mat')

# print()
# print("Final conditions:")
# print("        x = {0:.4f}".format(sim.x))
# print("    x_dot = {0:.4f}".format(sim.dx))
# print("    theta = {0:.4f}".format(sim.theta))
# print("theta_dot = {0:.4f}".format(sim.dtheta))
# print()
# print("Making movie...")

# make_movie(net, model.discrete_actuator_force, 15.0, "movie.mp4")