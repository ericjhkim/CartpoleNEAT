"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

from __future__ import print_function

import os
import pickle
import time

import neat
from neat import nn

import serial
import mycomms as fromArduino
import markercomms as comms

# arduino = serial.Serial('COM6', 115200, timeout=.1)

from math import pi

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# load the winner
with open('winner-feedforward', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
print(net)

comms.setupSerial(9600, 'COM6')
rxdata = fromArduino.receiveData()

# rxdata = fromArduino.receive(arduino)

# while rxdata == "Failed RX":
#     rxdata = fromArduino.receive(arduino)

theta = rxdata[0]
dtheta = rxdata[1]
x = rxdata[2]
dx = rxdata[3]

position_limit = 1
angle_limit_radians = 60*pi/180;

print()
print("Initial conditions:")
print("        x = {0:.4f}".format(x))
print("    x_dot = {0:.4f}".format(dx))
print("    theta = {0:.4f}".format(theta))
print("theta_dot = {0:.4f}".format(dtheta))
print()

# Run the given testCart for up to 10 seconds.
start_time = time.time()
while (time.time()-start_time) < 30.0:

    rxdata = fromArduino.receiveData()
    print(rxdata)
    # while rxdata == "Failed RX":
    #     rxdata = fromArduino.receive(arduino)

    theta = rxdata[0]
    dtheta = rxdata[1]
    x = rxdata[2]
    dx = rxdata[3]
    
    inputs = [0.5*(dx+position_limit)/position_limit,(dx+0.75)/1.5,0.5*(theta+angle_limit_radians)/angle_limit_radians,(dtheta+1.0)/2.0]
    print(inputs)
    action = net.activate(inputs)

    # Apply action to the cart-pole
    if action[0] > 0.5:
        comms.sendToArduino("f")
    else:
        comms.sendToArduino("b")

    # cmdstr = cmd + "\n"
    # arduino.write(cmdstr.encode())

    # Stop if the network fails to keep the cart within the position or angle limits.
    # The per-run fitness is the number of time steps the network can balance the pole
    # without exceeding these limits.
    if abs(x) >= position_limit or abs(theta) >= angle_limit_radians:
        # End all motor functions
        comms.sendToArduino("s")
        # cmd = "0"
        # cmdstr = cmd + "\n"
        # arduino.write(cmdstr.encode())

        if abs(x) >= position_limit:
            print("Out of position:\n"+str(abs(x))+' : '+str(position_limit))
        elif abs(theta) >= angle_limit_radians:
            print("Fell over:\n"+str(abs(theta))+' : '+str(angle_limit_radians))

        break
    #end
    
    balance_time = time.time()-start_time
#end

print('Pole balanced for {0:.1f} of 30.0 seconds'.format(balance_time))

print()
print("Final conditions:")
print("        x = {0:.4f}".format(x))
print("    x_dot = {0:.4f}".format(dx))
print("    theta = {0:.4f}".format(theta))
print("theta_dot = {0:.4f}".format(dtheta))
print()
