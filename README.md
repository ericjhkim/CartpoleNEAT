This repository contains code for:

1. Evolving neurocontrollers for physical inverted pendulum (cartpole) system --> code in main directory
2. Testing trained neurocontrollers on Arduino cartpole platform --> code in \ArduinoNEAT

To evolve neural networks:
- Use "config" to set hyperparameters (see documentation: https://neat-python.readthedocs.io/en/latest/)
- Run "evolve.py" to generate a neural network (networks will be stored in a Python pickle called "winner")

To visualize the network:
- A "winner-enabled-pruned.gv.svg" file will generate along with the "winner" file at every run of "evolve.py." Open the svg file to see the network.

To edit the cartpole dynamics and model:
- Change "model.py"

To load and evaluate a "winner" network:
- Run "simtest.py" to run a simulation of the cartpole task with the loaded neural network as the controller

Additional notes (can be ignored):
- "movie.py" can be used to generate a low-quality animation of a particular network's performance
- "plts.py" simply contains various matplotlib functions for plotting cartpole behaviour
- "matlab.py" is to save data to a .mat file for MATLAB-related analysis
