"""
Evolving neural network for inverted pendulum problem
"""

from __future__ import print_function
import os
import pickle
import model_ip as model
import neat
import visualize
import winsound
import numpy as np
import plots as myplts
import matlab
import animate
from time import process_time, perf_counter

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

network = "n0"
folder = "Results_IP"

generations = 200
runs_per_net = 1

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        
        sim = model.CartPole()
        
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < sim.simtime and not sim.crash and not sim.done:

            # Break if critical failure
            sim.check_done()
            # Get cartpole states
            inputs = sim.get_cnstates()
            # Apply inputs to ANN
            action = net.activate(inputs)
            # Obtain control values
            control = sim.continuous_actuator_force(action)
            # Apply control to simulation (artificial lag)
            sim.step(control)
            
        # Evaluate genome fitness
        fitness = sim.fitness_ip()
        fitnesses.append(fitness)
    
    # The genome's fitness is its worst performance across all runs.
    # return min(fitnesses)
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    t_start_real = perf_counter()
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    t_start = process_time() # Mark start of evolution
    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate,generations)
    t_stop = process_time() # Mark end of evolution

    # Save the winner.
    with open(f'./{folder}/{network}', 'wb') as f:
        pickle.dump(winner, f)
    #end
    print(winner)

    # visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    # visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'theta', -3: 'dx', -4: 'dtheta', 0: 'F'}

    visualize.draw_net(config, winner, view=True, node_names=node_names,filename=f"./{folder}/{network}-enabled-pruned.gv", show_disabled=False, prune_unused=True)

    duration = 1000  # milliseconds
    freq = 640  # Hz
    winsound.Beep(freq, duration)

    ########################
    # Stopwatches
    ########################
    cputime = round(t_stop-t_start,2)
    print('CPU time: '+str(cputime)+'s = '+str(round(cputime/60,2))+'m.') # Print CPU time
    t_stop_real = perf_counter()
    realtime = round(t_stop_real-t_start_real,2)
    print('Elapsed time: '+str(realtime)+'s = '+str(round(realtime/60,2))+'m.') # Print real world time

    ########################
    # Simtest
    ########################
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    sim = model.CartPole()
        
    # Run the given simulation for up to num_steps time steps.
    while sim.t < sim.simtime and not sim.crash and not sim.done:

        # Break if critical failure
        sim.check_done()
        # Get cartpole states
        inputs = sim.get_cnstates()
        # Apply inputs to ANN
        action = net.activate(inputs)
        # Obtain control values
        control = sim.continuous_actuator_force(action)
        # Apply control to simulation
        sim.step(control)

    sim.print_report()
    myplts.ip_states(sim)
    # animate.animate(sim)
    # matlab.py2mat(sim,'C:/EK_Projects/CP_NEAT/Matlab/cp_data.mat')

if __name__ == '__main__':
    run()