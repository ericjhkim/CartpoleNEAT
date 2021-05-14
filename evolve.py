"""
Single-pole balancing experiment using a feed-forward neural network.
"""

from __future__ import print_function
import os
import pickle
import model
import neat
import visualize
import winsound

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

generations = 100
runs_per_net = 5
input_del = [[0,0,0,0]] # Delayed input by 1 loop cycle (10ms delay)

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        
        sim = model.CartPole()
        
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < sim.simtime:

            # Break if critical failure
            if abs(sim.x) >= sim.x_max or abs(sim.theta) >= sim.theta_max:
                sim.crash = True
                break

            # Get pole states
            inputs = sim.get_scaled_state()

            # Apply inputs to ANN
            action = net.activate(inputs)
            
            # Apply action to the simulated cart-pole
            force = model.discrete_actuator_force(action)
            sim.step(force)

        # Evaluate genome fitness
        fitness = sim.fitness()
        fitnesses.append(fitness)
    
    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate,generations)

    # Save the winner.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)
    #end
    print(winner)

    duration = 1000  # milliseconds
    freq = 640  # Hz
    winsound.Beep(freq, duration)

    # visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    # visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}

    # visualize.draw_net(config, winner, view=True, node_names=node_names,filename="winner-feedforward.gv")
    # visualize.draw_net(config, winner, view=True, node_names=node_names,filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,filename="winner-enabled-pruned.gv", show_disabled=False, prune_unused=True)

if __name__ == '__main__':
    run()