# Compare inverted pendulum optimal trajectory

import numpy as np
import pickle
import plots as myplts
from scipy.io import loadmat
import model_ip as model
import animate

# # Load NEAT
# with open(f'./Results_IP/n0_sim', 'rb') as f:
#     simNEAT = pickle.load(f)
    
# Load trajopt
trajopt = loadmat(f'C:\\EK_Projects\\TrajOpt\\Matlab\\cp_data.mat')
states = trajopt["states"][0]
simTO = model.CartPole()

simTO.t_list = trajopt["t_list"][0]
simTO.x_list = states[:,0]
simTO.dx_list = states[:,1]
simTO.theta_list = states[:,2]
simTO.dtheta_list = states[:,3]
simTO.action_list = np.squeeze(trajopt["control"][0])

print(np.diff(simTO.t_list)[0],simTO.t_list[-1])

simTO.print_report()
myplts.ip_states(simTO)
animate.animate(simTO)

# myplts.states_compare(simRL,simNEAT)