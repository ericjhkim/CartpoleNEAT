from scipy.io import savemat
import numpy as np

def py2mat(sim,directory):
    mdic = {'t_list':sim.t_list,
            'x_list':sim.x_list,
            'theta_list':sim.theta_list,
            'dx_list':sim.dx_list,
            'dtheta_list':sim.dtheta_list}

    savemat(directory,mdic)