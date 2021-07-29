import matplotlib.pyplot as plt
import numpy as np
from numpy import square as sq
import model

def states(sim):
    fig = plt.figure()
    plt.title('Cart pole states')

    plt.subplot(2,2,1)
    plt.plot(sim.t_list,sim.x_list,'r-',label='x')
    # plt.plot(sim.t_list,np.ones(len(sim.t_list))*sim.x_max,'r')
    # plt.plot(sim.t_list,np.ones(len(sim.t_list))*-sim.x_max,'r')
    plt.plot(sim.t_list,sim.dx_list,'k-',label='dx')
    # plt.plot(sim.t_list,np.ones(len(sim.t_list))*sim.dx_max,'k')
    # plt.plot(sim.t_list,np.ones(len(sim.t_list))*-sim.dx_max,'k')
    # plt.plot(t[0][:-1],np.diff(x[0][:,0])/np.diff(t[0]))
    plt.xlabel('time')
    plt.legend()
    plt.title('Cart Pole states')

    plt.subplot(2,2,2)
    plt.plot(sim.t_list,np.rad2deg(sim.theta_list),'r-',label='theta')
    # plt.plot(sim.t_list,np.ones(len(sim.t_list))*np.rad2deg(sim.theta_max),'r')
    # plt.plot(sim.t_list,np.ones(len(sim.t_list))*np.rad2deg(-sim.theta_max),'r')
    plt.plot(sim.t_list,np.rad2deg(sim.dtheta_list),'k-',label='dtheta')
    # plt.plot(sim.t_list,np.ones(len(sim.t_list))*np.rad2deg(sim.dtheta_max),'k')
    # plt.plot(sim.t_list,np.ones(len(sim.t_list))*np.rad2deg(-sim.dtheta_max),'k')
    plt.xlabel('time')
    plt.legend()
    plt.title('Cart Pole states')

    plt.subplot(2,1,2)
    plt.plot(sim.t_list,sim.ctrl_list,'b-')
    plt.xlabel('time')
    plt.title('Cart Pole controls')
    plt.legend(['F'])

    plt.show()