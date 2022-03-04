import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams['axes.grid'] = True
plt.rcParams['savefig.dpi'] = 300

def states(sim):

    fig, axs = plt.subplots(2,2,figsize=(12,9))
    plt.subplots_adjust(hspace = 0.35)
    # fig.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace = 0.28)

    t = sim.t_list

    axs[0,0].set_title('Cart position (x)')
    axs[0,0].plot(t, sim.x_list)
    axs[0,0].plot(t, sim.x_max*np.ones(len(sim.x_list)),'r',alpha=0.5)
    axs[0,0].plot(t, -sim.x_max*np.ones(len(sim.x_list)),'r',alpha=0.5)
    axs[0,0].set_ylabel('m')
    axs[0,0].set_xlabel('Time, s')

    axs[0,1].set_title('Pole angle ('u'\u03b8)')
    axs[0,1].plot(t, np.rad2deg(sim.theta_list))
    axs[0,1].plot(t, np.rad2deg(sim.theta_max)*np.ones(len(sim.theta_list)),'r',alpha=0.5)
    axs[0,1].plot(t, np.rad2deg(-sim.theta_max)*np.ones(len(sim.theta_list)),'r',alpha=0.5)
    axs[0,1].set_ylabel('deg')
    axs[0,1].set_xlabel('Time, s')

    axs[1,0].set_title('Velocity (dx)')
    axs[1,0].plot(t, sim.dx_list)
    axs[1,0].set_ylabel('m/s')
    axs[1,0].set_xlabel('Time, s')

    axs[1,1].set_title('Angular velocity (d'u'\u03b8)')
    axs[1,1].plot(t, np.rad2deg(sim.dtheta_list))
    axs[1,1].set_ylabel('deg/s')
    axs[1,1].set_xlabel('Time, s')

    # Axis labels
    for i in range(2):
        for j in range(2):
            for item in ([axs[i,j].title, axs[i,j].xaxis.label, axs[i,j].yaxis.label]):
                item.set_fontsize(18)
            for item in (axs[i,j].get_xticklabels() + axs[i,j].get_yticklabels()):
                item.set_fontsize(14)

    plt.show()

def states_vert(sim):

    fig, axs = plt.subplots(3,figsize=(9,12))
    plt.subplots_adjust(hspace = 0.35)
    # fig.tight_layout(pad=0.5)
    # plt.subplots_adjust(wspace = 0.28)

    t = sim.t_list

    axs[0].set_title('Cart position (x)')
    axs[0].plot(t, sim.x_list)
    axs[0].plot(t, sim.x_max*np.ones(len(sim.x_list)),'r',alpha=0.5)
    axs[0].plot(t, -sim.x_max*np.ones(len(sim.x_list)),'r',alpha=0.5)
    axs[0].set_ylabel('m')
    # axs[0].set_xlabel('Time, s')

    axs[1].set_title('Pole angle ('u'\u03b8)')
    axs[1].plot(t, np.rad2deg(sim.theta_list))
    axs[1].plot(t, np.rad2deg(sim.theta_max)*np.ones(len(sim.theta_list)),'r',alpha=0.5)
    axs[1].plot(t, np.rad2deg(-sim.theta_max)*np.ones(len(sim.theta_list)),'r',alpha=0.5)
    axs[1].set_ylabel('deg')
    # axs[1].set_xlabel('Time, s')

    axs[2].set_title('Control force')
    axs[2].plot(t, sim.action_list)
    axs[2].set_ylabel('N')
    axs[2].set_xlabel('Time, s')

    # Axis labels
    for i in range(3):
        for item in ([axs[i].title, axs[i].xaxis.label, axs[i].yaxis.label]):
            item.set_fontsize(18)
        for item in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            item.set_fontsize(14)
        # axs[i].set_xlim([30-5.5,30])

    legend = [Line2D([0], [0], color='b', label='State'),
              Line2D([0], [0], color='r', label='Limits')]

    fig.legend(handles=legend,loc=(0.758,0.9),fontsize=14)

    plt.show()

def states_compare(sim1,sim2):

    fig, axs = plt.subplots(3,figsize=(9,12))
    plt.subplots_adjust(hspace = 0.35)
    # fig.tight_layout(pad=0.5)
    # plt.subplots_adjust(wspace = 0.28)

    t = sim1.t_list
    duration_1 = t[-1]-t[0]
    t2 = sim2.t_list
    t2_dur1 = np.argmin(abs(np.array(t2)-(t2[-1]-duration_1)))
    t2 = t2[t2_dur1:]
    t2 = np.array(t2)-t2[0]
    # Normalize time axis
    t /= np.array(t[-1])
    t2 /= t2[-1]

    axs[0].set_title('Cart position (x)')
    # RL
    axs[0].plot(t, sim1.x_list)
    axs[0].plot(t, sim2.x_max*np.ones(len(sim1.x_list)),':r')
    axs[0].plot(t, -sim2.x_max*np.ones(len(sim1.x_list)),':r')
    # NEAT
    axs[0].plot(t2, sim2.x_list[t2_dur1:],linestyle="--")
    axs[0].set_ylabel('m')

    axs[1].set_title('Pole angle ('u'\u03b8)')
    # RL
    axs[1].plot(t, np.rad2deg(sim1.theta_list))
    axs[1].plot(t, np.rad2deg(sim2.theta_max)*np.ones(len(sim1.theta_list)),':r')
    axs[1].plot(t, np.rad2deg(-sim2.theta_max)*np.ones(len(sim1.theta_list)),':r')
    # NEAT
    axs[1].plot(t2, np.rad2deg(sim2.theta_list[t2_dur1:]),linestyle="--")
    axs[1].set_ylabel('deg')

    axs[2].set_title('Control force')
    # RL
    axs[2].plot(t, sim1.action_list)
    # NEAT
    axs[2].plot(t2, sim2.action_list[t2_dur1:],linestyle="--")
    axs[2].set_ylabel('N')
    axs[2].set_xlabel(r'Normalized time, $t/t_f$')

    # Axis labels
    for i in range(3):
        for item in ([axs[i].title, axs[i].xaxis.label, axs[i].yaxis.label]):
            item.set_fontsize(18)
        for item in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            item.set_fontsize(14)
        # axs[i].set_xlim([30-5.5,30])

    legend = [Line2D([0], [0], color='b', label='RL'),
              Line2D([0], [0], color='orange', linestyle="--", label='NEAT'),
              Line2D([0], [0], color='r', linestyle=":", label='Limits')]

    fig.legend(handles=legend,loc=(0.758,0.9),fontsize=14)

    plt.show()

def ip_states(sim):
    fig, axs = plt.subplots(3,figsize=(9,12))
    plt.subplots_adjust(hspace = 0.35)

    axs[0].plot(sim.t_list,sim.x_list,'ro-',label='x')
    axs[0].plot(sim.t_list,sim.dx_list,'ko-',label='dx')

    axs[1].plot(sim.t_list,np.rad2deg(sim.theta_list),'ro-',label='theta')
    axs[1].plot(sim.t_list,np.rad2deg(sim.dtheta_list),'ko-',label='dtheta')

    axs[2].plot(sim.t_list,sim.action_list,'o-',label='F')

    for i in range(3):
        axs[i].set_xlabel("Time, s")
        axs[i].legend()

    plt.show()
