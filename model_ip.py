# Cart pole dynamics and modelling

import numpy as np
from numpy import square as sq
from math import cos, pi, sin
import random

class CartPole(object):
    
    simtime = 10
    ctrl_period = 0.04

    # Physical constants
    gravity = 9.8                                       # acceleration due to gravity, positive is downward, (m/sec^2)
    dt = 0.02                                           # time step (s)
    done = False
    crash = False                                       # crash flag
    force = 10                                          # force (N)

    # Cart Pole Attributes
    mcart = 0.711                                       # cart mass (kg)
    mpole = 0.209                                       # pole mass (kg)
    lpole = 0.326                                       # half the pole length (m)
    x_max = 2.4                                         # limits for cart position (m)
    theta_max = np.deg2rad(12)                          # limit for pole angle (rads)

    # Initial parameters
    x0 = 0.0
    dx0 = 0.0
    theta0 = np.pi                                      # 0 deg is straight up
    dtheta0 = 0.0
    
    # Final parameters
    xf = 0.0
    dxf = 0.0
    thetaf = 0.0                                        # 0 deg is straight up
    dthetaf = 0.0

    # Margin for termination condition
    # limit = 1e-3
    x_margin = 0.5
    dx_margin = 0.5
    theta_margin = np.deg2rad(1.0)
    dtheta_margin = np.deg2rad(3.0)

    def __init__(self):

        self.t = 0.0

        # Initialize states
        self.x = self.x0
        self.theta = self.theta0
        self.dx = self.dx0
        self.dtheta = self.dtheta0
        
        self.ddx = 0.0
        self.ddtheta = 0.0

        self.last_ctrl = self.t

        # Initialize data lists for analysis
        self.t_list = [self.t]
        self.x_list = [self.x]
        self.theta_list = [self.theta]
        self.dx_list = [self.dx]
        self.dtheta_list = [self.dtheta]
        self.action_list = [0.0]
    
    def step(self, force):

        # Locals for readability.
        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        mt = mp + mc
        L = self.lpole
        dt = self.dt
        
        # Update position/angle.
        self.x += dt * self.dx
        self.theta += dt * self.dtheta

        # Control frequency limit
        if self.t - self.last_ctrl >= self.ctrl_period or self.ctrl_period == 0.0:
            force_actual = force
            self.last_ctrl = self.t
        else:
            force_actual = self.action_list[-1]

        # Compute new accelerations
        st = sin(self.theta)
        ct = cos(self.theta)
        ddtheta1 = (mt*g*st-ct*(force_actual+st*mp*L*self.dtheta**2))/((4/3)*mt*L-mp*L*ct**2)
        ddx1 = (force_actual+mp*L*(st*self.dtheta**2 - ddtheta1*ct))/mt
        # ddtheta1 = -(L*mp*ct*st*sq(self.dtheta)+force_actual*ct+mt*g*st)/(L*mc+L*mp*(1-sq(ct)))
        # ddx1 = (L*mp*st*sq(self.dtheta)+force_actual+mp*g*ct*st)/(mc+mp*(1-sq(ct)))
        
        # Update velocities.
        self.dx += (ddx1) * dt
        self.dtheta +=  (ddtheta1) * dt
        
        # Remember current acceleration for next step.
        self.ddtheta = ddtheta1
        self.ddx = ddx1
        self.t += dt

        self.t_list.append(self.t)
        self.x_list.append(self.x)
        self.theta_list.append(self.theta)
        self.dx_list.append(self.dx)
        self.dtheta_list.append(self.dtheta)
        self.action_list.append(force_actual)

    def check_done(self):
        if abs(self.x) >= self.x_max:
            self.crash = True
            self.done = True

        delta_x = abs(self.x-self.xf)
        delta_dx = abs(self.dx-self.dxf)
        delta_theta = abs(self.theta-self.thetaf)
        delta_dtheta = abs(self.dtheta-self.dthetaf)
        if (delta_x < self.x_margin) and (delta_dx < self.dx_margin) and (delta_theta < self.theta_margin) and (delta_dtheta < self.dtheta_margin):
            self.done = True
            print(delta_x,delta_dx,np.rad2deg(delta_theta),np.rad2deg(delta_dtheta))

    def get_cstates(self): # continuous state space
        states = [self.x,self.theta,self.dx,self.dtheta]
        return states

    def get_cnstates(self): # continuous (normalized) state space
        states = [0.5*(self.x+self.x_max)/self.x_max,
                  0.5*(self.theta+self.theta_max)/self.theta_max,
                  (self.dx+0.75)/1.5,
                  (self.dtheta+1)/2]
        return states

    def continuous_actuator_force(self,action):
        return 2*self.force*action[0] - self.force

    def print_report(self):
        if self.crash or not self.done:
            print("Agent failed to upright pole.")
        elif not self.crash and self.done:
            print(f"Agent balanced pole in {self.t_list[-1]} seconds.")

        print("Target:----------------------------------------------------")
        print(f"X_0: {self.x0}\ndX_0: {self.dx0}\ntheta_0: {self.theta0}\ndtheta_0: {self.dtheta0}")
        print(f"X_f: {self.xf}\ndX_f: {self.dxf}\ntheta_f: {self.thetaf}\ndtheta_f: {self.dthetaf}")
        print("Actual:----------------------------------------------------")
        print(f"X_f: {self.x_list[-1]}\ndX_f: {self.dx_list[-1]}\ntheta_f: {np.rad2deg(self.theta_list[-1])}\ndtheta_f: {np.rad2deg(self.dtheta_list[-1])}")

    def fitness_ip(self):

        # fitness = 1/sum(self.action_list)
        # fitness = -sq(sq(self.x_list[-1]-self.xf) + sq(self.dx_list[-1]-self.dxf) + sq(self.theta_list[-1]-self.thetaf) + sq(self.dtheta_list[-1]-self.dthetaf))
        # fitness = -sq(self.x_list[-1]-self.xf)-sq(self.dx_list[-1]-self.dxf)-sq(np.rad2deg(self.theta_list[-1]-self.thetaf))-sq(np.rad2deg(self.dtheta_list[-1]-self.dthetaf))
        fitness = -sum(abs(np.rad2deg(np.array(self.theta_list)-self.thetaf)))
        # fitness = -sum(abs(np.array(self.x_list)-self.xf))-sum(abs(np.array(self.dx_list)-self.dxf))-sum(abs(np.rad2deg(np.array(self.theta_list)-self.thetaf)))-sum(abs(np.rad2deg(np.array(self.dtheta_list)-self.dthetaf)))
        
        if self.crash:
            fitness += -1e5
        # elif not self.crash and self.done:
        #     fitness = 1

        return fitness