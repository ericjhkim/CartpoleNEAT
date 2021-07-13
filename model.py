'''
General settings and implementation of the single-pole cart system dynamics.
'''

from math import cos, pi, sin
import random
import numpy as np
from numpy import square as sq

class CartPole(object):
    
    simtime = 600                                       # simulation duration (s)
    gravity = 9.8                                       # acceleration due to gravity, positive is downward, m/sec^2
    mcart = 0.1655                                      # cart mass in kg (battery approx 45 g) (prev. 0.1205)
    mpole = 0.035                                       # pole mass in kg (without extenders, 0.019) (chopsticks was 0.025)
    lpole = 0.18                                        # half the pole length in meters
    dt = 0.01                                            # time step in seconds
    time_lag = 0.00                                     # control delay
    ufric = 0.5                                         # estimated coefficient of friction
    inert = 1.0
    crash = False                                       # termination criteria

    f_max = 2                                           # maximum control force

    x_max = 2 # m
    theta_max = 2*pi # rads
    theta_max = np.deg2rad(60)
    dx_max = 10 # m/s
    dtheta_max = 200*pi

    def __init__(self, init, final):
        
        self.t = 0.0

        # Initial states
        self.x = init[0]
        self.theta = init[1]
        self.dx = init[2]
        self.dtheta = init[3]

        # Final states
        self.x_1 = final[0]
        self.theta_1 = final[1]
        self.dx_1 = final[2]
        self.dtheta_1 = final[3]
        
        self.xacc = 0.0
        self.tacc = 0.0

        self.x_list = [self.x]
        self.theta_list = [self.theta]
        self.dx_list = [self.dx]
        self.dtheta_list = [self.dtheta]
        self.t_list = [self.t]
        self.ctrl_list = [np.nan]
    
    def step(self, force):
        '''
        Update the system state using leapfrog integration.
            x_{i+1} = x_i + v_i * dt + 0.5 * a_i * dt^2
            v_{i+1} = v_i + 0.5 * (a_i + a_{i+1}) * dt
        '''
        # Locals for readability.
        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        mt = mp + mc
        L = self.lpole
        dt = self.dt
        ufric = self.ufric
        I = self.inert*(4*mp*(L**2)/3)
        
        # Remember acceleration from previous step.
        tacc0 = self.tacc
        xacc0 = self.xacc
        
        # Update position/angle.
        self.x += dt * self.dx + 0.5 * xacc0 * dt ** 2
        self.theta += dt * self.dtheta + 0.5 * tacc0 * dt ** 2

        # # Update states
        # self.x = self.dx*dt
        # self.theta = self.dtheta*dt
                
        # Compute new accelerations as given in "Correct equations for the dynamics of the cart-pole system"
        # by Razvan V. Florian (http://florian.io).
        # http://coneural.org/florian/papers/05_cart_pole.pdf
        st = sin(self.theta)
        ct = cos(self.theta)
        # tacc1 = (g * st + ct * (-force - mp * L * self.dtheta ** 2 * st) / mt) / (L * (4.0 / 3 - mp * ct ** 2 / mt))
        # xacc1 = (force + mp * L * (self.dtheta ** 2 * st - tacc1 * ct)) / mt

        # Online example
        # self.dx = (L*mp*st*sq(self.dtheta)+force+mp*g*ct*st)/(mc+mp*(1-sq(ct)))
        # self.dtheta = -(L*mp*ct*st*sq(self.dtheta)+force*ct+mt*g*st)/(L*mc+L*mp*(1-sq(ct)))

        # Nonlinear EOM (Perez)
        xacc1 = (force-ufric*self.dx+mp*L*((self.dtheta**2)*st-tacc0*ct))/mt
        tacc1 = -mp*L*(xacc1*ct-g*st)/(I+mp*L**2)

        # Linearized EOM (Perez)
        # xacc1 = (force-ufric*self.dx-mp*L*tacc0)/mt
        # tacc1 = mp*L*(xacc1+g*self.theta)/((4*mp*(L**2)/3)+mp*L**2)
        
        # Update velocities.
        self.dx += 0.5 * (xacc0 + xacc1) * dt
        if self.dx >= 0:
            self.dx = min(1.5,self.dx)
        elif self.dx < 0:
            self.dx = max(-1.5,self.dx)
        
        self.dtheta += 0.5 * (tacc0 + tacc1) * dt
        
        # Remember current acceleration for next step.
        self.tacc = tacc1
        self.xacc = xacc1

        self.t += dt

        # Update lists
        self.x_list.append(self.x)
        self.theta_list.append(self.theta)
        self.dx_list.append(self.dx)
        self.dtheta_list.append(self.dtheta)
        self.t_list.append(self.t)
        self.ctrl_list.append(force)
        
    def actuator(self,action): # Based on sigmoid activation function output [0,1]
        f_action = (action*(2*self.f_max)) -self.f_max
        
        return f_action

    def get_lag_state(self):

        # Locals for readability.
        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        mt = mp + mc
        L = self.lpole
        dt = self.dt-self.time_lag
        ufric = self.ufric
        I = self.inert*(4*mp*(L**2)/3)

        # Remember acceleration from previous step.
        tacc0 = self.tacc
        xacc0 = self.xacc

        x = self.x_list[-1]
        dx = self.dx_list[-1]
        theta = self.theta_list[-1]
        dtheta = self.dtheta_list[-1]
        
        try:
            force = self.ctrl_list[-1]
        except:
            force = 0

        st = sin(theta)
        ct = cos(theta)

        # Nonlinear EOM (Perez)
        xacc1 = (force-ufric*dx+mp*L*((dtheta**2)*st-tacc0*ct))/mt
        tacc1 = -mp*L*(xacc1*ct-g*st)/(I+mp*L**2)
        
        # Update position/angle.
        x += dt * dx + 0.5 * xacc0 * dt ** 2
        theta += dt * dtheta + 0.5 * tacc0 * dt ** 2

        # Update velocities.
        dx += 0.5 * (xacc0 + xacc1) * dt
        if dx >= 0:
            dx = min(1.5,dx)
        elif dx < 0:
            dx = max(-1.5,dx)
        
        dtheta += 0.5 * (tacc0 + tacc1) * dt

        return [0.5 * (x + self.x_max) / self.x_max,
                (dx + 0.75) / 1.5,
                0.5 * (theta + self.theta_max) / self.theta_max,
                (dtheta + 1.0) / 2.0]

    def get_scaled_state(self):
        '''Get full state, scaled into (approximately) [0, 1].'''
        return [0.5 * (self.x + self.x_max) / self.x_max,
                0.5 * (self.theta + self.theta_max) / self.theta_max,
                (self.dx + 0.75) / 1.5,
                (self.dtheta + 1.0) / 2.0]
        
    def fitness(self):

        # Boundary and continuity penalties
        # x_high = sum([abs(i-self.x_max) for i in self.x_list if i > self.x_max])
        # theta_high = sum([abs(i-self.theta_max) for i in self.theta_list if i > self.theta_max])
        # dx_high = sum([abs(i-self.dx_max) for i in self.dx_list if i > self.dx_max])
        # dtheta_high = sum([abs(i-self.dtheta_max) for i in self.dtheta_list if i > self.dtheta_max])

        # x_low = sum([abs(i+self.x_max) for i in self.x_list if i < -self.x_max])
        # theta_low = sum([abs(i+self.theta_max) for i in self.theta_list if i < -self.theta_max])
        # dx_low = sum([abs(i+self.dx_max) for i in self.dx_list if i <- self.dx_max])
        # dtheta_low = sum([abs(i+self.dtheta_max) for i in self.dtheta_list if i < -self.dtheta_max])

        # x_high /= self.x_max
        # theta_high /= self.theta_max
        # dx_high /= self.dx_max
        # dtheta_high /= self.dthetaa_max

        # x_low /= self.x_max
        # theta_low /= self.theta_max
        # dx_low /= self.dx_max
        # dtheta_low /= self.dthetaa_max

        x_pen = sum([abs(i-np.sign(i)*self.x_max) for i in self.x_list if abs(i) > self.x_max])
        theta_pen = sum([abs(i-np.sign(i)*self.theta_max) for i in self.theta_list if abs(i) > self.theta_max])
        dx_pen = sum([abs(i-np.sign(i)*self.dx_max) for i in self.dx_list if abs(i) > self.dx_max])
        dtheta_pen = sum([abs(i-np.sign(i)*self.dtheta_max) for i in self.dtheta_list if abs(i) > self.dtheta_max])

        x_pen /= self.x_max
        theta_pen /= self.theta_max
        dx_pen /= self.dx_max
        dtheta_pen /= self.dtheta_max

        total_pen = sum(sq(np.array([x_pen,theta_pen,dx_pen,dtheta_pen])))

        fitness = -total_pen/1e5

        fitness = self.t

        if self.crash:
            fitness = -1e10

        return fitness

def discrete_actuator_force(action):
    return 2 if action[0] > 0.5 else -2

def continuous_actuator_force(action):
    return -10.0 + 2.0 * action[0]
    
def noisy_continuous_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0

def noisy_discrete_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0
