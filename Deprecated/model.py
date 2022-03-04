'''
General settings and implementation of the single-pole cart system dynamics.
'''

from math import cos, pi, sin
import random
import numpy as np
from numpy import square as sq

class CartPole(object):
    
    simtime = 200                                        # simulation duration (s)
    gravity = 9.8                                       # acceleration due to gravity, positive is downward, m/sec^2
    mcart = 0.105                                       # cart mass in kg (battery approx 45 g) (prev. 0.1205)
    mpole = 0.0813                                      # pole mass in kg (without extenders, 0.019) (chopsticks was 0.025)
    lpole = 0.1                                         # half the pole length in meters
    dt = 0.01                                            # time step in seconds
    time_lag = 0.00                                     # control delay
    ufric = 0.5                                         # estimated coefficient of friction
    inert = 1
    crash = False                                       # termination criteria

    # Neurocontroller settings
    x_max = 1 # m
    theta_max = np.deg2rad(60)
    dx_max = 10 # m/s
    dtheta_max = 2*pi

    def __init__(self, x=None, theta=None, dx=None, dtheta=None, inertia=None, cfric=None,):
        
        if x is None:
            x = random.uniform(-0.5 * self.x_max, 0.5 * self.x_max)
        #end
        if theta is None:
            theta = random.uniform(-0.5 * self.theta_max, 0.5 * self.theta_max)
        #end
        if dx is None:
            dx = random.uniform(-1.0, 1.0)
        #end
        if dtheta is None:
            dtheta = random.uniform(-1.0, 1.0)
        #end

        if inertia is None:
            inertia = random.uniform(0.5, 1.5)
        #end
        if cfric is None:
            cfric = random.uniform(0, 1)
        #end

        self.f_max = np.random.uniform(5,15)
        self.prev_ctrl = 0.0
        self.prev_ctrl_time = 0.0

        self.inert = inertia
        self.ufric = cfric

        self.t = 0.0
        self.x = x
        self.theta = theta
        
        self.dx = dx
        self.dtheta = dtheta
        
        self.xacc = 0.0
        self.tacc = 0.0

        # Initialize lists
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
        # self.x += self.dx*dt
        # self.theta += self.dtheta*dt
                
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
            self.dx = min(self.dx_max,self.dx)
        elif self.dx < 0:
            self.dx = max(-self.dx_max,self.dx)
        
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

    def get_states(self):
        return [self.x, self.dx, self.theta, self.dtheta]

    def discrete_actuator_force(self,action):
        return self.f_max if action[0] > 0.5 else -self.f_max

    def continuous_motor_force(self,action):
        force = 2*self.f_max*action[0] - self.f_max
        return force

    def fitness(self):

        fitness = self.t

        if self.crash:
            fitness += -1e5

        return fitness

    # def step2(self,force): # Simplified EOM
    #     # Locals for readability.
    #     g = self.gravity
    #     mp = self.mpole
    #     mc = self.mcart
    #     mt = mp + mc
    #     L = self.lpole*2
    #     dt = self.dt
        
    #     # Update states
    #     self.x += self.dx*dt
    #     self.theta += self.dtheta*dt

    #     st = sin(self.theta)
    #     ct = cos(self.theta)

    #     # Update derivative states
    #     self.dx = (L*mp*st*sq(self.dtheta)+force+mp*g*ct*st)/(mc+mp*(1-sq(ct)))
    #     self.dtheta = -(L*mp*ct*st*sq(self.dtheta)+force*ct+mt*g*st)/(L*mc+L*mp*(1-sq(ct)))

    #     self.t += dt

    #     # Update lists
    #     self.x_list.append(self.x)
    #     self.theta_list.append(self.theta)
    #     self.dx_list.append(self.dx)
    #     self.dtheta_list.append(self.dtheta)
    #     self.t_list.append(self.t)
    #     self.ctrl_list.append(force)
        
    # def get_lag_state(self):

    #     # Locals for readability.
    #     g = self.gravity
    #     mp = self.mpole
    #     mc = self.mcart
    #     mt = mp + mc
    #     L = self.lpole
    #     dt = self.dt-self.time_lag
    #     ufric = self.ufric
    #     I = self.inert*(4*mp*(L**2)/3)

    #     # Remember acceleration from previous step.
    #     tacc0 = self.tacc
    #     xacc0 = self.xacc

    #     x = self.x_list[-1]
    #     dx = self.dx_list[-1]
    #     theta = self.theta_list[-1]
    #     dtheta = self.dtheta_list[-1]
        
    #     try:
    #         force = self.ctrl_list[-1]
    #     except:
    #         force = 0

    #     st = sin(theta)
    #     ct = cos(theta)

    #     # Nonlinear EOM (Perez)
    #     xacc1 = (force-ufric*dx+mp*L*((dtheta**2)*st-tacc0*ct))/mt
    #     tacc1 = -mp*L*(xacc1*ct-g*st)/(I+mp*L**2)
        
    #     # Update position/angle.
    #     x += dt * dx + 0.5 * xacc0 * dt ** 2
    #     theta += dt * dtheta + 0.5 * tacc0 * dt ** 2

    #     # Update velocities.
    #     dx += 0.5 * (xacc0 + xacc1) * dt
    #     if dx >= 0:
    #         dx = min(1.5,dx)
    #     elif dx < 0:
    #         dx = max(-1.5,dx)
        
    #     dtheta += 0.5 * (tacc0 + tacc1) * dt

    #     return [0.5 * (x + self.x_max) / self.x_max,
    #             (dx + 0.75) / 1.5,
    #             0.5 * (theta + self.theta_max) / self.theta_max,
    #             (dtheta + 1.0) / 2.0]

    # def get_scaled_state(self):
    #     '''Get full state, scaled into (approximately) [0, 1].'''
    #     return [0.5 * (self.x + self.x_max) / self.x_max,
    #             0.5 * (self.theta + self.theta_max) / self.theta_max,
    #             (self.dx + 0.75) / 1.5,
    #             (self.dtheta + 1.0) / 2.0,
    #             self.t/self.simtime]
        
    # def actuator(self,action):
    #     f_action = (action[0]*(2*self.f_max)) - self.f_max
    #     return f_action

#     def fitness(self):

#         # Boundary penalties
#         x_pen = sum([abs(i-np.sign(i)*self.x_max) for i in self.x_list if abs(i) > self.x_max])
#         theta_pen = sum([abs(i-np.sign(i)*self.theta_max) for i in self.theta_list if abs(i) > self.theta_max])
#         dx_pen = sum([abs(i-np.sign(i)*self.dx_max) for i in self.dx_list if abs(i) > self.dx_max])
#         dtheta_pen = sum([abs(i-np.sign(i)*self.dtheta_max) for i in self.dtheta_list if abs(i) > self.dtheta_max])

#         total_pen = sum(sq(np.array([x_pen,theta_pen,dx_pen,dtheta_pen])))

#         # Final state penalties
#         x_diff = abs(self.x-self.x_1)/(2*self.x_max)
#         theta_diff = np.rad2deg(abs(self.theta-self.theta_1))
#         dx_diff = abs(self.dx-self.dx_1)
#         dtheta_diff = np.rad2deg(abs(self.dtheta-self.theta_1))

#         state_pen = sum(sq(np.array([x_diff,theta_diff,dx_diff,dtheta_diff])))

#         # # Rewards for being in stable region
#         # x_rew = sum([self.dt for i in self.x_list if i <= self.x_1+self.e_x or i >= self.x_1-self.e_x])
#         # theta_rew = sum([self.dt for i in self.theta_list if i <= self.theta_1+self.e_theta or i >= self.theta_1-self.e_theta])
#         # # dx_rew = sum([self.dt for i in self.dx_list if i <= self.dx_1+self.e_dx or i >= self.dx_1-self.e_dx])
#         # # dtheta_rew = sum([self.dt for i in self.dtheta_list if i <= self.dtheta_1+self.e_dtheta or i >= self.dtheta_1-self.e_dtheta])

#         # # print(x_rew,theta_rew,dx_rew,dtheta_rew)
#         # # rewards = sum(sq(np.array([x_rew,theta_rew,dx_rew,dtheta_rew])))
#         # rewards = sum(sq(np.array([x_rew,theta_rew])))

#         # fitness = rewards*0-self.t*(total_pen+state_pen)

#         fitness = self.t

#         if self.crash:
#             fitness = -1e10

#         return fitness

#     def simple_fitness(self):

#         # Final state penalties
#         x_diff = abs(self.x-self.x_1)/(2*self.x_max)
#         theta_diff = np.rad2deg(abs(self.theta-self.theta_1))
#         dx_diff = abs(self.dx-self.dx_1)
#         dtheta_diff = np.rad2deg(abs(self.dtheta-self.theta_1))

#         state_pen = sum(sq(np.array([x_diff,theta_diff,dx_diff,dtheta_diff])))
        
#         fitness = -self.t-state_pen
#         # fitness = -self.t

#         if self.crash:
#             fitness = -1e10

#         return fitness

# def continuous_actuator_force(action):
#     return -10.0 + 2.0 * action[0]
    
# def noisy_continuous_actuator_force(action):
#     a = action[0] + random.gauss(0, 0.2)
#     return 10.0 if a > 0.5 else -10.0

# def noisy_discrete_actuator_force(action):
#     a = action[0] + random.gauss(0, 0.2)
#     return 10.0 if a > 0.5 else -10.0
