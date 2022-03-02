# Cart pole dynamics and modelling

from math import cos, pi, sin
import random
import numpy as np

class CartPole(object):
    
    simtime = 30
    ctrl_period = 0.04

    # Physical constants
    gravity = 9.8                                       # acceleration due to gravity, positive is downward, (m/sec^2)
    dt = 0.02                                           # time step (s)

    # Cart Pole Attributes
    mcart = 0.711                                       # cart mass (kg)
    mpole = 0.209                                       # pole mass (kg)
    lpole = 0.326                                       # half the pole length (m)
    x_max = 2.4                                         # limits for cart position (m)
    theta_max = np.deg2rad(12)                          # limit for pole angle (rads)

    def __init__(self, x=None, theta=None, dx=None, dtheta=None, box=None, force=None):

        # Randomize initial states
        if x is None:
            x = random.uniform(-0.5 * self.x_max, 0.5 * self.x_max)
        if theta is None:
            theta = random.uniform(-0.5 * self.theta_max, 0.5 * self.theta_max)
        if dx is None:
            dx = random.uniform(-1.0, 1.0)
        if dtheta is None:
            dtheta = random.uniform(-1.0, 1.0)
        if box is None:
            box = 0
        if force is None:
            force = 10
        
        self.box = box
        self.force = force

        self.t = 0.0
        self.x = x
        self.theta = theta
        
        self.dx = dx
        self.dtheta = dtheta
        
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
        # if self.t - self.last_ctrl < self.ctrl_period and self.last_ctrl != 0.0:
        #     force = self.action_list[-1]
        #     print(force)
        # elif self.last_ctrl != 0.0:
        #     self.last_ctrl = self.t
        #     print(self.last_ctrl)
        # print(self.t - self.last_ctrl)

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
        
    def get_x_state(self):
        x = self.x
        if x>=-2.4 and x<-0.8:
            state = 0
        elif x>=-0.8 and x<=0.8:
            state = 1
        elif x>0.8 and x<=2.4:
            state = 2
        else: # This last state occus when cartpole doesn't fall into any bin
            state = 3
        return state

    def get_theta_state(self):
        theta = self.theta
        if theta>=np.deg2rad(-12) and theta<np.deg2rad(-6):
            state = 0
        elif theta>=np.deg2rad(-6) and theta<np.deg2rad(-1):
            state = 1
        elif theta>=np.deg2rad(-1) and theta<np.deg2rad(0):
            state = 2
        elif theta>=np.deg2rad(0) and theta<np.deg2rad(1):
            state = 3
        elif theta>=np.deg2rad(1) and theta<np.deg2rad(6):
            state = 4
        elif theta>=np.deg2rad(6) and theta<=np.deg2rad(12):
            state = 5
        else: # This last state occus when cartpole doesn't fall into any bin
            state = 6
        return state

    def get_dx_state(self):
        dx = self.dx
        if dx<-0.5:
            state = 0
        elif dx>=-0.5 and dx<=0.5:
            state = 1
        elif dx>0.5:
            state = 2
        return state

    def get_dtheta_state(self):
        dtheta = self.dtheta
        if dtheta<np.deg2rad(-50):
            state = 0
        elif dtheta>=np.deg2rad(-50) and dtheta<=np.deg2rad(50):
            state = 1
        elif dtheta>np.deg2rad(50):
            state = 2
        return state

    def get_theta_state2(self):
        theta = self.theta
        if theta>=np.deg2rad(-12) and theta<np.deg2rad(-6):
            state = 0
        elif theta>=np.deg2rad(-6) and theta<np.deg2rad(-4):
            state = 1
        elif theta>=np.deg2rad(-4) and theta<np.deg2rad(-3):
            state = 2
        elif theta>=np.deg2rad(-3) and theta<np.deg2rad(-2):
            state = 3
        elif theta>=np.deg2rad(-2) and theta<np.deg2rad(-1):
            state = 4
        elif theta>=np.deg2rad(-1) and theta<np.deg2rad(0):
            state = 5
        elif theta>=np.deg2rad(0) and theta<np.deg2rad(1):
            state = 6
        elif theta>=np.deg2rad(1) and theta<np.deg2rad(2):
            state = 7
        elif theta>=np.deg2rad(2) and theta<np.deg2rad(3):
            state = 8
        elif theta>=np.deg2rad(3) and theta<np.deg2rad(4):
            state = 9
        elif theta>=np.deg2rad(4) and theta<np.deg2rad(6):
            state = 10
        elif theta>=np.deg2rad(6) and theta<=np.deg2rad(12):
            state = 11
        else: # This last state occus when cartpole doesn't fall into any bin
            state = 12
        return state

    def get_dstates(self): # discrete state space (box 1 and 2)
        if self.box == 1:
            states = (self.get_x_state(),self.get_theta_state(),self.get_dx_state(),self.get_dtheta_state())
        elif self.box == 2:
            states = (self.get_x_state(),self.get_theta_state2(),self.get_dx_state(),self.get_dtheta_state())
        return states

    def get_cstates(self): # continuous state space
        states = [self.x,self.theta,self.dx,self.dtheta]
        return states

    def get_cnstates(self): # continuous (normalized) state space
        states = [0.5*(self.x+self.x_max)/self.x_max,
                  0.5*(self.theta+self.theta_max)/self.theta_max,
                  (self.dx+0.75)/1.5,
                  (self.dtheta+1)/2]
        return states

    def get_dreward(self,states): # for discrete state space (box 1 or 2)
        done = False
        if self.box == 1:
            if states[0] == 3 or states[1] == 6:
                done = True
                reward = -1
            else:
                reward = 0
        elif self.box == 2:
            if states[0] == 3 or states[1] == 12:
                done = True
                reward = -1
            else:
                reward = 0
        return reward, done

    def get_dacreward(self,states): # for discrete state space (box 1 or 2)
        done = False
        if self.box == 1:
            if states[0] == 3 or states[1] == 6:
                done = True
                reward = 0
            else:
                reward = 1
        elif self.box == 2:
            if states[0] == 3 or states[1] == 12:
                done = True
                reward = 0
            else:
                reward = 1
        return reward, done

    def discrete_actuator_force(self,action): # go left when 0, right when 1
        return self.force if action[0] <= 0.5 else -self.force

    def get_creward(self,states): # get rewards with continuous states
        done = False
        if states[0] > self.x_max or states[0] < -self.x_max or states[1] > self.theta_max or states[1] < -self.theta_max:
            done = True
            reward = 0
        else:
            reward = 1
        return reward, done

    def get_cnreward(self,states): # get rewrads with normalized continuous states
        done = False
        if states[0] > 1 or states[0] < 0 or states[1] > 1 or states[1] < 0:
            done = True
            reward = 1
        else:
            reward = 1
        return reward, done

    def fitness(self):
        fitness = len(self.t_list)
        return fitness