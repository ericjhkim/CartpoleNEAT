'''
General settings and implementation of the single-pole cart system dynamics.
'''

from math import cos, pi, sin
import random

import serial
import receive as fromArduino
arduino = serial.Serial('COM6', 9600, timeout=.1)


class CartPole(object):
    
    gravity = 9.8  # acceleration due to gravity, positive is downward, m/sec^2
    mcart = 0.1205  # cart mass in kg
    mpole = 0.019  # pole mass in kg
    lpole = 0.15/2  # half the pole length in meters

    def __init__(self, x=None, theta=None, dx=None, dtheta=None,position_limit=1, angle_limit_radians=45 * pi / 180):
        
        self.position_limit = position_limit
        self.angle_limit_radians = angle_limit_radians
        
        data = fromArduino.receive(arduino)
            while data == "Failed RX":
            data = fromArduino.receive(arduino)

        self.theta = data[0]
        self.dtheta = data[1]
        self.x = data[2]
        self.dx = data[3]
        
    def get_scaled_state(self):
        '''Get full state, scaled into (approximately) [0, 1].'''
        data = fromArduino.receive(arduino)
            while data == "Failed RX":
            data = fromArduino.receive(arduino)

        self.theta = data[0]
        self.dtheta = data[1]
        self.x = data[2]
        self.dx = data[3]

        return [0.5 * (self.x + self.position_limit) / self.position_limit,
                (self.dx + 0.75) / 1.5,
                0.5 * (self.theta + self.angle_limit_radians) / self.angle_limit_radians,
                (self.dtheta + 1.0) / 2.0]
        

def continuous_actuator_force(action):
    return -10.0 + 2.0 * action[0]
    

def noisy_continuous_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0
    

def discrete_actuator_force(action):
    return "10" if action[0] > 0.5 else "-10"
    

def noisy_discrete_actuator_force(action):
    a = action[0] + random.gauss(0, 0.2)
    return 10.0 if a > 0.5 else -10.0
