# import serial
# import time
# import mycomms as fromArduino

# arduino = serial.Serial('COM6', 9600, timeout=.1)

# import markercomms as comms

# comms.setupSerial(115200, 'COM6')

# rxdata = fromArduino.receiveData()
# print(rxdata)

#####__________________________

# while(True):
#     comms.sendToArduino("t")
#     arduinoReply = comms.recvLikeArduino()
#     if not (arduinoReply == 'XXX'):
#         print(arduinoReply.splitlines())
#         break

# count = 0
# prevTime = time.time()
# while True:
#             # check for a reply
#     arduinoReply = comms.recvLikeArduino()
#     # print(arduinoReply)
#     if not (arduinoReply == 'XXX'):
#         print (arduinoReply)
        
#         # send a message at intervals
#     if time.time() - prevTime > 1.0:
#         comms.sendToArduino("t")
#         prevTime = time.time()
#         count += 1
#         print("sent...")
#     # print(str(comms.serialPort.in_waiting)+"_"+str(comms.serialPort.out_waiting)+"_"+str(count))

# toArduino.send("c1",arduino)

# cmd = ""
# cmdstr = cmd + "\n"

# while(arduino.readline()[:-2]!='<'):
#     # if cmd:
#     #     arduino.write(cmdstr.encode())
#     # print(arduino.in_waiting)
#     data = fromArduino.receive(arduino)
#     print(data)

#     # data = arduino.readline()[:-2]
#     # # data = fromArduino.receive(arduino)
#     # # while data == "Failed RX":
#     # #     data = fromArduino.receive(arduino)
#     # print(data)

# cmd = "t"
# cmdstr = cmd + "\n"
# arduino.write(cmdstr.encode())
# for i in range(4):
#     data = arduino.readline()[:-2]
#     print(data)

from __future__ import print_function

import os
import pickle
import time

from movie import make_movie

import neat
from neat import nn

from math import pi

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# load the winner
with open('winner-feedforward', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)