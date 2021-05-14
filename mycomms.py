import serial
import math
import markercomms as comms

def receive(ardobj):
    while True:
        data = ardobj.readline()[:-2]
        if data.decode() == "<":

            try:
                theta = float(ardobj.readline()[:-2])
                # theta = -math.asin(xlZ) #rads
            except:
                theta = 0

            try:
                gX = float(ardobj.readline()[:-2])
                dtheta = -gX*math.pi/180 #rads/s
            except:
                dtheta = 0

            try:
                x = float(ardobj.readline()[:-2]) #metres
                if x == -0.0:
                    x = 0.0
            except:
                x = 0

            try:
                dx = float(ardobj.readline()[:-2]) #m/s
            except:
                dx = 0

            data = [theta,dtheta,x,dx]
            defaultdata = data

            return(data)
            break


def send(ardobj,cmd):
    ardobj.write(cmd.encode())

def receiveData():
    while True:
        comms.sendToArduino("t")
        arduinoReply = comms.recvLikeArduino()
        if not (arduinoReply == 'XXX'):
            try:
                xlZ = float(arduinoReply.splitlines()[0])
                # theta = -math.asin(xlZ) #rads
            except:
                theta = 0
            try:
                gX = float(arduinoReply.splitlines()[1])
                dtheta = -gX*math.pi/180 #rads/s
            except:
                dtheta = 0
            try:
                x = float(arduinoReply.splitlines()[2]) #metres
                if x == -0.0:
                    x = 0.0
            except:
                x = 0
            try:
                dx = float(arduinoReply.splitlines()[3]) #m/s
            except:
                dx = 0

            data = [theta,dtheta,x,dx]
            return(data)
            
            break