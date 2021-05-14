import serial
import math
import time
import struct

# arduino = serial.Serial('COM6', 9600, timeout=.1)
# while True:
# 	data = arduino.readline()[:-2] #the last bit gets rid of the new-line chars
# 	if data:
# 		#print(data)
# 		z = float(data) #collects z-axis measurements
# 		#print(z)
# 		#
# 		theta = 180*math.asin(z)/math.pi
# 		print(theta)

arduino = serial.Serial('COM6', 9600, timeout=.1)
time.sleep(1) #give the connection a second to settle

# Preformat tx data
motorspeed = -400
motorstring = str(motorspeed) + "\n"

# Sending to Arduino
while True:

	# Reading sensor
	data = arduino.readline()[:-2] #the last bit gets rid of the new-line chars
	if data:
		z = float(data) #collects z-axis measurements
		theta = 180*math.asin(z)/math.pi
		print(theta)

	arduino.write(motorstring.encode())

	# Checking if Arduino received
	data = arduino.readline()[:-2] #range removes the new-line chars
	if data:
		print(data)
