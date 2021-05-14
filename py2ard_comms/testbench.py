import serial
import send as toArduino

arduino = serial.Serial('COM6', 9600, timeout=.1)

toArduino.send(400)