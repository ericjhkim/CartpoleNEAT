import serial

arduino = serial.Serial('COM6', 9600, timeout=.1)

# def send(cmd,ardobj):

cmd = ""
cmdstr = str(cmd) + "\n"

# Include a filter to remove values outside +-2^7 bounds

while True:
    if cmd:
        arduino.write(cmdstr.encode())

    data = arduino.readline()[:-2] #range removes the new-line chars
    if data:
        print(data)
