import serial

def send(motorspeed):
        
    arduino = serial.Serial('COM6', 9600, timeout=.1)
    time.sleep(1) #give the connection a second to settle

    # Preformat tx data
    motorstring = str(motorspeed) + "\n"

    # Sending to Arduino
    while True:

        arduino.write(motorstring.encode())

        # Checking if Arduino received
        data = arduino.readline()[:-2] #range removes the new-line chars
        if data:
            print(data)
