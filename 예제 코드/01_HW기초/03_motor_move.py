from HandsON_BuildHat_API import Motor
import time

motor = Motor('A')

motor.start(50)  # Set motor speed to 50%
time.sleep(2)
motor.stop()   # Stop the motor
