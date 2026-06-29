from HandsON_BuildHat_API import Motor
import time

motor = Motor('A')

motor.set(50)  # Set motor speed to 50%
time.sleep(2)
motor.set(0)   # Stop the motor
