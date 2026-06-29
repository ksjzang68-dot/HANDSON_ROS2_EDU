from HandsON_BuildHat_API import MotorPair
import time

robot = MotorPair('E', 'F')

print("Drive forward (speed: 50)")
robot.start(0,50)
time.sleep(2)
robot.stop()
print("Stopped!")