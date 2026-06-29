from HandsON_BuildHat_API import MotorPair
import time

robot = MotorPair('E', 'F')

print("Drive forward (speed: 30)")
robot.start_tank(30,30)
time.sleep(2)
robot.stop()
print("Stopped!")

