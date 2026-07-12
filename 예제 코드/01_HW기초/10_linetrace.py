from HandsON_BuildHat_API import ColorSensor
import time

sensor = ColorSensor('A')
robot = MotorPair('E', 'F')

BASE_SPEED = 30
kp = 0.8

while True:
    reflected = sensor.get_reflected_light()
    error = 50 - reflected
    corrected_error = kp * error
    left_speed = BASE_SPEED + corrected_error
    right_speed = BASE_SPEED - corrected_error

    robot.start_tank(left_speed, right_speed)