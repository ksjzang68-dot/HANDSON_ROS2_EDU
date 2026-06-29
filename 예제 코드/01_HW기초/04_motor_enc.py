from HandsON_BuildHat_API import Motor
import time

motor = Motor('A')

for i in range(2):
    motor.run_for_degrees(90, 50)  # Run motor for 90 degrees at 50% speed
    time.sleep(1)  # Wait for 1 second

for i in range(2):
    motor.run_to_position(0, 50)  # Run motor to position 0 at 50% speed
    time.sleep(1)  # Wait for 1 second
    