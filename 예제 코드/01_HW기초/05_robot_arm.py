from HandsON_BuildHat_API import Motor
import time

motor = Motor('A')

print("Motor start (speed: 20)")
motor.start(-20)  # Start the motor at 20% speed
while True:
    old_degree = motor.get_degrees_counted()
    time.sleep(0.2)
    new_degree = motor.get_degrees_counted()
    if old_degree == new_degree:
        break
print("Motor stop")
motor.stop()  # Stop the motor

time.sleep(1)

motor.run_for_degrees(360,50)
print("Motor Grabbed!")