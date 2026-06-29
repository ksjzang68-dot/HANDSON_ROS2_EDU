from bno055 import BNO055
import time

with BNO055(bus=7) as sensor:
    while True:
        yaw, pitch, roll = sensor.euler
        print(f'\rYaw: {yaw:7.2f}°  Pitch: {pitch:7.2f}°  Roll: {roll:7.2f}°', end='', flush=True)
        time.sleep(0.01)
