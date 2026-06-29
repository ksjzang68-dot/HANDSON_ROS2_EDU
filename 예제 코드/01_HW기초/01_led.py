from HandsON_BuildHat_API import BuildHat
import time

hat = BuildHat()

while True:
    hat.start_led.set(1)
    time.sleep(1)
    hat.start_led.set(0)
    time.sleep(1)
    hat.stop_led.set(1)
    time.sleep(1)
    hat.stop_led.set(0)
    time.sleep(1)


    