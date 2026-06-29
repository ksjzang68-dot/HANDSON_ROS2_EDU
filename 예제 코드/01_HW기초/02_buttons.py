from HandsON_BuildHat_API import BuildHat
import time

hat = BuildHat()

while True:
    start_value = hat.start_button.is_pressed()
    stop_value = hat.stop_button.is_pressed()
    print(f'start button : {start_value},  Stop button : {stop_value}')
    time.sleep(0.1)
