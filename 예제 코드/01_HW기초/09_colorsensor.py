from HandsON_BuildHat_API import ColorSensor
import time

sensor = ColorSensor('A')

while True:
    color = sensor.get_color()
    reflected = sensor.get_reflected_light()
    rgb = sensor.get_rgb_intensity()

    print(f'color : {color}, reflected : {reflected}, rgb : {rgb}')
    time.sleep(0.1)