import time
from pynput import keyboard
import pandas as pd

listOfObjects = pd.DataFrame(columns=['c<', 'c>', 'o<', 'o>', 'r<', 'r>', 'r<', 'r>', 'e<',
                             'e>', 'c<', 'c>', 't<', 't>', 'h<', 'h>', 'o<', 'o>', 'r<', 'r>', 's<', 's>', 'e<', 'e>',])

# Array to store the log of key presses
keystroke_log = []

# Variables to track the timing
lastActionTime = time.time()


def on_press(key):
    global lastActionTime, keystroke_log

    currentTime = time.time()
    timeUnPressed = (currentTime - lastActionTime)
    keystroke_log.append([key, timeUnPressed, 0])
    lastActionTime = currentTime


def on_release(key):
    global lastActionTime, keystroke_log

    current_time = time.time()
    time_pressed = (current_time - lastActionTime)
    keystroke_log[-1][2] = time_pressed
    lastActionTime = current_time

    # Stop listener if Esc is pressed
    if key == keyboard.Key.enter:
        return False


persona = input('Quien eres?\n>>> ')

print(f'Va, gracias {persona}. Antes que nada, vamos a pedirte que escribas hola un total de 50 veces. Plis hazlo. La prueba inicia en 5 segundos...')

time.sleep(5)

i = 0

while i < 50:
    print(f'Paso {i}: ')

    keystroke_log = []

    # Start listening to keyboard events
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    input()

    keystroke_log = keystroke_log[:-1]

    written = ''.join([str(j[0]) for j in keystroke_log]).replace("'", '')

    if written != 'correcthorse':
        print('Escribe correcthorse plis')
        continue

    toSave = [j for k in keystroke_log for j in k[1:]]

    listOfObjects.loc[len(listOfObjects)] = toSave

    i += 1


listOfObjects.to_csv(f'{persona}_password.csv')
