import os

def writeLogIntoFileAndPrint(message):
    if not os.path.exists('log.txt'):
        with open('log.txt', 'w') as f:
            # file operations here
            pass
    with open('log.txt', 'a') as file:
        print(message)
        file.write(message + '\n')