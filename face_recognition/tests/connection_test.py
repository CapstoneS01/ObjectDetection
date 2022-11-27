from djitellopy import tello
from time import sleep


def main():

    print("1. Connection Test: ")
    drone = tello.Tello()
    drone.connect
    print("\n")
    print("2. Stream Test: ")
    drone.streamon()
    print("\n")

    drone.end()


    main()
