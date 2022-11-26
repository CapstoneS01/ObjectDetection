# Retrieves stream and displays it using OpenCV

import cv2
from djitellopy import Tello

tello = Tello()
tello.connect()

tello.streamoff()
tello.streamon()

while True:
  frame_read = tello.get_frame_read()
  frame = frame_read.frame

  img = cv2.resize(frame, (320, 240))

  cv2.imshow("Stream", img)

  key = cv2.waitKey(1) & 0xFF == ord('q')
  if key:
    break
