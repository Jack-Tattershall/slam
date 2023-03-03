import cv2
import numpy as np

fname = "data/sample.mp4"
W = 1024
H = 512

cap = cv2.VideoCapture(fname)

try:
    while cap.isOpened():

        ret, frame = cap.read()

        if ret:

            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            print(frame.shape)

            cv2.imshow("frame", frame)
            cv2.waitKey(1)

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)
