import cv2
import numpy as np
import time
from extractor import KeypointExtractor

fname = "data/sample.mp4"
W = 1024
H = 512

cap = cv2.VideoCapture(fname)
extractor = KeypointExtractor()


try:
    while cap.isOpened():

        ret, frame = cap.read()

        if ret:

            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            kp, des = extractor.extract_keypoints(frame)
            frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
            print(des.shape)

            cv2.imshow(fname, frame)
            cv2.waitKey(1)


except KeyboardInterrupt:

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)
