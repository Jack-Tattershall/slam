import cv2
import numpy as np
import time
from extractor import KeypointExtractor

fname = "data/sample.mp4"
W = 1024
H = 512

cap = cv2.VideoCapture(fname)
extractor = KeypointExtractor()

is_first = True
try:
    while cap.isOpened():

        ret, frame = cap.read()
        if is_first:
            last_frame = frame
            is_first = False

        if ret:

            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            kp_frame, des_frame = extractor.extract_keypoints(frame)
            kp_last_frame, des_last_frame = extractor.extract_keypoints(last_frame)
            matches = extractor.match_keypoints(des_frame, des_last_frame)
            extractor.compute_lines(matches)

            frame = cv2.drawKeypoints(frame, kp_frame, None, color=(0, 255, 0), flags=0)

            cv2.imshow(fname, frame)
            cv2.waitKey(1)

            last_frame = frame


except KeyboardInterrupt:

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)
