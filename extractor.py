import numpy as np
import cv2


class KeypointExtractor:
    def __init__(self) -> None:
        # self.H = H
        # self.W = W
        self.orb = cv2.ORB_create()

    def extract_keypoints(self, img):

        kp = self.orb.detect(img, None)
        kp, des = self.orb.compute(img, kp)

        return kp, des
