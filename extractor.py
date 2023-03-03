import numpy as np
import cv2


class KeypointExtractor:
    def __init__(self) -> None:

        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_keypoints(self, img):

        kp = self.orb.detect(img, None)
        kp, des = self.orb.compute(img, kp)

        return kp, des

    def match_keypoints(self, des1, des2):

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    def compute_lines(self, matches):

        dist = [match.distance for match in matches]
        frame_idxs = [match.queryIdx for match in matches]
        last_frame_idxs = [match.trainIdx for match in matches]

        print(dist[0], frame_idxs[0], last_frame_idxs[0])
