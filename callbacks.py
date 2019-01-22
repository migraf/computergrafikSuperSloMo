from tensorpack import Callback
import numpy as np
import cv2 as cv


class VisualisationCallback(Callback):

    def __init__(self, names):
        """
        :param names: Names of the flow maps to visualize
        """
        self.names = names

    def visualize_flow(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr

    def _trigger_epoch(self):
        for name in self.names:


        