from tensorpack import Callback
import numpy as np
import cv2 as cv


class FlowVisualisationCallback(Callback):

    def __init__(self, names):
        """
        :param names: Names of the flow maps to visualize
        """
        self.names = names
        intermediate_flows_0 = ["flow_" + str(x) + "_0" for x in range(1,8)]
        intermediate_flows_1 = ["flow_" + str(x) + "_1" for x in range(1,8)]
        names = names + intermediate_flows_0
        names = names + intermediate_flows_1
        print("Printing flow names")
        print(names)


    def visualize_flow(self, flow):
        flow = np.squeeze(flow, axis=0)
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        print("Flow shape:")
        print(flow.shape)
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr

    def _setup_graph(self):
        self.tensors = self.get_tensors_maybe_in_tower(self.names)

    def _trigger_epoch(self):
            for i in range(len(self.tensors)):
                print(type(self.tensors[i]))
                flow = self.visualize_flow(self.tensors[i].eval(session=self.trainer.sess))
                self.trainer.monitors.put_image(self.names[i], flow)


        