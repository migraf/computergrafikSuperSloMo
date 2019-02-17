import os
import numpy as np
import random
import tensorflow as tf
from tensorpack.dataflow.base import DataFlow
import tensorpack.dataflow as dataflow
from tensorpack.dataflow.serialize import LMDBSerializer
from tensorpack.train import TrainConfig
import glob
import cv2
import argparse
from sklearn.model_selection import train_test_split
from callbacks import *


def chairs_train_test_split_lists(data_folder):
    """
    Creates ordered list of left_img, right_img, flow for splitting the data into training and testing set
    saves it in a 2d numpy array
    :param data_folder:
    :return:
    """
    # only look at the numbers in the chairs dataset and sort by them
    right_left_image_list = sorted(glob.glob(data_folder + "/" + "*.ppm"), key= lambda x : int(x[-14:-9]) + int(x[-5]))
    left_images = right_left_image_list[0::2]
    right_images = right_left_image_list[1::2]
    flow_paths = sorted(glob.glob(data_folder + "/" + "*.flo"), key= lambda x : int(x[-14:-9]))
    train_left, test_left, train_right, test_right, train_flow, test_flow = train_test_split(left_images, right_images, flow_paths, test_size=0.2)
    np.save("train_paths", [train_left, train_right, train_flow])
    np.save("test_paths", [test_left, test_right, test_flow])



def read_flow(flow_path):
    """https: // stackoverflow.com / questions / 28013200 / reading - middlebury - flow - files -
    with-python - bytes - array - numpy"""
    with open(flow_path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
            return data2D

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
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr


class FlownetDataflow(DataFlow):
    def __init__(self, files_path):
        self.path = files_path
        self.file_names = np.load(files_path)
        input_shape = cv2.imread(self.file_names[0][0]).shape
        print("input shape")
        print(input_shape)
        self.height = input_shape[0]
        self.width = input_shape[1]


    def __iter__(self):
        for i in range(len(self.file_names[0])):
            left_image = cv2.imread(self.file_names[0][i])
            right_image = cv2.imread(self.file_names[1][i])
            flow = read_flow(self.file_names[2][i])

            yield (left_image, right_image, flow)
    def __len__(self):
        return self.file_names.shape[1]




df = FlownetDataflow("/graphics/scratch/students/graf/computergrafikSuperSloMo/train_paths.npy")
print("DataFlow created - size:")
print(len(df))
