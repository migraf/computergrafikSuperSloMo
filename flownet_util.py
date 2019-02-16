import os
import numpy as np
import random
import tensorflow as tf
from tensorpack.dataflow.base import DataFlow
import tensorpack.dataflow as dataflow
from tensorpack.dataflow.serialize import LMDBSerializer
import glob
import cv2
import argparse


def chairs_train_test_split_lists(data_folder):
    # only look at the numbers in the chairs dataset and sort by them
    right_left_image_list = sorted(glob.glob(data_folder + "/" + "*.ppm"), key= lambda x : int(x[-14:-9]))
    left_images = right_left_image_list[0::2]
    right_images = right_left_image_list[1::2]
    flow_paths = sorted(glob.glob(data_folder + "/" + "*.flo"), key= lambda x : int(x[-14:-9]))

    print(left_images[:5], right_images[:5], flow_paths[:5])


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


class FlownetDataflow(DataFlow):
    def __init__(self, height, width):
        self.height = height
        self.width = width


chairs_train_test_split_lists("/graphics/scratch/students/graf/data/flownet/FlyingChairs_release/data")