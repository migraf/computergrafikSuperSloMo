import tensorflow as tf
from tensorpack import ModelDesc
from tensorpack import logger
from dataflow import *
from tensorpack.tfutils.summary import add_moving_summary
import cv2


class FlowNetModel(ModelDesc):
    def __init__(self, name, height, width):
        self.name = name
        self.height = height
        self.width = width

    def input_names(self):
        return [tf.placeholder(tf.float32, (1,3, self.height, self.width), name="left_image"),
                tf.placeholder(tf.float32, (1,3,self.height, self.width), name="right_image"),
                tf.placeholder(tf.float32, (1,2, self.height, self.width), name="flow")]
