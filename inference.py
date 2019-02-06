import tensorflow as tf
from tensorpack import *
import numpy as np
import cv2
import argparse
from models import *


def split_video_into_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, image = cap.read()
    while success:
        frames.append(image)
        success, image = cap.read()
    return frames

def fuse_frames_into_video(frames, fps):
    video = cv2.VideoWriter("ouput_video", cv2.VideoWriter_fourcc("X", "2", "6", "4"), frames, fps, frames[0].shape[0])



if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video", help="path of the source video")
    parser.add_argument("--output_video", help="name of the output video")
    parser.add_argument("--model_dir", help="path to the saved model")


    args = parser.parse_args()

    frames = split_video_into_frames(args.source_video)

    output_frames = []

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with TowerContext("context", is_training=False):
            for i in range(len(frames) - 1):
                frame_0 = tf.placeholder(tf.float32, (1,128,128,3), "it_0")
                frame_1 = tf.placeholder(tf.float32, (1,128,128,3), "it_1")
                frame_2 = tf.placeholder(tf.float32, (1,128,128,3), "it_2")
                frame_3 = tf.placeholder(tf.float32, (1,128,128,3), "it_3")
                frame_4 = tf.placeholder(tf.float32, (1,128,128,3), "it_4")
                frame_5 = tf.placeholder(tf.float32, (1,128,128,3), "it_5")
                frame_6 = tf.placeholder(tf.float32, (1,128,128,3), "it_6")
                frame_7 = tf.placeholder(tf.float32, (1,128,128,3), "it_7")
                frame_8 = tf.placeholder(tf.float32, (1,128,128,3), "it_8")

                flow_net_result =




