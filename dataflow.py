import os
import numpy as np
import random
import tensorpack
import tensorflow as tf
from tensorpack.dataflow.base import DataFlow
import glob

def create_train_files(train_folder, intermediate_frames):
    """
    Create list of filepaths with a number of sequential images from one video
    :param train_folder: the folder in which the images are contained
    :param intermediate_frames: number of intermediate frames
    :return: list of lists of file paths of training samples
    """
    train_list = []
    folder_list = [x[0] for x in os.walk(train_folder)]

    for folder in folder_list:
        #print(glob.glob(folder + "\\" + "*.jpg"))
        # Sort list of filenames by number
        image_list = sorted(glob.glob(folder + "\\" + "*.jpg"), key=lambda x: int(x.split("\\")[-1].split(".")[0]))
        for i in range(0, len(image_list), intermediate_frames):
            intermed_frames = image_list[i: i + intermediate_frames]
            if(len(intermed_frames) == 8):
                train_list.append(intermed_frames)

    return train_list


class IntermediateDataFlow(DataFlow):
    def __init__(self, image_list):
        self.image_list = image_list


    def __iter__(self):
        # TODO scale images down

        print(1)

    def __len__(self):
        return len(self.image_list)