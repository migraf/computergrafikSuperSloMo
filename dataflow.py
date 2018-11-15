import os
import numpy as np
import random
import tensorflow as tf
from tensorpack.dataflow.base import DataFlow
import tensorpack.dataflow as dataflow
import glob
import cv2


class IntermediateDataFlow(DataFlow):
    def __init__(self, train_folder, num_intermediate_frames):
        self.file_list = self.create_train_files(train_folder, num_intermediate_frames)

    def create_train_files(self, train_folder, num_intermediate_frames):
        """
        Create list of filepaths with a number of sequential images from one video
        :param train_folder: the folder in which the images are contained
        :param num_intermediate_frames: number of intermediate frames
        :return: list of lists of file paths of training samples
        """
        train_list = []
        folder_list = [x[0] for x in os.walk(train_folder)]

        for folder in folder_list:
            # print(glob.glob(folder + "\\" + "*.jpg"))
            # Sort list of filenames by number
            image_list = sorted(glob.glob(folder + "\\" + "*.jpg"), key=lambda x: int(x.split("\\")[-1].split(".")[0]))
            for i in range(0, len(image_list), num_intermediate_frames):
                intermed_frames = image_list[i: i + num_intermediate_frames]
                if (len(intermed_frames) == 8):
                    train_list.append(intermed_frames)

        return train_list

    def __iter__(self):
        # TODO scale images down?
        # TODO actually load images use cv2 and return array of images
        for image_list in self.file_list:
            image_tensors = []
            for image_path in image_list:
                # convert to tensor
                image = cv2.imread(image_path)
                # resize to 360 x 360
                image_tensors.append(tf.image.resize_images(image, [360, 360]))
            yield image_tensors

    def __len__(self):
        return len(self.file_list)

    def get_data(self):
        data = []
        for image_list in self.file_list:
            image_tensors = []
            for image_path in image_list:
                # convert to tensor
                image = cv2.imread(image_path)
                # resize to 360 x 360
                image_tensors.append(tf.image.resize_images(image, [360, 360]))
            data.append(image_tensors)
        return data



df = IntermediateDataFlow("C:\\Uni\\computergrafik\\frames", 8)
for dp in df:
    print(dp)