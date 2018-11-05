import cv2
import numpy as np
import os

# Set input directory and output directory



file_directory = "C:\\Uni\\computergrafik\\"
def convert_to_frames(file):

    vid = cv2.VideoCapture(file)
    if not os.path.exists(file_directory + "frames\\" + file.title().split(".")[0]):
        os.mkdir(file_directory + "frames\\" + file.title().split(".")[0])

    success, image = vid.read()
    index = 0
    while success:
        cv2.imwrite(file_directory + "frames" + "\\" + file.title().split(".")[0] + "\\" + "frame%d.jpg" % index, image)
        success, image = vid.read();
        index += 1

def convert_folder(video_directory):
    for file in os.listdir(video_directory):
        print(file.title())
        path = os.path.join(video_directory, file)
        print(path)
        convert_to_frames(path)

convert_folder("C:\\Uni\\computergrafik\\videos")