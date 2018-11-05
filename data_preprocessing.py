import cv2
import numpy as np
import os
import random, string

# Set input directory and output directory



file_directory = "C:\\Uni\\computergrafik\\"
print(os.path.exists(file_directory + "\\frames"))
def convert_to_frames(file):
    print(file_directory + "frames\\" + file.title().split(".")[0])
    vid = cv2.VideoCapture(file)
    directory_name = "".join(random.choices(string.ascii_uppercase, k=5))
    os.mkdir(file_directory + "frames\\" + directory_name)
    success, image = vid.read()
    index = 0
    while success:
        # TODO maybe scale down image 720p, is too large
        cv2.imwrite(file_directory + "frames" + "\\" + directory_name + "\\" + "frame%d.jpg" % index, image)
        success, image = vid.read();
        index += 1

def convert_folder(video_directory):
    for file in os.listdir(video_directory):
        print(file.title())
        path = os.path.join(video_directory, file)
        convert_to_frames(path)

convert_folder("C:\\Uni\\computergrafik\\videos")