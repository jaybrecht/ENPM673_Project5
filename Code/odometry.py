import cv2
import numpy as np
import os
import sys

sys.path.append('../Oxford_dataset')

from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

path = '../Oxford_dataset/stereo/centre/'

frame_paths = []
for img in os.listdir(path):
    frame_paths.append(path+img)

frame_paths.sort()

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('../Oxford_dataset/model')

for path in frame_paths:
    img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img,cv2.COLOR_BayerGR2BGR)
    img = UndistortImage(img,LUT)
    cv2.imshow("Frame",img)
    cv2.waitKey(1)





