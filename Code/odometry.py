import cv2
import numpy as np
import random
import os
import sys

sys.path.append('../Oxford_dataset')

from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

path = '../Oxford_dataset/stereo/centre/'


def raw2Undistorted(img_path,LUT):
    img =  cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img,cv2.COLOR_BayerGR2BGR)
    img = UndistortImage(img,LUT)
    return img


def EstimateFundamentalMatrix(x1,x2):
    A = np.zeros((8,9))

    for i in range(8):
        x,x_,y,y_ = x1[i][0],x2[i][0],x1[i][1],x2[i][1]
        
        A[i,0] = x*x_
        A[i,1] = x*y_
        A[i,2] = x
        A[i,3] = y*x_
        A[i,4] = y*y_
        A[i,5] = y
        A[i,6] = x_
        A[i,7] = y_
        A[i,8] = 1

    U,S,V = np.linalg.svd(A)
    f = V[:,-1]
    F = np.reshape(f,(3,3)).T

    return F


def inlierRANSAC(matches):
    S_in = set()
    M = len(matches)
    N = 8
    n = 0
    epsilon = 10
    for i in range(M):
        x1,x2 = [],[]
        k_list = []
        for j in range(8):
            k = random.randint(0,len(matches)-1)
            ind1 = matches[k][0].trainIdx
            ind2 = matches[k][1].trainIdx
            x1.append(kp1[ind1].pt)
            x2.append(kp2[ind2].pt)
            k_list.append(k)
        F = EstimateFundamentalMatrix(x1,x2)

        S = []
        for j in range(N):
            x1j = np.array([x1[j][0],x1[j][1],1])
            x2j = np.array([x2[j][0],x2[j][1],1])
            val = x2j.T.dot(F).dot(x1j)
            if abs(val) < epsilon:
                S.append(k_list[j])
        
        if n <= len(S):
            n = len(S)
            for k in k_list:
                S_in.add(k)
    
    inliers = []
    while S_in:
        inliers.append(S_in.pop())

    print(len(inliers))
    print(M)

    return inliers




onlyFirstFrame = True
frame_paths = []
for img in os.listdir(path):
    frame_paths.append(path+img)

frame_paths.sort()

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('../Oxford_dataset/model')

# removes poorly exposed frames
for i in range(22):
    frame_paths.pop(0)

surf = cv2.xfeatures2d.SURF_create()

FLANN_INDEX_KDTREE = 1
index_params = {'algorithm':FLANN_INDEX_KDTREE, 'trees':5}
search_params = {'checks':50}   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

prev_img =  raw2Undistorted(frame_paths.pop(0),LUT)

for path in frame_paths:
    cur_img = raw2Undistorted(path,LUT)

    # Find keypoints
    kp1, des1 = surf.detectAndCompute(prev_img,None)
    kp2, des2 = surf.detectAndCompute(cur_img,None)

    # Match keypoints to find correspondencies
    matches = flann.knnMatch(des1,des2,k=2)

    # Find set of inliers using RANSAC
    inliers = inlierRANSAC(matches)

    prev_img = cur_img

    cv2.imshow('Frame',cur_img)

    if (cv2.waitKey(10) == ord('q')):
        exit()

    if onlyFirstFrame:
        exit()





