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
    A = np.zeros((len(x1),9))

    for i in range(len(x1)):
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

    U,S,Vh = np.linalg.svd(A)
    f = Vh[-1,:]

    F = np.reshape(f,(3,3))

    U,S,Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diagflat(S) @ Vh

    return F


def inlierRANSAC(x_a,x_b):
    S_in = []
    M = 10
    points = 8
    n = 0
    epsilon = .05

    for i in range(M):
        x1,x2 = [],[]

        for j in range(points):
            k = random.randint(0,len(x_a)-1)
            x1.append(x_a[k])
            x2.append(x_b[k])

        F = EstimateFundamentalMatrix(x1,x2)

        S = []
        for p1,p2 in zip(x_a,x_b):
            x1j = np.array([p1[0],p1[1],1])
            x2j = np.array([p2[0],p2[1],1])
            val = abs(x1j.T @ F @ x2j)
            if val < epsilon:
                S.append([p1,p2])

        if len(S) >= n:
            n = len(S)
            S_in = S
            Best_F = F

    print(len(S))
    return Best_F,S_in


def showMatches(img1,img2,inliers):
    w = img1.shape[1]
    match_img = np.concatenate((img1,img2),axis=1)
    for p1,p2 in inliers:
        x,y = p1
        x_,y_ = p2
        p1 = (int(x),int(y))
        p2 = (int(x_+w),int(y_))
        match_img = cv2.circle(match_img,p1,3,(0,0,255),-1)
        match_img = cv2.circle(match_img,p2,3,(0,0,255),-1)
        match_img = cv2.line(match_img,p1,p2,(0,255,0),1)

    return match_img


def analyzeVideo():
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
        Best_F,inliers = inlierRANSAC(matches)

        match_img = showMatches(prev_img,cur_img,matches,inliers)
        cv2.imshow("Matches",match_img)
        cv2.waitKey(0)

        prev_img = cur_img

        cv2.imshow('Frame',cur_img)

        if (cv2.waitKey(10) == ord('q')):
            exit()

        if onlyFirstFrame:
            exit()

if __name__ == "__main__":
    img1 = cv2.imread('pic1.jpg')
    img2 = cv2.imread('pic2.jpg')

    surf = cv2.xfeatures2d.SURF_create()

    FLANN_INDEX_KDTREE = 1
    index_params = {'algorithm':FLANN_INDEX_KDTREE, 'trees':5}
    search_params = {'checks':50}   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)

    # Match keypoints to find correspondencies
    matches = flann.knnMatch(des1,des2,k=2)

    x_a, x_b = [],[]
    for i in range(len(matches)):
        ind1 = matches[i][0].queryIdx
        ind2 = matches[i][1].trainIdx
        x,y = kp1[ind1].pt
        x_,y_ = kp2[ind2].pt
        x_a.append((x,y))
        x_b.append((x_,y_))

    Best_F,inliers = inlierRANSAC(x_a,x_b)

    match_img = showMatches(img1,img2,inliers)
    cv2.imshow("Our Matches",match_img)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.7*n.distance:
    #         matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = None,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    cv2.imshow("CV2 Matches",img3)
    cv2.waitKey(0)



