import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import sys


sys.path.append('../Oxford_dataset')

from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

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

    # Enforce rank 2 constraint on F
    U,S,Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diagflat(S) @ Vh

    return F


def inlierRANSAC(x_a,x_b,iterations):
    S_in = []
    points = 8
    n = 0
    epsilon = .01

    for i in range(iterations):
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
            val = abs(x2j.T @ F @ x1j)
            if val < epsilon:
                S.append([p1,p2])

        if len(S) >= n:
            n = len(S)
            S_in = S
            Best_F = F

    return Best_F,S_in


def extractCameraPose(E):
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    U,D,Vt = np.linalg.svd(E)

    C = [U[:,2],-U[:,2],U[:,2],-U[:,2]]
    R = [U @ W @ Vt,U @ W @ Vt,U @ W.T @ Vt,U @ W.T @ Vt]

    Cset,Rset = [],[]
    for c,r in zip(C,R):
        det = np.linalg.det(r)
        if int(det) == -1:
            c *= -1
            r *= -1
        c.shape = (3,1)
        Cset.append(c)
        Rset.append(r)

    return Cset,Rset


# def plotTrajectory():



def LinearTriangulation(K, C1, R1, C2, R2, inliers):
    I=np.identity(3)
    IC1=np.concatenate((I,-C1),axis=1)
    P1=K @ R1 @ IC1

    IC2=np.concatenate((I,-C2),axis=1)
    P2=K @ R2 @ IC2

    p1=P1[:,0]
    p2=P1[:,0]
    p3=P1[:,0]

    p1_=P2[:,0]
    p2_=P2[:,0]
    p3_=P2[:,0]

    X=[]
    for pt1, pt2 in inliers:
        x=pt1[0]
        y=pt1[1]
        x_=pt2[0]
        y_=pt2[1]

        A=np.array([y*p3-p2,p1-x*p3,y_*p3_-p2_,p1_-x_*p3_])

        print(A.shape)

        U,D,Vt = np.linalg.svd(A)

        X_ = Vt[-1,:]
        X_.shape = (3,1)
        X.append(X_) 
    return X


def Cheirality(Cset,Rset,Xset):
    counts=[]
    for C,R,X in zip(Cset,Rset,Xset):
        count=0
        r3=R[:,-1]
        for x in X:
            val = r3@(x-C)
            if val>0:
                count+=1
        counts.append(count)

    ind=counts.index(max(counts))

    return Cset[ind], Rset[ind]

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


def drawMatches(matches):
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.2*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    cv2.imshow("CV2 Matches",img3)
    cv2.waitKey(0)


def analyzeVideo():
    onlyFirstFrame = False
    path = '../Oxford_dataset/stereo/centre/'
    frame_paths = []
    for img in os.listdir(path):
        frame_paths.append(path+img)

    frame_paths.sort()

    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('../Oxford_dataset/model')
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

    # removes poorly exposed frames
    for i in range(22):
        frame_paths.pop(0)

    surf = cv2.xfeatures2d.SURF_create()

    FLANN_INDEX_KDTREE = 1
    index_params = {'algorithm':FLANN_INDEX_KDTREE, 'trees':5}
    search_params = {'checks':50}   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    prev_img =  raw2Undistorted(frame_paths.pop(0),LUT)
    kp1, des1 = surf.detectAndCompute(prev_img,None)

    camera_origin = np.array([[0,0,0]],dtype='float64').T
    C0 = np.array([[0,0,0]]).T
    R0 = np.identity(3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    for path in frame_paths:
        cur_img = raw2Undistorted(path,LUT)

        # Find keypoints
        kp2, des2 = surf.detectAndCompute(cur_img,None)

        # Match keypoints to find correspondencies
        matches = flann.knnMatch(des1,des2,k=2)

        x_a,x_b = [],[]
        for k in range(len(matches)):
            x_a.append(kp1[matches[k][0].queryIdx].pt)
            x_b.append(kp2[matches[k][0].trainIdx].pt)

        # Find set of inliers using RANSAC
        F,inliers = inlierRANSAC(x_a,x_b,iterations=10)

        # Estimate Essential Matrix
        E = K.T @ F @ K

        # Extract 4 Possible Camera Poses
        Cset,Rset = extractCameraPose(E)

        # Compute the real world coordinates for each set of matched points 
        Xset = []

        for C,R in zip(Cset,Rset):
            X = LinearTriangulation(K, C0, R0, C, R, inliers)
            Xset.append(X)

        C,R = Cheirality(Cset,Rset,Xset)

        last_point = camera_origin.copy()
        camera_origin += C

        # ax.scatter3D(camera_origin[0], camera_origin[1], camera_origin[2])

        xs = [last_point[0,0],camera_origin[0,0]]

        ys = [last_point[1,0],camera_origin[1,0]]

        zs = [last_point[2,0],camera_origin[2,0]]

        ax.plot3D(xs,ys,zs,'gray')
        plt.draw()
        plt.pause(.001)

        match_img = showMatches(prev_img,cur_img,inliers)
        cv2.imshow("Matches",match_img)
        # cv2.waitKey(0)

        # change current values to previous values for next loop
        prev_img = cur_img
        kp1, des1 = kp2, des2
        C0 = C
        R0 = R

        # cv2.imshow('Frame',cur_img)

        if (cv2.waitKey(10) == ord('q')):
            exit()

        if onlyFirstFrame:
            exit()


if __name__ == "__main__":
    analyzeVideo()
    




