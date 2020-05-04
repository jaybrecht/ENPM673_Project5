
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np



def visualize(fig,ax,match_img,xs,ys,zs):
    ax.scatter(xs,ys,zs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig("graph.png")

    graph=cv2.imread("graph.png")
    # cv2.imshow("graph",graph)
    # cv2.waitKey(0)
    threepanel=np.concatenate((match_img,graph),axis=1)
    small=cv2.resize(threepanel,(1500,500))
    cv2.imshow("Matches and Graph",small)
    cv2.waitKey(0)

pic1=cv2.imread("pic1.jpg")
pic2=cv2.imread("pic2.jpg")
match_img=np.concatenate((pic1,pic2),axis=1)
# cv2.imshow("Match",match_img)
# cv2.waitKey(0)
# xs=[(1,2),(2,3),(3,4),(4,5),(5,6)]
# ys=[(1,2),(2,3),(3,4),(4,5),(5,6)]
# zs=[(1,2),(2,3),(3,4),(4,5),(5,6)]
# visualize(match_img,xs,ys,zs)

dpi=300
fig=plt.figure(figsize=(pic1.shape[1]/dpi,pic1.shape[0]/dpi),dpi=dpi)
ax=fig.add_subplot(111,projection='3d')

for i in range (15):
    # xs=[i+1,i+2]
    # ys=[i+1,i+2]
    # zs=[i+1,i+2]
    xs=i
    ys=i
    zs=i
    print(xs,ys,zs)  
    visualize(fig,ax,match_img,xs,ys,zs)


# plt.show()
