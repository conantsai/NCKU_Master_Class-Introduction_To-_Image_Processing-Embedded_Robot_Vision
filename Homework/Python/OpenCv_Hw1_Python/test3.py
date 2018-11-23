from scipy import signal
import cv2
import numpy as np
import matplotlib.pyplot as plt



img = cv2.imread('OriginalPerspective.png')
rows,cols,ch = img.shape
pts1 = np.float32([[200,350],[1200,400],[1300,900],[0,950]])
pts2 = np.float32([[20,20],[450,20],[450,450],[20,450]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

   

   
