# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)

    # button for problem 1.1
    def on_btn1_1_click(self):
        imagedog = cv2.imread('dog.bmp') 
        size = imagedog.shape 
        dogheight = size[0]   
        dogwidth = size[1]
        print('Height = {0}\nWidth = {1} '.format(dogheight,dogwidth))
        #cv2.namedWindow("Load Image") 
        cv2.imshow("Load_Image", imagedog) 
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 1.2
    def on_btn1_2_click(self):
        imagecolor = cv2.imread('color.png') 
        cv2.namedWindow("ImageOriginal") 
        cv2.imshow("ImageOriginal", imagecolor) 
        r,b,g = cv2.split(imagecolor) # get r,b,g
        imagecolorconvert = cv2.merge([b,g,r]) # switch it to b,g,r
        #cv2.namedWindow("Image Color Conversion") 
        cv2.imshow("Image_Color_Conversion", imagecolorconvert) 
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 1.3
    def on_btn1_3_click(self):
        imagedog = cv2.imread('dog.bmp') 
        imagedogflip = cv2.flip(imagedog,1,dst=None) #Flip
        #cv2.namedWindow("Image Flip") 
        cv2.imshow("Image_Flip", imagedogflip) 
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 1.4
    def on_btn1_4_click(self):
        def do_nothing(x):
            pass

        cv2.namedWindow('Image_Blend')
        cv2.createTrackbar('Blend','Image_Blend',0,100,do_nothing)
        while(1):          
            imagedog = cv2.imread('dog.bmp') 
            imagedogflip = cv2.flip(imagedog,1,dst=None)
            k=cv2.waitKey(1)
            if k == ord('q'): #按q键退出
                break
            blendvalue = cv2.getTrackbarPos('Blend','Image_Blend')
            imagedogblending = cv2.addWeighted(imagedog, blendvalue/100, imagedogflip, 1-blendvalue/100, 0) #alpha Blend
            cv2.imshow("Image_Blend", imagedogblending) 
        cv2.destroyAllWindows()
 
    # button for problem 2
    def on_btn2_1_click(self):
        #textboxValue = self.textbox.text()

        #image = np.array(Image.open('M8.jpg')).astype(np.uint8)
        gray_img = cv2.imread('M8.jpg',0)
        gray_img = cv2.GaussianBlur(gray_img,(3,3),0)
        #gray_img = np.array(gray_img).astype(np.uint8)
        cv2.imshow("Gaussian_Blur", gray_img)

        # Sobel Operator
        h, w = gray_img.shape
        # define filters
        horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
        vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

        # define images with 0s
        newhorizontalImage = np.zeros((h, w))
        newverticalImage = np.zeros((h, w))
        newgradientImage = np.zeros((h, w))

        #Returns the maximum value from gradient_y/gradient_x
        def maximum(gradient):
           max = gradient[0][0]
           for i in range(len(gradient)):
               for j in range(len(gradient[0])):
                   if (max < gradient[i][j]):
                       max = gradient[i][j]
           return max

        # offset by 1
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) + \
                         (horizontal[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal[2, 2] * gray_img[i + 1, j + 1])

                newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)/255

                verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])

                newverticalImage[i - 1, j - 1] = abs(verticalGrad)/255

                # Edge Magnitude
                mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
                newgradientImage[i - 1, j - 1] = mag/255
                
        cv2.imshow("HorizontalImage",newhorizontalImage)
        cv2.imshow("VerticalImage",newverticalImage)
        cv2.imshow("GradientImage",newgradientImage)

        theta = math.atan2(verticalGrad,horizontalGrad)

        norm_image = cv2.normalize(newgradientImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #textboxValue = float(textboxValue)
        #test = newverticalImage*abs(math.cos(theta - textboxValue))
        #ret,testimage = cv2.threshold(test ,textboxValue,255,cv2.THRESH_BINARY)
        #cv2.imshow("testimage",testimage)

        def do_nothing(x):
            pass

        cv2.namedWindow("Magnitude")
        cv2.createTrackbar('magnitude','Magnitude',40,255,do_nothing)
        cv2.namedWindow("Direction")
        cv2.createTrackbar('Angle','Direction',0,360,do_nothing)

        while True:        
            k = cv2.waitKey(1)
            magnitudevalue = cv2.getTrackbarPos('magnitude','Magnitude')
            magnituderet,magnitudeimage = cv2.threshold(norm_image ,magnitudevalue,255,cv2.THRESH_BINARY)
            #norm_image1 = cv2.normalize(magnitudeimage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imshow("Magnitude",magnitudeimage)

            anglevalue = cv2.getTrackbarPos('Angle','Direction')
            test = norm_image*abs(math.cos(theta - anglevalue))
            ret,testimage = cv2.threshold(test ,anglevalue,255,cv2.THRESH_BINARY)
            cv2.imshow("Direction",testimage)

       
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 3
    def on_btn3_1_click(self):
        imagepyramids = cv2.imread('pyramids_Gray.jpg') 
        gaussionlevel1 = imagepyramids.copy()
        gaussionlevel1 = cv2.GaussianBlur(gaussionlevel1,(5,5),0)
        gaussionlevel1 = cv2.pyrDown(gaussionlevel1)
        
        gaussionlevel2 = gaussionlevel1.copy()
        gaussionlevel2 = cv2.GaussianBlur(gaussionlevel2,(5,5),0)
        gaussionlevel2 = cv2.pyrDown(gaussionlevel2)

        expendgaussionlevel1 = cv2.pyrUp(gaussionlevel1)
        expendgaussionlevel1 = cv2.GaussianBlur(expendgaussionlevel1,(5,5),0)
        laplacianlevel0 = cv2.subtract(imagepyramids, expendgaussionlevel1)
        row_num = laplacianlevel0.shape[0]
        column_num = laplacianlevel0.shape[1]

        expendgaussionlevel2 = cv2.pyrUp(gaussionlevel2)
        expendgaussionlevel2 = cv2.GaussianBlur(expendgaussionlevel2,(5,5),0)
        laplacianlevel1 = cv2.subtract(gaussionlevel1, expendgaussionlevel2)

        #inverselevel0 = cv2.add(expendgaussionlevel1.laplacianlevel0)
        #inverselevel1 = cv2.add(expendgaussionlevel2.laplacianlevel1)
        #inverselevel1 = cv2.addWeighted(expendgaussionlevel2, 0.5, laplacianlevel1, 0.5, 0)
        #inverselevel0 = cv2.addWeighted(expendgaussionlevel1, 0.5, laplacianlevel0, 0.5, 0)
        inverselevel1 = expendgaussionlevel2 + laplacianlevel1
        inverselevel0 = expendgaussionlevel1 + laplacianlevel0

        #cv2.namedWindow("Origin") 
        cv2.imshow("Origin", imagepyramids) 
        #cv2.namedWindow("Gaussian_Level_1") 
        cv2.imshow("Gaussian_Level_1", gaussionlevel1) 
        #cv2.namedWindow("Laplacian_Level_0") 
        cv2.imshow("Laplacian_Level_0", laplacianlevel0) 
        #cv2.namedWindow("Gaussian_Level_2") 
        cv2.imshow("Gaussian_Level_2", gaussionlevel2)
        #cv2.namedWindow("Laplacian_Level_1") 
        cv2.imshow("Laplacian_Level_1", laplacianlevel1) 
        #cv2.namedWindow("Expendgaussion_Level_1") 
        cv2.imshow("Expendgaussion_Level_1", expendgaussionlevel1) 
        #cv2.namedWindow("Expendgaussion_Level_2") 
        cv2.imshow("Expendgaussion_Level_2", expendgaussionlevel2) 
        #cv2.namedWindow("Inverse_Level_1") 
        cv2.imshow("Inverse_Level_1", inverselevel1)
        #cv2.namedWindow("Inverse_Level_0") 
        cv2.imshow("Inverse_Level_0", inverselevel0)
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 4.1
    def on_btn4_1_click(self):
        imageQR = cv2.imread('QR.png',0) 
        cv2.namedWindow("ImageOriginal") 
        cv2.imshow("ImageOriginal", imageQR) 

        ret,qrgthreshold = cv2.threshold(imageQR ,80,255,cv2.THRESH_BINARY)
        cv2.namedWindow("ImageGlobalThreshold") 
        cv2.imshow("ImageGlobalThreshold", qrgthreshold) 
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 4.2
    def on_btn4_2_click(self):
        imageQR = cv2.imread('QR.png',0)
        imageQR = cv2.medianBlur(imageQR,5)

        qrlthreshold = cv2.adaptiveThreshold(imageQR,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19,-1)
        #th3 = cv2.adaptiveThreshold(imageQR,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,-1)
        #cv2.namedWindow("Image_Original") 
        cv2.imshow("Image_Original", imageQR) 

        #cv2.namedWindow("Image_Global_Threshold") 
        cv2.imshow("Image_Global_Threshold", qrlthreshold) 
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 5.1
    def on_btn5_1_click(self):
        # edtAngle, edtScale. edtTx, edtTy to access to the ui object
        angle = self.edtAngle.text()
        angle = int (angle)
        scale = self.edtScale.text()
        scale = float(scale)
        tx = self.edtTx.text()
        tx = int(tx)
        ty = self.edtTy.text()
        ty = int(ty)

        imgtransform = cv2.imread('OriginalTransform.png')
        size = imgtransform.shape
        imgtransformheight = size[0]   
        imgtransformwidth = size[1]

        transform = np.float32([[1,0, tx],[0,1,ty]])
        imagetransform = cv2.warpAffine(imgtransform,transform,(imgtransformwidth,imgtransformheight))

        rotation = cv2.getRotationMatrix2D((imgtransformwidth/2,imgtransformheight/2),angle,scale)
        imagerotation = cv2.warpAffine(imagetransform,rotation,(imgtransformwidth,imgtransformheight))

        #cv2.namedWindow("Transforms: Origin") 
        cv2.imshow("Transforms: Origin", imgtransform) 
        #cv2.namedWindow("Transforms: Rotation, Scaling, Translation") 
        cv2.imshow("Transforms: Rotation, Scaling, Translation", imagerotation) 
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 5.2
    def on_btn5_2_click(self):
        def perspective(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                position = [x,y]
                listposition.append(position)
                print(listposition)
                while(len(listposition)==4):
                    pts1 = np.float32(listposition)
                    pts2 = np.float32([[20,20],[450,20],[450,450],[20,450]])
                    M = cv2.getPerspectiveTransform(pts1,pts2)
                    dst = cv2.warpPerspective(imageperspective,M,(430,430))
                    listposition.clear()
                    cv2.namedWindow("Image_Perspective") 
                    cv2.setMouseCallback('Image_Perspective',perspective)
                    cv2.imshow("Image_Perspective", dst) 
                    cv2.waitKey (0)
                    cv2.destroyAllWindows()
                    return 
        while(1):
            imageperspective = cv2.imread('OriginalPerspective.png')
            listposition = []
            cv2.namedWindow("Image_Origin_Perspective") 
            cv2.setMouseCallback('Image_Origin_Perspective',perspective)
            cv2.imshow("Image_Origin_Perspective", imageperspective) 
            cv2.waitKey (0)
            cv2.destroyAllWindows()
       

    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
