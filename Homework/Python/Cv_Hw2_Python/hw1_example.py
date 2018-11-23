# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
import matplotlib.pyplot as plt
from scipy import stats

class MainWindow(QMainWindow, Ui_MainWindow):
    boardSize = (11, 8)
    optlist = []

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
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn2_3.clicked.connect(self.on_btn2_3_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn3_3.clicked.connect(self.on_btn3_3_click)
        self.btn3_4.clicked.connect(self.on_btn3_4_click)
        self.btn4.clicked.connect(self.on_btn4_click)

    # button for problem 1.1
    def on_btn1_1_click(self):
        imageplant = cv2.imread('images/plant.jpg',0) 
        cv2.imshow("Original_Image", imageplant) 
        plt.hist(imageplant.ravel(),256,[0,256],color="red")
        plt.show()
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 1.2
    def on_btn1_2_click(self):
        imageplant = cv2.imread('images/plant.jpg',0) 
        equalizedplant = cv2.equalizeHist(imageplant)
        cv2.imshow("Equalized_Image", equalizedplant) 
        plt.hist(equalizedplant.ravel(),256,[0,256],color="red")
        plt.show()
        cv2.waitKey (0)
        cv2.destroyAllWindows()

    # button for problem 2.1
    def on_btn2_1_click(self):
        image = cv2.imread('images/q2_train.jpg')
        source = image.copy()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.medianBlur(image_gray, 5)
        rows = image_gray.shape[0]
        circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, rows / 64, param1=100, param2=15, minRadius=15, maxRadius=20)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(image, center, 1, (0, 97, 255), 3)
                # circle outline
                radius = i[2]
                cv2.circle(image, center, radius, (0, 255, 0), 2)

        cv2.imshow("Original_Image", source)
        cv2.imshow("Hough_Circle_Transform", image)
        cv2.waitKey(0)

    # button for problem 2.2
    def on_btn2_2_click(self):
        #def Hist_and_Backproj(val):
            #bins = val
            #histSize = max(bins, 2)
            #ranges = [0,180] # hue_range
            #hist = cv2.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
            #cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            #w = 400
            #h = 400
            #bin_w = int(np.round(w / histSize))
            #histImg = np.zeros((h, w, 3), dtype=np.uint8)
            #for i in range(bins):
            #    cv2.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(np.round( hist[i]*h/255.0 )) ), (0, 0, 255), cv2.FILLED)
            #cv2.imshow('Histogram', histImg)

        #src = cv2.imread('images/q2_train.jpg')
        #hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        #ch = (0, 0)
        #hue = np.empty(hsv.shape, hsv.dtype)
        #cv2.mixChannels([hsv], [hue], ch)
        #window_image = 'Source image'
        #cv2.namedWindow(window_image)
        #bins = 40
        #Hist_and_Backproj(bins)
        #cv2.imshow(window_image, src)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        img = cv2.imread('images/q2_train.jpg')
        # create a mask
        #mask = np.zeros(img.shape[:2], np.uint8)
        #mask[300:600, 300:700] = 255
        #masked_img = cv2.bitwise_and(img,img,mask = mask)

        cv8uc = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        hsv = cv2.cvtColor(cv8uc, cv2.COLOR_BGR2HSV)
        low_hsv = np.array([100, 43, 46])
        high_hsv = np.array([124, 255, 255])
        dst = cv2.inRange(hsv, low_hsv, high_hsv)

        # Calculate histogram with mask and without mask
        # Check third argument for mask
        hist_full = cv2.calcHist([img],[0],dst,[256],[0,256])
        #hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
        plt.plot(hist_full)
        plt.xlim([0,256])
        plt.show()

    # button for problem 2.3
    def on_btn2_3_click(self):
       imagetrain = cv2.imread('images/q2_train.jpg')
       trainhsv = cv2.cvtColor(imagetrain,cv2.COLOR_BGR2HSV)
       imagetarget = cv2.imread('images/q2_test.jpg')
       targethsv = cv2.cvtColor(imagetarget,cv2.COLOR_BGR2HSV)

       # calculating object histogram
       trainhist = cv2.calcHist([trainhsv ],[0, 1], None, [180, 256], [103, 121, 48, 190] )
       pdf = stats.norm.pdf(trainhist)
       # normalize histogram and apply backprojection
       cv2.normalize(trainhist,trainhist,0,255,cv2.NORM_MINMAX)
       targetdst = cv2.calcBackProject([targethsv],[0,1],trainhist,[103, 121, 48, 190],1)
       # Now convolute with circular disc
       disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
       cv2.filter2D(targetdst,-1,disc,targetdst)
       # threshold and binary AND
       ret,thresh = cv2.threshold(targetdst,80,255,0)

       cv2.imshow("Train ",imagetrain)
       cv2.imshow("Target ",imagetarget)
       cv2.imshow("Back_Projection ",thresh)
       cv2.waitKey (0)
       cv2.destroyAllWindows()

    # button for problem 3.1
    def on_btn3_1_click(self):
        showSize = QtWidgets.QDesktopWidget().screenGeometry(-1).height() * 0.8
        resH = QtWidgets.QDesktopWidget().screenGeometry(-1).height()
        resW = QtWidgets.QDesktopWidget().screenGeometry(-1).width()
        for i in range(1,16):
            image = cv2.imread('images/CameraCalibration/' + str(i) + '.bmp')
            found, corners = cv2.findChessboardCorners(image, self.boardSize)
            if found:
                cv2.drawChessboardCorners(image, self.boardSize, corners, found)
                cv2.namedWindow("Image " + str(i), cv2.WINDOW_GUI_NORMAL)
                height, width, channels = image.shape
                cv2.resizeWindow("Image " + str(i), (int(showSize / height * width), int(showSize)))
                cv2.moveWindow("Image " + str(i), int((resW - showSize / height * width) / 2), int((resH - showSize) / 2))
                cv2.imshow("Image " + str(i), image)
                while cv2.getWindowProperty("Image " + str(i), 0) >= 0:
                    keyCode = cv2.waitKey(50)
                    if keyCode >= 0:
                        break
                cv2.destroyAllWindows()

    # button for problem 3.2
    def on_btn3_2_click(self):
        objp = np.zeros((self.boardSize[0] * self.boardSize[1],3), np.float32)
        objp[:, :2] = np.mgrid[0:self.boardSize[0], 0:self.boardSize[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        for i in range(1,16):
            image = cv2.imread('images/CameraCalibration/' + str(i) + '.bmp')
            found, corners = cv2.findChessboardCorners(image, self.boardSize)
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs= cv2.calibrateCamera(objpoints,imgpoints, self.boardSize, None, None)
        print(cameraMatrix)

    # button for problem 3.3
    def on_btn3_3_click(self):
        # cboxImgNum to access to the ui object
        objp = np.zeros((self.boardSize[0] * self.boardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.boardSize[0], 0:self.boardSize[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        for i in range(1, 16):
            image = cv2.imread('images/CameraCalibration/' + str(i) + '.bmp')
            found, corners = cv2.findChessboardCorners(image, self.boardSize)
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.boardSize, None, None)
        dst, jacobian= cv2.Rodrigues(rvecs[self.cboxImgNum.currentIndex()])
        tmat = tvecs[self.cboxImgNum.currentIndex()]
        print(np.concatenate((dst, tmat), 1))

    # button for problem 3.4
    def on_btn3_4_click(self):
        objp = np.zeros((self.boardSize[0] * self.boardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.boardSize[0], 0:self.boardSize[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        for i in range(1, 16):
            image = cv2.imread('images/CameraCalibration/' + str(i) + '.bmp')
            found, corners = cv2.findChessboardCorners(image, self.boardSize)
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.boardSize, None, None)
        print(distCoeffs)

    def _cube_draw(self, img, imgpts):
        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[4].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[5].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[6].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[7].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[5].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[7].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[5].ravel()), tuple(imgpts[6].ravel()), (0, 0, 255), 10)
        img = cv2.line(img, tuple(imgpts[6].ravel()), tuple(imgpts[7].ravel()), (0, 0, 255), 10)
        return img
    
    # button for problem 4
    def on_btn4_click(self):
        showSize = QtWidgets.QDesktopWidget().screenGeometry(-1).height() * 0.8
        resH = QtWidgets.QDesktopWidget().screenGeometry(-1).height()
        resW = QtWidgets.QDesktopWidget().screenGeometry(-1).width()
        objp = np.zeros((self.boardSize[0] * self.boardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.boardSize[0], 0:self.boardSize[1]].T.reshape(-1, 2)
        objp = objp[::-1]
        objpoints = []
        imgpoints = []
        imgs = []

        for i in range(1, 6):
            image = cv2.imread('images/CameraCalibration/' + str(i) + '.bmp')
            found, corners = cv2.findChessboardCorners(image, self.boardSize)
            if found:
                objpoints.append(objp)
                imgs.append(image)
                imgpoints.append(corners)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.boardSize, None, None)
        axis = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                           [0, 0, -2],[0, 2, -2],[2, 2, -2],[2, 0, -2]]).reshape(-1,3)
        for i in range(0, 5):
            imgpts, jac = cv2.projectPoints(axis, rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
            self._cube_draw(imgs[i], imgpts)
            cv2.namedWindow("Image " + str(i + 1), cv2.WINDOW_GUI_NORMAL)
            height, width, channels = imgs[i].shape
            cv2.resizeWindow("Image " + str(i + 1), (int(showSize / height * width), int(showSize)))
            cv2.moveWindow("Image " + str(i + 1), int((resW - showSize / height * width) / 2), int((resH - showSize) / 2))
            cv2.imshow("Image " + str(i + 1), imgs[i])
            cv2.waitKey(500)
            cv2.destroyAllWindows()
       
    ### ### ###

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
