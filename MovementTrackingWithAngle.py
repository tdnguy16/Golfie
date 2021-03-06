import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("Videos/tien3.mp4")

detector = pm.poseDetector()

while True:
    success, img = cap.read()
    #img = cv2.resize(img, (1280,690))
    #img = cv2.imread("Videos/Testpic.PNG")
    img = detector.findPose(img)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        #detector.findAngle(img,23,25,27)   #Leg Angle
        #detector.findanglehoz(img,13,11)   #Swing Plane
        #detector.spineangle(img)           #Spine Angle
        #detector.lag(img,11,12,23,24)      #Chest and hip rotation
        #detector.hinge(img, 13, 15)        #Hinge
        #detector.hipmove(img, 23)          #Hip move
        #detector.headmove(img, 7)          #Head move
        detector.movetracing(img, 11)          #Head move


    cv2.imshow("Image", img)
    cv2.waitKey(1)
