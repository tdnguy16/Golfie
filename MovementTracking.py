import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('Videos/5.mp4')
#'Videos/4.mp4'
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if lmList:
        #print(lmList[14])
        cv2.circle(img, (lmList[12][1], lmList[12][2]), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (lmList[11][1], lmList[11][2]), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (lmList[24][1], lmList[24][2]), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (lmList[23][1], lmList[23][2]), 15, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(30,80), cv2.FONT_HERSHEY_DUPLEX, 3, (255,0,0), 3)
    cv2.imshow("Image", img)

    cv2.waitKey(1)
