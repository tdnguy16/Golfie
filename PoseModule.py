import cv2
import mediapipe as mp
import time
import math

class poseDetector():

    def __init__(self,
               mode=False,
               model_complexity=2,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               detectionCon=0.5,
               trackCon=0.5):


        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_landmarks,self.enable_segmentation,self.smooth_segmentation,self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw = True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

        if angle <0:
            angle += 360

        #print(angle)

        # Draw
        if draw:
            cv2.line(img,(x1, y1),(x2, y2),(0,255,0),3)
            cv2.line(img,(x2, y2),(x3, y3),(0,255,0),3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)),(x2-90,y2+20),
                        cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)


    def findanglehoz(self, img, p1, p2, draw = True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = x2 + 100, y2

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

        if angle <0:
            angle += 360

        #print(angle)

        # Draw
        if draw:
            cv2.line(img,(x1, y1),(x2, y2),(0,255,0),3)
            cv2.line(img,(x2, y2),(x3, y3),(0,255,0),3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)),(x2-90,y2+20),
                        cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

    def spineangle(self, img, draw = True):
        # Get the landmarks
        x1, y1 = (self.lmList[11][1]+self.lmList[12][1])/2 , (self.lmList[11][2]+self.lmList[12][2])/2
        x2, y2 = (self.lmList[23][1]+self.lmList[24][1])/2 , (self.lmList[23][2]+self.lmList[24][2])/2
        x3, y3 = x2 + 100, y2

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

        if angle <0:
            angle += 360

        #print(angle)

        # Draw
        if draw:
            cv2.line(img,(int(x1), int(y1)),(int(x2), int(y2)),(0,255,0),3)
            cv2.line(img,(int(x2), int(y2)),(int(x3), int(y3)),(0,255,0),3)
            cv2.circle(img, (int(x1), int(y1)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x1), int(y1)), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(x2), int(y2)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x2), int(y2)), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(x3), int(y3)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x3), int(y3)), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)),(int(x2)-90,int(y2)+20),
                        cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

    def lag(self, img, p1, p2, p3, p4, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        x4, y4 = self.lmList[p4][1:]
        x5, y5 = x1 + 100, y1
        x6, y6 = x3 + 100, y3

        # Calculate the Angle
        angle1 = math.degrees(math.atan2(y5 - y2, x5 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle2 = math.degrees(math.atan2(y6 - y4, x6 - x4) - math.atan2(y3 - y4, x3 - x4))

        if angle1 < 0:
            angle1 += 360

        if angle2 < 0:
            angle2 += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.line(img, (int(x2), int(y2)), (int(x5), int(y5)), (0, 255, 0), 3)
            cv2.circle(img, (int(x1), int(y1)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x1), int(y1)), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(x2), int(y2)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x2), int(y2)), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(x5), int(y5)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x5), int(y5)), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle1)), (int(x2) - 90, int(y2) + 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 3)
            cv2.line(img, (int(x4), int(y4)), (int(x6), int(y6)), (0, 255, 0), 3)
            cv2.circle(img, (int(x3), int(y3)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x3), int(y3)), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(x4), int(y4)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x4), int(y4)), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(x6), int(y6)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x6), int(y6)), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle2)), (int(x4) - 90, int(y4) + 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    def hinge(self, img, p1, p2, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]

        a = (y2-y1)/(x2-x1)
        b = y2 - ((y2-y1)/(x2-x1))*x2

        y3 = y2 + 200
        if a != 0:
            x3 = (y3 - b)/a
        else:
            x3 = x2 - 200


        # Draw
        if draw:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)), (0, 255, 0), 3)




def main():
    cap = cv2.VideoCapture('Videos/5.mp4')
    #'Videos/4.mp4'
    pTime = 0
    detector = poseDetector()
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


if __name__ == "__main__":
    main()