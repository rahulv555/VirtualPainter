import cv2
from matplotlib.pyplot import draw
import numpy as np
import time
import os

from sqlalchemy import over

import HandTrackingModule as htm
# ADDED FUNCTION FOR CHECKING FINGER UP IN THIS MODULE


# for UI
UIfolderPath = "UI"
UIfiles = os.listdir(UIfolderPath)
overlayList = []
for imPath in UIfiles:
    image = cv2.imread(f'{UIfolderPath}/{imPath}')
    overlayList.append(image)

UIheader = overlayList[0]

# DEFAULT COLOR
drawColor = (0, 0, 255)
brushThickness = 15
eraserThickness = 40
xprev = -1
yprev = -1

# webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


# hand detector
Hdetector = htm.HandDetector(detectionConfd=0.85)


# imgcanvas
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


#######################
while True:
    success, img = cap.read()

    # flipping image
    img = cv2.flip(img, 1)

    # finding hand landmarks
    img = Hdetector.findHands(img)
    lmList = Hdetector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # index finger tip
        x1, y1 = lmList[8][1], lmList[8][2]
        # middle finger
        x2, y2 = lmList[12][1], lmList[12][2]

        # check whether fingers up
        fingersUp = Hdetector.fingersUp()

        # SELECTION MODE - INDEX AND MIDDLE UP
        if fingersUp[1] and fingersUp[2]:
            print("Selection")
            xprev, yprev = -1, -1
            cv2.rectangle(img, (x1, y1-15), (x2, y2+15),
                          drawColor, cv2.FILLED)

            if y1 < 125:
                if 0 < x1 < 180:
                    UIheader = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 180 < x1 < 360:
                    UIheader = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 390 < x1 < 555:
                    UIheader = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 580 < x1 < 750:
                    UIheader = overlayList[3]
                    drawColor = (0, 255, 255)
                elif 770 < x1 < 945:
                    UIheader = overlayList[4]
                elif 1140 < x1 < 1280:
                    UIheader = overlayList[5]
                    drawColor = (0, 0, 0)

        elif fingersUp[1] == 1:
            print("Drawing")
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xprev == -1 and yprev == -1:
                xprev, yprev = x1, y1

            # bigger eraser
            if drawColor == (0, 0, 0):
                cv2.line(imgCanvas, (xprev, yprev),
                         (x1, y1), drawColor, eraserThickness)
                cv2.line(img, (xprev, yprev),
                         (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(imgCanvas, (xprev, yprev),
                         (x1, y1), drawColor, brushThickness)
                cv2.line(img, (xprev, yprev),
                         (x1, y1), drawColor, brushThickness)

            xprev = x1
            yprev = y1
        else:
            xprev, yprev = -1, -1

    # TO OVERLAY IMG CANVAS ON IMG
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CONVERTING TO BINARY IMAGE and inversing
    # BLACK WHEREVER DRAWING IS, EVERYWHERE ELSE WHITE
    _, img_inv = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)
    # BACK TO BGR
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, imgCanvas)

    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    # displaying the header
    img[0:125, 0:1280] = UIheader

    cv2.imshow("Video Output", img)
    # cv2.imshow("Canvas Output", img_inv)
    cv2.waitKey(1)
