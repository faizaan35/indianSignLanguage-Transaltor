import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap=cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


offset = 20
imgSize = 300

folder= 'Data/Help'
counter = 0

while True:
    success, img = cap.read()
    hands , img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]


        imgCropShape = imgCrop.shape



        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wcal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wcal,imgSize))
            imgResizedShape = imgResize.shape
            wGap = math.ceil((imgSize-wcal)/2)
            imgWhite[:,wGap:wcal+wGap] = imgResize

        else :
            k = imgSize / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hcal))
            imgResizedShape = imgResize.shape
            hGap = math.ceil((imgSize - hcal) / 2)
            imgWhite[hGap:hcal + hGap , : ] = imgResize


        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key=cv2.waitKey(1)
    if key == ord('q'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)





