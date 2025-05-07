import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Hand detector & classifier
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Parameters
offset = 20
imgSize = 300
labels = ["Hello", "Help", "I Love You","Thank You","Yes","No"]

# Variables for display
sentence = ""
current_prediction = ""  # Store last prediction
last_added = None  # Track last added word

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    if hands:
        # Get bounding box of first hand
        x, y, w, h = hands[0]['bbox']
        x, y = max(0, x - offset), max(0, y - offset)
        x_end, y_end = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y:y_end, x:x_end]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Get prediction
        prediction, index = classifier.getPrediction(imgWhite)
        current_prediction = labels[index]

        # Prevent consecutive duplicates
        if labels[index] != last_added:
            sentence += labels[index] + " "  # Add new unique word
            last_added = labels[index]  # Update last added word

    # Display
    cv2.putText(img, f"Prediction: {current_prediction}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display sentence
    cv2.putText(img, f"Sentence: {sentence}", (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show reset button
    cv2.putText(img, "Press 'R' to Reset", (50, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('r'):  # Reset sentence
        sentence = ""
        current_prediction = ""
        last_added = None

