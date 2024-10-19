import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z", "SPACE", "BACKSPACE", "ENTER", "PAUSE"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        try:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                      max(0, x - offset):min(img.shape[1], x + w + offset)]

            if imgCrop.size == 0:
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            accuracy = prediction[index] * 100  # Convert to percentage
            print(f"Prediction: {labels[index]} (Index: {index}, Accuracy: {accuracy:.2f}%)")

            if 0 <= index < len(labels):
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 300, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, f"{labels[index]} ({accuracy:.2f}%)", (x, y - 26),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        except Exception as e:
            print(f"Error processing hand: {e}")

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()