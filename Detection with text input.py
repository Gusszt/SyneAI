import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from collections import deque

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z", "SPACE", "BACKSPACE", "ENTER", "PAUSE"]

# Text conversion variables
text = ""
last_prediction = ""
prediction_queue = deque(maxlen=5)
is_paused = False
confidence_threshold = 0.7

def execute_command(command):
    global text, is_paused
    if command == "SPACE":
        text += " "
    elif command == "BACKSPACE" and text:
        text = text[:-1]
    elif command == "ENTER":
        print(f"Submitted text: {text}")
        text = ""
    elif command == "PAUSE":
        is_paused = True

# Create a larger blank image for the main display
display_width, display_height = 1280, 720
main_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)

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

            if accuracy > confidence_threshold:
                prediction_queue.append(labels[index])
                current_prediction = max(set(prediction_queue), key=prediction_queue.count)

                if current_prediction != last_prediction:
                    if current_prediction in ["SPACE", "BACKSPACE", "ENTER", "PAUSE"]:
                        execute_command(current_prediction)
                    elif not is_paused:
                        text += current_prediction
                    is_paused = False
                    last_prediction = current_prediction

                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 300, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, f"{current_prediction} ({accuracy:.2f}%)", (x, y - 26),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

        except Exception as e:
            print(f"Error processing hand: {e}")

    # Clear the main display
    main_display.fill(0)

    # Resize and place the main camera feed
    resized_output = cv2.resize(imgOutput, (960, 540))
    main_display[90:630, 10:970] = resized_output

    # Place the cropped and white images
    if 'imgCrop' in locals() and 'imgWhite' in locals():
        main_display[90:390, 980:1280] = cv2.resize(imgCrop, (300, 300))
        main_display[400:700, 980:1280] = cv2.resize(imgWhite, (300, 300))

    # Display text box
    cv2.rectangle(main_display, (10, 10), (1270, 80), (255, 255, 255), cv2.FILLED)
    cv2.putText(main_display, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    # Display the main window
    cv2.imshow("Sign Language to Text Conversion", main_display)
    key = cv2.waitKey(1)

    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()