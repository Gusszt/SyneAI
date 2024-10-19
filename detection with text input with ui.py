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
current_character = ""
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

def clear_text():
    global text
    text = ""

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
                    current_character = current_prediction

        except Exception as e:
            print(f"Error processing hand: {e}")

    # Clear the main display
    main_display.fill(255)  # White background

    # Add title
    cv2.putText(main_display, "Sign Language To Text Conversion", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    # Place the main camera feed
    resized_output = cv2.resize(imgOutput, (600, 450))
    main_display[70:520, 20:620] = resized_output

    # Place the hand skeleton image
    if 'imgWhite' in locals():
        main_display[70:370, 640:940] = cv2.resize(imgWhite, (300, 300))

    # Display current character
    cv2.putText(main_display, f"Character : {current_character}", (20, 560),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display sentence
    cv2.putText(main_display, f"Sentence : {text}", (20, 600),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add clear button
    cv2.rectangle(main_display, (1000, 600), (1100, 650), (200, 200, 200), -1)
    cv2.putText(main_display, "Clear", (1020, 635),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display the main window
    cv2.imshow("Sign Language to Text Conversion", main_display)
    key = cv2.waitKey(1)

    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('c'):  # Press 'c' to clear text
        clear_text()

cap.release()
cv2.destroyAllWindows()