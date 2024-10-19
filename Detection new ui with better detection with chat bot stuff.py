import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from collections import deque
from spellchecker import SpellChecker

# Initialize video capture
cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
spell = SpellChecker()

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z", "SPACE", "BACKSPACE", "PAUSE", "ENTER"]

# Text conversion variables
text = ""
current_character = ""
last_prediction = ""
is_paused = False
is_temp_paused = False
confidence_threshold = 0.75
current_accuracy = 0
corrected_words = set()

# New variables for improved detection
prediction_times = {}
alphabet_detection_time = 0.79
command_detection_time = 0.5
command_signs = ["SPACE", "BACKSPACE", "PAUSE", "ENTER"]

# Cursor variables
cursor_visible = True
last_cursor_toggle = time.time()
cursor_blink_interval = 0.5  # seconds

# Define colors (in BGR format for OpenCV)
BACKGROUND_COLOR = (240, 240, 240)  # Light gray
ACCENT_COLOR = (255, 120, 0)  # Orange
TEXT_COLOR = (60, 60, 60)  # Dark gray
PANEL_COLOR = (255, 255, 255)  # White
BUTTON_COLOR = (0, 180, 60)  # Green

# Create a larger blank image for the main display
display_width, display_height = 1280, 720
main_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)

# Define button positions
buttons = {
    "SPACE": {"pos": (970, 550), "size": (140, 60)},
    "BACKSPACE": {"pos": (970, 620), "size": (140, 60)},
    "ENTER": {"pos": (1120, 550), "size": (140, 60)},
    "PAUSE": {"pos": (1120, 620), "size": (140, 60)}
}

# New variables for chat UI
chat_messages = []
max_messages = 5  # Maximum number of messages to display
message_height = 60  # Height of each message bubble
chat_area_height = max_messages * message_height
chat_area_width = 600

def autocorrect_word(word):
    if word.lower() in spell:
        return word
    correction = spell.correction(word)
    if correction and correction.lower() != word.lower():
        corrected_words.add(correction.lower())
    return correction if correction else word

def add_message_to_chat(message):
    global chat_messages
    chat_messages.append(message)
    if len(chat_messages) > max_messages:
        chat_messages.pop(0)

def draw_chat_bubble(img, message, pos, is_user=True):
    x, y = pos
    padding = 10
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(message, font, font_scale, 1)
    text_width, text_height = text_size

    bubble_width = min(text_width + padding * 2, chat_area_width - 20)
    bubble_height = text_height + padding * 2

    if is_user:
        bubble_color = (200, 230, 255)  # Light blue for user messages
        text_color = (0, 0, 0)  # Black text
        bubble_x = x + chat_area_width - bubble_width - 10
    else:
        bubble_color = (220, 220, 220)  # Light gray for system messages
        text_color = (0, 0, 0)  # Black text
        bubble_x = x + 10

    # Draw the bubble
    draw_rounded_rectangle(img, (bubble_x, y), (bubble_x + bubble_width, y + bubble_height), bubble_color, 10, 2, fill=True)

    # Draw the text
    cv2.putText(img, message, (bubble_x + padding, y + bubble_height - padding), font, font_scale, text_color, 1)

    return bubble_height + 5  # Return the height of the bubble plus some spacing

def execute_command(command, is_gesture=False):
    global text, is_paused, is_temp_paused
    if command == "SPACE":
        words = text.split()
        if words:
            words[-1] = autocorrect_word(words[-1])
        text = " ".join(words) + " "
    elif command == "BACKSPACE":
        if text:
            text = text[:-1]
    elif command == "ENTER":
        if text.strip():  # Only add non-empty messages
            add_message_to_chat(text.strip())
            text = ""
    elif command == "PAUSE":
        if is_gesture:
            is_temp_paused = not is_temp_paused
        else:
            is_paused = not is_paused

def clear_text():
    global text, corrected_words
    text = ""
    corrected_words.clear()
    print("Text cleared")

def draw_rounded_rectangle(img, pt1, pt2, color, radius, thickness, fill=False):
    x1, y1 = pt1
    x2, y2 = pt2

    if fill:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_button(img, text, pos, size, color):
    x, y = pos
    w, h = size
    draw_rounded_rectangle(img, (x, y), (x + w, y + h), color, 10, 2, fill=True)
    cv2.putText(img, text, (x + 10, y + h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 820 <= x <= 960 and 630 <= y <= 690:
            clear_text()
        for button, data in buttons.items():
            bx, by = data["pos"]
            bw, bh = data["size"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                execute_command(button)

cv2.namedWindow("Sign Language Chat")
cv2.setMouseCallback("Sign Language Chat", mouse_callback)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands and not is_paused:
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
            current_accuracy = prediction[index] * 100  # Convert to percentage

            if current_accuracy > confidence_threshold:
                current_prediction = labels[index]
                current_time = time.time()

                # Determine required detection time based on the sign type
                required_time = command_detection_time if current_prediction in command_signs else alphabet_detection_time

                # Update prediction times
                if current_prediction in prediction_times:
                    if current_time - prediction_times[current_prediction]['start_time'] >= required_time:
                        if current_prediction != last_prediction:
                            if current_prediction in command_signs:
                                execute_command(current_prediction, is_gesture=True)
                            else:
                                text += current_prediction
                                if not current_prediction.isalpha():
                                    words = text.split()
                                    if words:
                                        words[-1] = autocorrect_word(words[-1])
                                    text = " ".join(words)
                            last_prediction = current_prediction
                        prediction_times[current_prediction]['start_time'] = current_time
                else:
                    prediction_times[current_prediction] = {'start_time': current_time}

                # Reset timer for other predictions
                for pred in prediction_times.keys():
                    if pred != current_prediction:
                        prediction_times[pred]['start_time'] = current_time

                current_character = current_prediction

        except Exception as e:
            print(f"Error processing hand: {e}")

    # Clear the main display
    main_display[:] = BACKGROUND_COLOR

    # Add title
    cv2.putText(main_display, "Syne AI Chat", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, ACCENT_COLOR, 2)

    # Draw panels
    draw_rounded_rectangle(main_display, (10, 70), (630, 530), PANEL_COLOR, 20, 2, fill=True)
    draw_rounded_rectangle(main_display, (640, 70), (1270, 380), PANEL_COLOR, 20, 2, fill=True)
    draw_rounded_rectangle(main_display, (10, 540), (1270, 710), PANEL_COLOR, 20, 2, fill=True)

    # Place the main camera feed
    resized_output = cv2.resize(imgOutput, (600, 450))
    main_display[80:530, 20:620] = resized_output

    # Place the hand skeleton image
    if 'imgWhite' in locals():
        main_display[80:380, 650:950] = cv2.resize(imgWhite, (300, 300))

    # Draw chat area
    draw_rounded_rectangle(main_display, (640, 390), (1270, 530), PANEL_COLOR, 20, 2, fill=True)

    # Display chat messages
    y_offset = 400
    for message in chat_messages:
        bubble_height = draw_chat_bubble(main_display, message, (640, y_offset))
        y_offset += bubble_height

    # Display current input
    cv2.putText(main_display, f"Current input: {text}", (30, 590),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

    # Display current character
    cv2.putText(main_display, f"Current Character: {current_character}", (30, 630),
                cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

    # Display accuracy
    cv2.putText(main_display, f"Accuracy: {current_accuracy:.2f}%", (650, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 1, ACCENT_COLOR, 2)

    # Display the time left for the current prediction
    if current_character:
        required_time = command_detection_time if current_character in command_signs else alphabet_detection_time
        time_left = max(0, required_time - (time.time() - prediction_times.get(current_character, {'start_time': time.time()})['start_time']))
        cv2.putText(main_display, f"Time left: {time_left:.1f}s", (650, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, ACCENT_COLOR, 2)

    # Add clear button
    draw_rounded_rectangle(main_display, (820, 630), (960, 690), ACCENT_COLOR, 10, 2, fill=True)
    cv2.putText(main_display, "Clear", (865, 670),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add command buttons
    for button, data in buttons.items():
        if button == "PAUSE":
            button_text = "CONTINUE" if is_paused else "PAUSE"
        else:
            button_text = button
        draw_button(main_display, button_text, data["pos"], data["size"], BUTTON_COLOR)

    # Display pause status
    pause_status = "PAUSED" if is_paused else ("TEMP PAUSED" if is_temp_paused else "ACTIVE")
    cv2.putText(main_display, f"Status: {pause_status}", (650, 510),
                cv2.FONT_HERSHEY_SIMPLEX, 1, ACCENT_COLOR, 2)

    # Display the main window
    cv2.imshow("Sign Language Chat", main_display)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('c'):  # Press 'c' to clear text
        clear_text()
    elif key == 13:  # Press 'Enter' to send message
        execute_command("ENTER")

cap.release()
cv2.destroyAllWindows()