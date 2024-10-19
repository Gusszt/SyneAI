import cv2
import numpy as np
import math
import time
import ollama
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from spellchecker import SpellChecker
import threading

# Error handling decorator
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
    return wrapper

# Initialize video capture
cap = cv2.VideoCapture(0)

try:
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    spell = SpellChecker()
    desired_model = 'llama3.2:latest'

    offset = 20
    imgSize = 300
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
              "V", "W", "X", "Y", "Z", "SPACE", "BACKSPACE", "PAUSE", "ENTER"]

    # Text conversion variables
    text = ""
    current_character = ""
    last_prediction = ""
    is_paused = False
    is_temp_paused = False
    confidence_threshold = 0.75
    current_accuracy = 0
    corrected_words = set()

    # Detection time variables
    prediction_times = {}
    alphabet_detection_time = 0.79
    command_detection_time = 0.5
    command_signs = ["SPACE", "BACKSPACE", "PAUSE", "ENTER"]

    # Define colors (in BGR format for OpenCV)
    BACKGROUND_COLOR = (240, 240, 240)
    ACCENT_COLOR = (255, 120, 0)
    TEXT_COLOR = (60, 60, 60)
    PANEL_COLOR = (255, 255, 255)
    BUTTON_COLOR = (0, 180, 60)

    # Create a blank image for the main display (downsized)
    main_display_width, main_display_height = 800, 600
    main_display = np.zeros((main_display_height, main_display_width, 3), dtype=np.uint8)

    # Create a blank image for the chat display
    chat_display_width, chat_display_height = 400, 600
    chat_display = np.zeros((chat_display_height, chat_display_width, 3), dtype=np.uint8)

    # Define button positions (downsized)
    buttons = {
        "SPACE": {"pos": (560, 450), "size": (110, 50)},
        "BACKSPACE": {"pos": (560, 510), "size": (110, 50)},
        "ENTER": {"pos": (680, 450), "size": (110, 50)},
        "PAUSE": {"pos": (680, 510), "size": (110, 50)}
    }

    # Chat variables
    chat_messages = []
    max_messages = 10
    scroll_position = 0
    total_height = 0

    @error_handler
    def get_ai_response(user_message):
        response = ollama.chat(model=desired_model, messages=[
            {'role': 'user', 'content': user_message},
        ])
        ai_response = response['message']['content']
        add_message_to_chat(f"AI: {ai_response}")

    @error_handler
    def add_message_to_chat(message):
        global chat_messages
        chat_messages.append(message)
        if len(chat_messages) > max_messages:
            chat_messages.pop(0)
        update_chat_display()

    @error_handler
    def wrap_text(text, max_width, font, font_scale, thickness):
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_size, _ = cv2.getTextSize(word + " ", font, font_scale, thickness)
            word_width = word_size[0]

            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    @error_handler
    def update_chat_display():
        global chat_display, scroll_position, total_height
        chat_display[:] = BACKGROUND_COLOR
        y_offset = 20 - scroll_position
        total_height = 0

        for message in chat_messages:
            bubble_height = draw_chat_bubble(chat_display, message, (10, y_offset), is_user="You:" in message)
            y_offset += bubble_height + 10
            total_height += bubble_height + 10

        # Draw scrollbar
        scrollbar_height = min(chat_display_height, (chat_display_height / total_height) * chat_display_height)
        scrollbar_position = (scroll_position / total_height) * chat_display_height
        cv2.rectangle(chat_display, (chat_display_width - 10, int(scrollbar_position)),
                      (chat_display_width, int(scrollbar_position + scrollbar_height)), (150, 150, 150), -1)

    @error_handler
    def autocorrect_word(word):
        if word.lower() in spell:
            return word
        correction = spell.correction(word)
        if correction and correction.lower() != word.lower():
            corrected_words.add(correction.lower())
        return correction if correction else word

    @error_handler
    def draw_chat_bubble(img, message, pos, is_user=True):
        x, y = pos
        padding = 10
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        max_width = 360  # Adjusted for padding

        lines = wrap_text(message, max_width, font, font_scale, thickness)

        line_height = int(cv2.getTextSize(" ", font, font_scale, thickness)[0][1] * 1.5)
        bubble_height = len(lines) * line_height + padding * 2

        if is_user:
            bubble_color = (200, 230, 255)
            text_color = (0, 0, 0)
            bubble_x = x + 380 - max_width - padding * 2
        else:
            bubble_color = (220, 220, 220)
            text_color = (0, 0, 0)
            bubble_x = x

        cv2.rectangle(img, (bubble_x, y), (bubble_x + max_width + padding * 2, y + bubble_height), bubble_color, -1)

        for i, line in enumerate(lines):
            cv2.putText(img, line, (bubble_x + padding, y + (i + 1) * line_height), font, font_scale, text_color,
                        thickness)

        return bubble_height

    @error_handler
    def send_message():
        global text
        if text.strip():
            add_message_to_chat(f"You: {text.strip()}")
            threading.Thread(target=get_ai_response, args=(text.strip(),)).start()
            text = ""

    @error_handler
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
            send_message()
        elif command == "PAUSE":
            if is_gesture:
                is_temp_paused = not is_temp_paused
            else:
                is_paused = not is_paused

    @error_handler
    def clear_text():
        global text, corrected_words
        text = ""
        corrected_words.clear()
        print("Text cleared")

    @error_handler
    def draw_button(img, text, pos, size, color):
        x, y = pos
        w, h = size
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.putText(img, text, (x + 5, y + h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    @error_handler
    def mouse_callback(event, x, y, flags, param):
        global scroll_position, text
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Scroll up
                scroll_position = max(0, scroll_position - 20)
            else:  # Scroll down
                scroll_position = min(max(0, total_height - chat_display_height), scroll_position + 20)
            update_chat_display()
        elif event == cv2.EVENT_LBUTTONDOWN:
            if 440 <= x <= 550 and 520 <= y <= 570:
                clear_text()
            for button, data in buttons.items():
                bx, by = data["pos"]
                bw, bh = data["size"]
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    if button == "ENTER":
                        send_message()
                    else:
                        execute_command(button)

    cv2.namedWindow("Sign Language Recognition")
    cv2.setMouseCallback("Sign Language Recognition", mouse_callback)
    cv2.namedWindow("AI Chat")
    cv2.setMouseCallback("AI Chat", mouse_callback)

    print("Starting main loop...")
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
            current_accuracy = prediction[index] * 100

            if current_accuracy > confidence_threshold:
                current_prediction = labels[index]
                current_time = time.time()

                required_time = command_detection_time if current_prediction in command_signs else alphabet_detection_time

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

                for pred in prediction_times.keys():
                    if pred != current_prediction:
                        prediction_times[pred]['start_time'] = current_time

                current_character = current_prediction

        main_display[:] = BACKGROUND_COLOR

        cv2.putText(main_display, "Syne AI", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT_COLOR, 2)

        resized_output = cv2.resize(imgOutput, (520, 390))
        main_display[50:440, 20:540] = resized_output

        if 'imgWhite' in locals():
            main_display[50:290, 550:790] = cv2.resize(imgWhite, (240, 240))

        cv2.putText(main_display, f"Current input: {text}", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        cv2.putText(main_display, f"Current Character: {current_character}", (550, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        cv2.putText(main_display, f"Accuracy: {current_accuracy:.2f}%", (550, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_COLOR, 2)

        if current_character:
            required_time = command_detection_time if current_character in command_signs else alphabet_detection_time
            time_left = max(0, required_time - (
                        time.time() - prediction_times.get(current_character, {'start_time': time.time()})[
                    'start_time']))
            cv2.putText(main_display, f"Time left: {time_left:.1f}s", (550, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_COLOR, 2)

        cv2.rectangle(main_display, (440, 520), (550, 570), ACCENT_COLOR, -1)
        cv2.putText(main_display, "Clear", (475, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for button, data in buttons.items():
            if button == "PAUSE":
                button_text = "CONTINUE" if is_paused else "PAUSE"
            else:
                button_text = button
            draw_button(main_display, button_text, data["pos"], data["size"], BUTTON_COLOR)

        pause_status = "PAUSED" if is_paused else ("TEMP PAUSED" if is_temp_paused else "ACTIVE")
        cv2.putText(main_display, f"Status: {pause_status}", (550, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_COLOR, 2)

        cv2.imshow("Sign Language Recognition", main_display)
        update_chat_display()
        cv2.imshow("AI Chat", chat_display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Press 'Esc' to exit
            break
        elif key == ord('c'):  # Press 'c' to clear text
            clear_text()
        elif key == 13:  # Press 'Enter' to send message
            send_message()

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    print("Closing application...")
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully")

print("Script execution completed")