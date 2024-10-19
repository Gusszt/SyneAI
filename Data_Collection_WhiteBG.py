import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import string
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

characters = list(string.ascii_uppercase) + ['SPACE', 'BACKSPACE', 'PAUSE', 'ENTER']
dataset_size = 150

imgSize = 300
offset = 20
capture_delay = 0.075

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

def process_hand(img, hand):
    try:
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                      max(0, x - offset):min(img.shape[1], x + w + offset)]

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

        return imgWhite, (x, y, w, h)
    except Exception as e:
        logging.error(f"Error in process_hand: {str(e)}")
        return None, None

def safe_draw_hand(imgWhite, hand, x, y, w, h):
    try:
        scaled_lmList = []
        for lm in hand['lmList']:
            px, py, _ = lm
            px = int((px - x + offset) * imgWhite.shape[1] / (w + 2 * offset))
            py = int((py - y + offset) * imgWhite.shape[0] / (h + 2 * offset))
            scaled_lmList.append([px, py, _])

        scaled_hand = hand.copy()
        scaled_hand['lmList'] = scaled_lmList

        detector.drawHand(imgWhite, scaled_hand)
        return imgWhite
    except Exception as e:
        logging.error(f"Error in safe_draw_hand: {str(e)}")
        return imgWhite

def collect_data_for_character(char):
    char_dir = os.path.join(DATA_DIR, char)
    os.makedirs(char_dir, exist_ok=True)

    logging.info(f'Collecting data for character: {char}')

    print(f'Get ready to show the gesture for "{char}". Press "S" to start, "Q" to quit.')
    while True:
        success, img = cap.read()
        if not success:
            logging.warning("Failed to capture frame. Skipping...")
            continue

        hands, img = detector.findHands(img)

        cv2.putText(img, f'Ready to collect "{char}"? Press "S" to start!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Image', img)

        key = cv2.waitKey(1)
        if key == ord('s'):
            break
        elif key == ord('q'):
            return False

    counter = 0
    last_capture_time = time.time() - capture_delay
    while counter < dataset_size:
        success, img = cap.read()
        if not success:
            logging.warning("Failed to capture frame. Skipping...")
            continue

        hands, img = detector.findHands(img)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if hands:
            hand = hands[0]
            processed_result = process_hand(img, hand)
            if processed_result[0] is not None:
                imgWhite, (x, y, w, h) = processed_result
                imgWhite = safe_draw_hand(imgWhite, hand, x, y, w, h)

                current_time = time.time()
                if current_time - last_capture_time >= capture_delay:
                    counter += 1
                    img_name = f'{char_dir}/{char}_{time.time()}.jpg'
                    cv2.imwrite(img_name, imgWhite)
                    logging.info(f"Saved image {counter}/{dataset_size} for character {char}")
                    last_capture_time = current_time
                    capturing = True
                else:
                    capturing = False

                color = (0, 255, 0) if capturing else (0, 0, 255)
                cv2.putText(imgWhite, f"Captured: {counter}/{dataset_size}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.circle(imgWhite, (280, 20), 10, color, cv2.FILLED)

                cv2.imshow("ImageWhite", imgWhite)

        cv2.putText(img, f'Collecting "{char}": {counter}/{dataset_size}. Press "Q" to stop.', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Image', img)

        if cv2.waitKey(1) == ord('q'):
            break

    logging.info(f'Finished collecting data for character: {char}')
    return True

def main():
    try:
        for char in characters:
            if not collect_data_for_character(char):
                logging.info("Data collection stopped by user.")
                break
        logging.info("Data collection complete!")
    except Exception as e:
        logging.error(f"An error occurred during data collection: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
