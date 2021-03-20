from toad import ToadGenerator
import mediapipe as mp
import cv2
from config.config import CAMERA


def toad_game_mediapipe():
    x1 = x2 = y1 = y2 = 0
    cap = cv2.VideoCapture(CAMERA)
    td = ToadGenerator()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    while True:
        flag, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
                x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)
                y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                break

        xx1 = min(x1, x2)
        xx2 = max(x1, x2)
        yy1 = min(y1, y2)
        yy2 = max(y1, y2)

        image = td.step(image, (xx1, yy1, xx2, yy2))

        cv2.imshow('Toad collection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
