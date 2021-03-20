import cv2
from toad import ToadGenerator
from utils.utils import get_hand_prediction, draw_boxes, non_max_suppression_fast
from config.config import CAMERA

def toad_game_custom():
    x1 = x2 = y1 = y2 = 0

    cap = cv2.VideoCapture(CAMERA)
    td = ToadGenerator()
    i = 0
    boxes = []

    while True:
        flag, img = cap.read()
        if i % 25 == 0:
            boxes = get_hand_prediction(img)
            boxes = non_max_suppression_fast(boxes)
        if len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]

        img = td.step(img, (x1, y1, x2, y2))

        try:
            img = draw_boxes(img, boxes)
            cv2.imshow('Toad collection', img)
        except:
            cap.release()
            raise
        i += 1

        ch = cv2.waitKey(1)
        if ch == 27:
            break

    cap.release()
    cv2.destroyAllWindows()