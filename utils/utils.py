from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np
import os
from PIL import Image
import time

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "models/faster_rcnn_R_50_FPN_3x.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)


def get_hand_prediction(image):
    outputs = predictor(image)
    boxes = outputs['instances'].pred_boxes
    return boxes.tensor.cpu().numpy().astype(int)


def draw_boxes(image, boxes):
    if len(boxes)> 0:
        xmin, ymin, xmax, ymax = boxes[0]
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    return image


def draw_toad(image, boxes):
    for box in boxes:
        toad = cv2.imread('./data/toads/' + np.random.choice(os.listdir('./data/toads/')), cv2.IMREAD_UNCHANGED)
        xmin, ymin, xmax, ymax = box
        toad = correct_toad_color(image[ymin:ymax, xmin:xmax], toad)
        perc0 = 0.75 * min(xmax - xmin, ymax - ymin) / max(toad.shape)
        w0 = int(toad.shape[1] * perc0)
        h0 = int(toad.shape[0] * perc0)
        toad = cv2.resize(toad, (w0, h0))
        pilimg = Image.fromarray(image)
        piltoad = Image.fromarray(toad)
        pilimg.paste(piltoad, ((xmin + xmax) // 2 - w0 // 2, (ymin + ymax) // 2 - h0 // 2), piltoad)
        image = np.array(pilimg)
    return image


def non_max_suppression_fast(boxes, overlapThresh=0.01):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def correct_toad_color(hand, toad):
    pre_hand = cv2.cvtColor(hand.copy(), cv2.COLOR_BGR2HSV)
    pre_toad = cv2.cvtColor(toad[:, :, 0:3].copy(), cv2.COLOR_BGR2HSV)
    satur_coef = pre_toad[:, :, 1].mean() / pre_hand[:, :, 1].mean()

    pre_toad[:, :, 1] = pre_toad[:, :, 1] / satur_coef
    new_toad = cv2.cvtColor(np.uint8(pre_toad), cv2.COLOR_HSV2BGR)

    new_toad_alpha = toad.copy()
    new_toad_alpha[:, :, 0] = new_toad[:, :, 0]
    new_toad_alpha[:, :, 1] = new_toad[:, :, 1]
    new_toad_alpha[:, :, 2] = new_toad[:, :, 2]

    return new_toad_alpha


def predict_and_draw(image):
    boxes = get_hand_prediction(image)
    print(boxes)
    print(len(boxes))
    boxes_final = non_max_suppression_fast(boxes, 0.6)
    image = draw_boxes(image, boxes_final)
    img = draw_toad(image, boxes_final)
    return img, len(boxes_final)
