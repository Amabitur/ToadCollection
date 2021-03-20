import numpy as np
import cv2
import os
from PIL import Image
import time

SCORE = 0


class Toad:
    def __init__(self):
        self.x = np.random.randint(100, 500)
        self.y = -5
        self.image = cv2.imread('./toads/' + np.random.choice(os.listdir('./toads/')), cv2.IMREAD_UNCHANGED)
        scale = np.random.randint(70, 100)
        self.image = cv2.resize(self.image, (scale, scale ))
        self.size = self.image.shape[0]
        self.velocity = np.random.randint(3, 10)

    def update(self, hand):
        self.y += self.velocity
        return self.check_for_intersection(hand)

    def check_for_intersection(self, hand):
        global SCORE
        x1, y1, x2, y2 = hand
        if (self.y + self.size // 2 in range(y1, y2)) and (self.x + self.size // 2 in range(x1, x2)):
            SCORE += 1
        return (self.y > 600) or ((self.y + self.size // 2 in range(y1, y2)) and (self.x + self.size // 2 in range(x1, x2)))


class ToadGenerator:
    def __init__(self):
        self.toad_list = []
        self.time = time.time()
        global SCORE
        SCORE = 0

    def step(self, image, hand):
        if abs(self.time - time.time()) > 5:
            self.time = time.time()
            self.toad_list.append(Toad())

        x1, y1, x2, y2 = hand
        for toad in self.toad_list:
            if not toad.update((x1, y1, x2, y2)):
                image = self.draw_toad(image, toad)
            else:
                self.toad_list.remove(toad)
        image = self.add_score(image)
        return image

    def draw_toad(self, image, toad):
        pilimg = Image.fromarray(image)
        piltoad = Image.fromarray(toad.image)
        pilimg.paste(piltoad, (toad.x, toad.y), piltoad)
        image = np.array(pilimg)
        return image

    def add_score(self, image):
        global SCORE
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(image, 'Total score: ' + str(SCORE),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        return image