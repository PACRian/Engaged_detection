import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from utiltis import read_img


def erode_edge(img):
    _, bin_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    k, iter_num = np.ones(3, dtype=np.uint8), 15
    dila_img = cv.dilate(bin_img, kernel=k, iterations=iter_num)
    bin_img = cv.erode(dila_img, kernel=k, iterations=iter_num)

    return cv.Sobel(bin_img, -1, 1, 1, ksize=3)

# https://www.jianguoyun.com/p/DSKfIakQxu2wCxiWqfUEIAA
if __name__ == '__main__':
    # Load image
    img = read_img(img_name='test1_r.jpg', gray=True)
    plt.imshow(img)
    plt.grid()
    plt.show()