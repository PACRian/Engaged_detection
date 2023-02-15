import sys

import cv2
import numpy as np

from utiltis import many_plot, read_img

img = read_img(gray=True)
print(img.shape)
cv2.imshow('img', img)


sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
print(sobelx.shape)
print('here')
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)
cv2.imshow('Sobel X and Y gradients', np.hstack((sobelx, sobely)))
cv2.waitKey(0)

h = cv2.calcHist(img, [0], None, [256], [0, 256])
# Parameters: list of images as `numpy` arrays, channels, mask(optional) mask of the same size as the input image, histSize



