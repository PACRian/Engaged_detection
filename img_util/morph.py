import cv2
import numpy as np
from proc import ImgProc


class ImgErode(ImgProc):
    def proc(self):
        return 

class ImgClosing(ImgProc):
    def proc(self, ksize=3, iters=1, op=cv2.MORPH_CLOSE):
        return cv2.morphologyEx(self.img, op, \
            np.ones((ksize, ksize)), iterations=iters, borderType=cv2.BORDER_REFLECT)

class ImgDilate(ImgProc):
    def proc(self, ksize=3, iters=1):
        return cv2.dilate(self.img, np.ones((ksize, ksize)), iterations=iters)