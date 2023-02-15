import cv2
import numpy as np
from proc import ImgProc


class OTUSBinary(ImgProc):
    # Check documentation here:
    # https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    def proc(self, range=(0, 255), ada_ran=False):
        if ada_ran:
            range = (self.img.min(), self.img.max())
        return cv2.threshold(self.img, *range, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

class AdaBinary(ImgProc): 
    # https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa42a3e6ef26247da787bf34030ed772c  
    def proc(self, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        bin_inv=False, bs=7, c=2):
        thres_type = cv2.THRESH_BINARY_INV if bin_inv else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(self.img, self.img.max(), method, \
            thres_type, bs, c)

def is_binary_img(img, b=255):
    return np.all((img==0) | (img==b))
