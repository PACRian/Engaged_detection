import cv2
import numpy as np
from proc import ImgProc

from config import CONFIGS


def _get_locator():
    try:
        temp = cv2.imread('../pics/locator.jpg', 0)
    except:
        temp = CONFIGS["DEFAULT_TEMPLATE"]
    return temp

class NccMatcher(ImgProc):
    def __init__(self, template=None) -> None:
        if isinstance(template, np.ndarray):
            self._temp = template
        elif isinstance(template, str):
            self._temp = cv2.imread(template, 0)
        elif template is None:
            self._temp = _get_locator()

    def proc(self, meth=cv2.TM_CCOEFF_NORMED):
        matched = cv2.matchTemplate(self.img, self._temp, meth)
        return matched