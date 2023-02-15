import numpy as np
from matplotlib import patches as patches
from proc import ImgProc


def top_index(arr, top_k=10):
    w = arr.shape[-1]
    assert top_k>0 and type(top_k)==int

    top_idx = np.argsort(arr.ravel())[-top_k:]
    return top_idx//w, top_idx%w # (i, j) format 

def get_clips(arr, match_arr, top_k=10, var_thres=100, rect_return=False, crop_return=False, shirnk_bias=0):
    w, h = arr.shape[::-1]

    def _get_lim(s):
        return min(s), max(s)

    i, j = top_index(match_arr, top_k)
    i_var, j_var = np.var(i), np.var(j)
    if i_var > j_var and i_var>=var_thres:
        # horizontal clip slice 
        xy=(shirnk_bias, min(i))
        h=max(i)-min(i)
        w=w-2*shirnk_bias
    elif j_var > i_var and j_var>=var_thres:
        # vertical clip slice 
        # lim = _get_lim(j)
        xy=(min(j), shirnk_bias)
        w=max(j)-min(j)
        h=h-2*shirnk_bias
    else:
        raise RuntimeError("Not enough locator provided")
    
    if crop_return:
        return arr[xy[1]:xy[1]+h, xy[0]:xy[0]+w]
    return patches.Rectangle(xy, w, h) if rect_return else xy, w, h


class LocCropper(ImgProc):
    '''
    Use template to locate the specific position, 
    then using the locator to crop original images.
    ===============#===============
    Example:

    # Load image
    img = cv2.imread('test_r.jpg', 0)

    # Template matching
    proc = NccMatcher()  # Use default template setting
    proc.load_img(img)
    matched = proc()

    # Do cropping
    proc = LocCropper(matched)
    cropped = proc(img)
    '''
    def proc(self, img, top_k=15, var_thres=100, shirnk_bias=0, crop_return=True):
        match_obj = self.img
        while True:
            try:
                return get_clips(img, match_obj, top_k, var_thres, \
                    shirnk_bias=shirnk_bias, crop_return=crop_return)
            except RuntimeError:
                top_k = int(top_k*1.5)
