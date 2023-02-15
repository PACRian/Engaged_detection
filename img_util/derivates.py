import cv2
from proc import ImgProc


def get_derivates(arr, pre_smooth=True, method='sobel', smooth_args=[], direction='both', **kwargs):

    smooth_args = smooth_args if smooth_args else [(3, 3), 0] # ksize, sigmaX
    if pre_smooth:
        arr = cv2.GaussianBlur(arr, *smooth_args)

    if method == 'sobel':
        edge_filter = cv2.Sobel
    elif method == 'scharr':
        edge_filter = cv2.Scharr
    else:
        raise ValueError("Derivate methods must be sobel or scharr")
    

    if direction == 'x':
        dxdy = (1, 0)
    elif direction == 'y':
        dxdy = (0, 1)
    elif direction == 'both' or direction == 'xy':
        return cv2.addWeighted(
            edge_filter(arr, -1, 1, 0, **kwargs), .5, 
            edge_filter(arr, -1, 0, 1, **kwargs), .5, 0
        )
    
    return edge_filter(arr, -1, *dxdy, **kwargs)


class ImgDerivates(ImgProc):
    def regist_bin_proc(self, bin_proc):
        if isinstance(bin_proc, ImgProc):
            self.bin_proc = bin_proc

    def proc(self, bin_proc = None, **kwargs):
        grad =  get_derivates(self.img, **kwargs)
        
        if bin_proc is not None:
            bin_proc.load_img(grad)
            return bin_proc()
        elif hasattr(self, 'bin_proc'):
            self.bin_proc.load_img(grad)
            return self.bin_proc()
        else:
            return grad