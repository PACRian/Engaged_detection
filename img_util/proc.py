import numpy as np


class ImgProc:
    def __init__(self, img=None) -> None:
        if img is not None:
            self.load_img(img)

    def load_img(self, img):
        assert isinstance(img, np.ndarray), "Image must be of `numpy ndarray`"
        self.img = img
    
    def proc(self):
        raise TypeError("Not implemented")
    
    def __call__(self, *args, **kwds):
        return self.proc(*args, **kwds)
    

class ImgPipeline(ImgProc):
    def __init__(self, *cls) -> None:
        # A imgproc pipeline
        self.cls = cls
    
    @property
    def cls(self):
        return self._cls
    
    @cls.setter
    def cls(self, cls_ls):
        cls_ls = [cls for cls in cls_ls if callable(cls)]
        if cls_ls:
            self._cls = cls_ls
    
    def proc(self, keep_idx=None, args_dict={}, kwargs_dict={}):
        '''
        keep_idx: bool | list 
        if `keep_idx` is set to be True, then each image is recorded.
        '''
        if keep_idx and isinstance(keep_idx, bool):
            keep_idx = list(range(len(self.cls)))
        else:
            keep_idx = [len(self.cls)-1]
        print(keep_idx)

        img_list = []
        img = self.img
        for i, cls in enumerate(self.cls):
            img_proc = cls()
            img_proc.load_img(img)

            args = args_dict[i] if i in args_dict.keys() else []
            kwargs = kwargs_dict[i] if i in kwargs_dict.keys() else dict()
            img = img_proc(*args, **kwargs)
            if keep_idx and i==keep_idx[0]:
                img_list.append(img)
                keep_idx = keep_idx.pop(0)
        
        return img_list


