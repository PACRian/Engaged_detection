import abc
import math

import matplotlib.pyplot as plt
import numpy as np

# def _update_configs(configs, key, default_val):
#     assert isinstance(configs, dict), ValueError("configs should be type `dict")
    
#     v = configs.pop(key, default_val)
#     return configs, v

'''
Line plotting and processing module
Base class: 
+ LinePlotter: Abstract class for line plotting.
    Succession method: 
    1. __init__(self, lc, lx)
      where `lc` and `lx` represents the counts of total lines and
      the axis that works
    implementing method:
    1. plot(self)
+ ParametricLines: Abstract metaclass, 
    declares a parametric line(2-element space).
    Succession method: 
    1. __init__(self, *args, 
        para_names=None, 
        arg_proc=None,
        xbound=None, ybound=None,
        bdobj=None, 
        select_indices=None)
      the initialization of the parametric line, where `xbound`, `ybound` or 
      `bdobj` are the bounding rectangle that limit the lines for plotting
    2. gen_pts_by_boundary(self, 
        xbound=None, ybound=None, 
        bdobj=None, 
        select_indices=None, 
        attach=True)
      to generate the two-points of each line, then the `xdata` and `ydata`
      are filled in the "blank" as the attribute of `TwoPointLines` of its successor
    implementing(developing) method:
    1. _get_y_coords(self, xdata)
    2. _get_x_coords(self, ydata)
    3. _trans(self, cls)

Point-base class: 
+ TwoPointLines: Realization of the lines determined by the edge points
    TwoPointLines(*pdatas, [line_axis, [pstyle]])

Parameter-determined class:
+ InterceptLines: 
    Init method:
    InterceptLines(*kbdatas, [line_axis, [xbound, ybound, [bdobj], [select_indices]]])
    The MRO chain: 
    InterceptLines -> ParametricLines 
    -> TwoPointLines -> LinePlotter
+ RhoThetaLines:
    Init method:
    RhoThetaLines(*rtdatas, [line_axis, [xbound, ybound, [bdobj], [select_indices]]])
    The MRO chain: 
    RhoThetaLines -> ParametricLines 
    -> TwoPointLines -> LinePlotter

A full usage example can be checked at `unittest_lines.ipynb`
'''

def _to_array(v):
    return v if isinstance(v, np.ndarray) else np.array(v)

class RectBoundary:
    def __init__(self, xbound=None, ybound=None) -> None:
        if xbound is None and ybound is None:
            raise ValueError("Both `xbound` and `ybound` can not be 'NoneType' at the same time")
        
        self.bound = (xbound, ybound)   

    def __str__(self) -> str:
        return f'RectBound object: \
            \nX-axis range:{self._xbound};\nY-axis range:{self._ybound}.'  
    
    @staticmethod
    def _isnumeric(v):
        try:
            r = all(np.issubdtype(i, np.number) for i in v)
        except TypeError:
            r = all(isinstance(i, (int, float)) for i in v)
        finally:
            return r
    
    @staticmethod
    def _get_bound(b):
        try:
            m, M = min(b), max(b)
        except TypeError:
            raise ValueError('`bound` is not iterable')

        if m==M:
            raise ValueError('`bound` has no varying range')
        
        return [m, M]

    @property
    def xbound(self): # the getter of attr `xbound` 
        return self._xbound

    @xbound.setter
    def xbound(self, xbound):
        if xbound is None:
            self._xbound = None
            return 

        if self._isnumeric(xbound):
            self._xbound = self._get_bound(xbound)
        else:
            raise ValueError("each element of `xbound` should be numerical")
    
    @property
    def ybound(self): # the getter of attr `ybound` 
        return self._ybound
    
    @ybound.setter
    def ybound(self, ybound):
        if ybound is None:
            self._ybound = None
            return 

        if self._isnumeric(ybound):
            self._xbound = self._get_bound(ybound)
        else:
            raise ValueError("each element of `ybound` should be numerical")

    @property
    def bound(self):
        return dict(xbound=self._xbound, ybound=self._ybound)
    
    @bound.setter
    def bound(self, con_bound):
        self._xbound, self._ybound = con_bound[0], con_bound[1]
            

class LinePlotter:
    def __init__(self, line_counts, line_axis) -> None:
        self.lc = line_counts
        self.lx = line_axis
    
    def plot(self, *args, **kwargs):
        raise ValueError("Not implemented")

        # if len(args)==1:
        #     sedata = _to_array(args[0])
        #     # assert xedata.shape[-1]==4 and xedata.ndim<3, \
        #     #     ValueError('Xedata should be a vectorin the form of (x1, y1, x2, y2) or a M-by-4(4-by-M) array')

        #     sdata, edata = np.split(sedata, 2, axis=line_axis-1)        
        # elif len(args)==2:
        #     sdata, edata = _to_array(args[0]), -_to_array(args[1])
        # else:
        #     raise ValueError("Too many data arguments input")

class TwoPointLines(LinePlotter):

    @staticmethod
    def _biargs_sep(args, deco=None, sepper=None, escape_len=[]):
        if len(args)==1:
            com_arg = deco(args[0]) if callable(deco) else args[0]
            return sepper(com_arg)
        elif len(args)==2:
            if callable(deco):
                return deco(args[0]), deco(args[1])
            return args
        elif len(args) in escape_len:
            return 
        else:
            raise ValueError("Arguments too many(>2)")

    def __init__(self, *args, line_axis=0, pstyle=True) -> None:

        sdata, edata = self._biargs_sep(args, \
            deco=_to_array, sepper=lambda v: np.split(v, 2, axis=line_axis-1))

        assert sdata.shape[line_axis-1]==2 and edata.shape[line_axis-1]==2 \
                and sdata.shape[line_axis]==edata.shape[line_axis] and sdata.ndim<3 and edata.ndim<3, \
                    ValueError('x or y data not correctly given')

        lc = sdata.shape[line_axis] if sdata.ndim==2 else 1
        super().__init__(lc, line_axis)

        if pstyle:
            if lc==1:
                self.xdata, self.ydata = np.array([sdata[0], edata[0]]), \
                    np.array([sdata[1], edata[1]])
                return 
            x1, y1 = np.split(sdata, [1], axis=line_axis-1)
            x2, y2 = np.split(edata, [1], axis=line_axis-1)
            self.xdata, self.ydata = np.hstack((x1, x2)).T, np.hstack((y1, y2)).T
        else:
            self.xdata, self.ydata = sdata.reshape(2, -1), edata.reshape(2, -1)
    
    def __getitem__(self, k):
        if self.lc==1:
            xdata, ydata = self.xdata, self.ydata
        else:
            xdata, ydata = self.xdata[:, k], self.ydata[:, k]
        if xdata.ndim==1:
            xdata, ydata = xdata.reshape(-1, 1), ydata.reshape(-1, 1)

        return TwoPointLines(xdata, ydata, line_axis=1, pstyle=False)

    def __str__(self) -> str:
        if self.lc==1:
            return f"TwoPoints line from ({self.xdata[0]}, {self.ydata[0]}) to ({self.xdata[1]}, {self.ydata[1]})"
        else:
            return f"TwoPoints line with {self.lc} counts"

    def tran_to(self, cls, return_cls=False, tran_configs={}, cls_configs={}):
        if self.__class__ == cls:
            return # No transformation should be done
        
        paras = cls._trans(self.xdata, self.ydata, **tran_configs)
        if return_cls:
            return cls(paras, **cls_configs)
        return paras
    
    def array(self,  pstyle=False, indices=None):

        if indices is None or self.lc==1:
            xdata, ydata = self.xdata, self.ydata
        else:
            xdata, ydata = self.xdata[:, indices], self.ydata[:, indices]
        
        if pstyle:
            packed = np.stack((xdata, ydata)).reshape(4, -1, order='F')
            return np.split(packed, 2)
        else:
            return xdata, ydata

    def _select_by_indices(self, data=None, indices=None, axis=0):

        if data is None:
            data = [self.xdata, self.ydata]

        if indices is not None and self.lc > 1:
            ind = [i for i in indices if i<=self.lc]
            print(f'axis={axis}')
            if axis==0:
                return [l[ind] for l in data]
            elif axis==1:
                return [l[..., ind] for l in data]
            else:
                raise ValueError("`axis` should be only assigned as 0 or 1")
        else:
            return data

    def plot(self, ax=None, select_indices=None, **kwargs):
        xdata, ydata = self._select_by_indices(indices=select_indices, axis=1)

        if ax is None:
            p=plt.plot(xdata, ydata, **kwargs)
        else:
            p=ax.plot(xdata, ydata, **kwargs)
        
        return p
    
    def mark_vertex(self, ax=None, select_indices=None, edge=None, **kwargs):
        xdata, ydata = self._select_by_indices(self, select_indices)

        if edge==0 or edge==1:
            xdata, ydata = xdata[edge], ydata[edge]
        elif edge is not None:
            raise ValueError("`edge` paras should only \
            select to be 0 or 1(Meaning the 'left' and \
                'right' endpoint of this line.")

        dots = np.stack((xdata, ydata)).flatten(order='F')
        kwargs = dict(marker="s", mfc="none").update(kwargs)
        if ax is None:
            p=plt.plot(*dots, **kwargs)
        else:
            p = ax.plot(*dots, **kwargs)
        
        return p


class ParametricLines(metaclass=abc.ABCMeta):   
    __para_names__ = ()
    def __init__(self, *args, 
        para_names=None, 
        arg_proc=None,
        xbound=None, ybound=None,
        bdobj=None, 
        select_indices=None) -> None:
        # variable check `args`
        if len(args)==0:
            raise ValueError('No valid arguments input')

        # Variable check `para_names`
        try:
            if para_names is None:
                para_names = self.__para_names__

            iter(para_names)
        except TypeError:
            msg = '''`para_names` shoule be iterable.
            Eg: para_names=['para1', 'para2']'''
            raise ValueError(msg)
        
        # Variable check `arg_proc`
        assert callable(arg_proc), \
            '`arg_proc` is a pre-processor to deal with length-variable arguments'
        
        # Set line parameters 
        paras = arg_proc(args)
        assert len(paras)==len(para_names), \
            'The number of arguments can not fit with pre-defined names'

        _lc=0
        for pn, pv in zip(para_names, paras):
            assert _lc==0 or _lc==pv.size, 'Parameter can not got fully grouped'
            _lc = pv.size
            setattr(self, pn, pv)
        
        # Set line points(If boundary knowledge is given) 可选
        if xbound is None and ybound is None and bdobj is None:
            self.lc = _lc
            return 

        # Init points
        xdata, ydata, _ = self.set_pts(xbound, ybound, bdobj, select_indices, attach=False)
        super().__init__(xdata, ydata, line_axis=1, pstyle=False)
    
    def __getitem__(self, k):
        if self.lc==1:
            para_vas = [getattr(self, pn) for pn in self.__para_names__]
        else:
            para_vas = [getattr(self, pn)[:, k] for pn in self.__para_names__]

        cls = self.__class__
        return cls(*para_vas, bdobj=getattr(self, 'bound', None))

    def _raise_base_err(self):
        err_msg = '''
            This class is abstract, which should not be instantiated
            If you see this error, please implement another specific class based on {curr_type}'''
        raise TypeError(err_msg.format(self.__class__.__name__))

    @abc.abstractmethod
    def _get_y_coords(self, _):
        # Note:
        # RE implemented it !
        pass

    @abc.abstractmethod
    def _get_x_coords(self, _):
        pass

    @abc.abstractmethod
    def _trans(self, _):
        pass
    # @abc.abstractmethod
    # def _pts_to_paras(self, xdata, ydata):
    #     self._raise_base_err()

    @property
    def bound(self):
        if hasattr(self, '_bound'):
            return self._bound
    
    @bound.setter
    def bound(self, bound):
        if isinstance(bound, RectBoundary):
            self._bound=bound

    def tran_to(self, cls, return_cls, **kwargs):
        self_cls = self.__class__
        if cls==self_cls:
            return 

        try:
            x, y =  self.xdata, self.ydata
        except:
            x, y, _ =  self.gen_pts_by_boundary(xbound=[0, 1], attach=False)
        
        if cls==TwoPointLines and not return_cls:
            return x, y

        line_obj = TwoPointLines(x, y, line_axis=1, pstyle=False)
        return line_obj.tran_to(cls, return_cls=return_cls, **kwargs)
    
    def set_pts(self, 
        xbound=None, ybound=None, 
        bdobj=None, 
        select_indices=None,
        attach=True):
        
        try:
            self.bound = RectBoundary(xbound, ybound) if bdobj is None else bdobj
        except ValueError:
            # Use default bound
            if not hasattr(self, 'bound'):
                raise ValueError("No valid boundary knowledge given(please set `xbound`, `ybound` or `bdobj`")
        # NOTE: if `xbound`, `ybound` and `bdobj` has set to be `NoneType`, 
        # then this line won't work so that it'll use old bound knowledge(`self.bound`)
        return self.gen_pts_by_boundary(bdobj=self.bound, select_indices=select_indices, attach=attach)

    def gen_pts_by_boundary(self, 
        xbound=None, ybound=None, 
        bdobj=None, 
        select_indices=None, 
        attach=True):
        _lc = getattr(self, 'lc', getattr(self, self.__para_names__[0]).size)

        if isinstance(bdobj, RectBoundary):
            xbound, ybound = bdobj.bound.values()

        e_flag = 0
        if xbound is not None:
            xbound = [min(xbound), max(xbound)]
            ydata = self._get_y_coords(xbound) # Note here
            if ybound is None:
                xdata = np.repeat(np.array(xbound)[:, np.newaxis], _lc, axis=1)
                e_flag = 1
                _exceed = None
        
        if ybound is not None:
            xdata = self._get_x_coords(ybound) # 
            if xbound is None:
                ydata = np.repeat(np.array(ybound)[:, np.newaxis], _lc, axis=1)
                e_flag = 1
                _exceed = None

        def con_sort(ldata, small_v=None, large_v=None, small_ind=0, large_ind=1):
            _is_ascend = ldata[large_ind]>ldata[small_ind]
            ldata = np.where(_is_ascend, ldata, np.flipud(ldata))
            if small_v is not None and large_v is not None:
                varray = np.vstack((np.where(_is_ascend, small_v, large_v), np.where(_is_ascend, large_v, small_v)))
                return _is_ascend, ldata, varray
            return _is_ascend, ldata

        if not e_flag:
            try:
                _, xdata, ybound = con_sort(xdata, ybound[0], ybound[1])
            except UnboundLocalError:
                raise ValueError('`xbound` and `ybound` are not defined.')

            _should_choose_p3 = xdata[0]>xbound[0]
            _should_choose_p4 = xdata[1]<xbound[1]
            _exceed = (xdata[0]>xbound[1]) | (xdata[1]<xbound[0])

            x1 = np.where(_should_choose_p3, xdata[0], xbound[0])
            x2 = np.where(_should_choose_p4, xdata[1], xbound[1])
            y1 = np.where(_should_choose_p3, ybound[0], ydata[0])
            y2 = np.where(_should_choose_p4, ybound[1], ydata[1])
            xdata, ydata = np.vstack((x1, x2)), np.vstack((y1, y2))

        if attach:
            self.xdata, self.ydata = xdata, ydata

        return (*self._select_by_indices(data=[xdata, ydata], indices=select_indices, axis=1), _exceed)
        # self.xdata, self.ydata = self._select_by_indices(x, y, select_indices)

    def plot(self, 
        xbound=None, ybound=None, 
        bdobj=None,
        select_indices=None, 
        pre_style=True,
        back_img=None,
        ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()
        
        if back_img is not None:
            ax.imshow(back_img)
            return self.plot(ax.xlim(), ax.ylim(), select_indices, \
                pre_style, back_img=None, ax=ax, **kwargs)

        self.set_pts(xbound, ybound, bdobj, select_indices, attach=True)
        

        # NOTE: The MRO chain should be like:
        # SpecificParasLines -> ParametricLines -> TwoPointLines
        p = super().plot(ax, select_indices=select_indices, **kwargs)

        if pre_style:
            ax.grid(); ax.axis('equal')
            ax.set_xlim(xbound); ax.set_ylim(ybound)
        
        return p


class InterceptLines(ParametricLines, TwoPointLines):
    __para_names__ = ('k', 'b')
    def __init__(self, *args, 
        line_axis=0,
        xbound=None, ybound=None, 
        bdobj=None, 
        select_indices=None) -> None:
        # arg_proc = lambda args: self._biargs_sep(args, \
        #     deco=_to_array, sepper=lambda v: np.split(v, 2, axis=line_axis-1))
        # super().__init__(*args, para_names=['k', 'b'], arg_proc=arg_proc)

        _proc = lambda args: [p.reshape(1, -1) for p in self._biargs_sep(args, \
                deco=_to_array, sepper=lambda v: np.split(v, 2, axis=line_axis-1))]
        super().__init__(*args, arg_proc=_proc, \
             xbound=xbound, ybound=ybound, bdobj=bdobj, select_indices=select_indices)

        assert self.k.size == self.b.size, ValueError("Cannot pair each arguments group")

    def __str__(self) -> str:
        TOP_PAGES = 5
        end_idx=self.lc+1 if self.lc<=TOP_PAGES else TOP_PAGES+1

        return f"Intercept-parametered lines\n (k,b) pairs as below[with top {end_idx-1} lines]:\n" \
            + ', '.join([f"({k} {b})" for k, b in zip(self.k[:, :end_idx].ravel(), self.b[:, :end_idx].ravel())])
    
    @staticmethod
    def _trans(xdata, ydata):
        assert xdata.shape[0]==ydata.shape[0]==2, xdata.size==ydata.size
        
        _get_diff = lambda x: x[1]-x[0]
        dx, dy, d=_get_diff(xdata), _get_diff(ydata), _get_diff(xdata*np.flipud(ydata))
        return dy/dx, d/dx

    def _get_x_coords(self, ybound):
        '''
        The intercept-line formula is:
        $$ y = kx+b $$
        where $(k,b)$ is the parameter pair,
        we can solve x from y, given by:
        $$ x = (y-b)/k $$

        If k=0, then the point should be:
        $$ (x_{b1}, y_{given}), (x_{b2}, y_{given}) $$
        '''
        k_reciproc=1/self.k
        return np.array([ybound]).T @ k_reciproc - self.b*k_reciproc

    def _get_y_coords(self, xbound):
        '''
        $$ y = kx+b $$
        '''

        return np.array([xbound]).T @ self.k + self.b


class RhoThetaLines(ParametricLines, TwoPointLines):
    __para_names__ = ('rho', 'theta')
    def __init__(self, *args, 
        line_axis=0,
        xbound=None, ybound=None, 
        bdobj=None, 
        select_indices=None, 
        is_deg=False) -> None:

        _proc = lambda args: [p.reshape(1, -1) for p in self._biargs_sep(args, \
                deco=_to_array, sepper=lambda v: np.split(v, 2, axis=line_axis-1))]
        super().__init__(*args, arg_proc=_proc, \
             xbound=xbound, ybound=ybound, bdobj=bdobj, select_indices=select_indices)

        if is_deg:
            self.theta = np.deg2rad(self.theta)

    @staticmethod
    def _trans(xdata, ydata, plane=1):
        assert plane==1 or plane==-1, ValueError("`plane` arguments must be 1 or -1, \
            determine whether it sits in the right or the left numberplane.")
        '''
        \rho = D / \sqrt{\Delta^2 x + \Delta^2 y}
        \theta = - \arctan \Delta x /  \Delta y

        '''
        assert xdata.shape[0]==ydata.shape[0]==2, xdata.size==ydata.size
        
        _get_diff = lambda x: x[1]-x[0]
        dx, dy, d=_get_diff(xdata), _get_diff(ydata), _get_diff(xdata*np.flipud(ydata))
        return plane*d/np.sqrt(dx**2+dy**2), -np.arctan(dx/dy)

    def _get_x_coords(self, ybound):
        '''
        x *\cos\theta + y*\sin\theta = \rho

        According the x-coordination, we can solve that:
        y = \frac \rho \sin\theta - x*\cot\theta (1)

        Also we can have:
        x = \frac \rho \cos\theta -y*\tan\theta (2)

        The total algorithm defines 4 point namely P1, P2, P3 and P4
        P1, P2 are the points based on the x-axis boundary according to eq(1)
        so that P3, P4 are the points from eq(2)
        A clear fact we can see is that:
        P1 and P3 can HAVE and ONLY HAVE chosen as the "left vertex" of this line, while 
        the "right vertex" should be chosen from the group composed of P2 and P4
        Basic geometric knowledge shows that if $min{X_{bound}}<x_{P_3}<max{X_{bound}}$, then P3 should be selected 
        otherwise P1
        '''
        return self.rho/np.cos(self.theta) - np.array([ybound]).T @ np.tan(self.theta)

    def _get_y_coords(self, xbound):
        return self.rho/np.sin(self.theta) - np.array([xbound]).T @ (1/np.tan(self.theta))


class GridLines(ParametricLines, TwoPointLines): 
    __para_names__=('direction', 'intercept')
    def __init__(self, *args, 
        line_axis=0,
        xbound=None, ybound=None, 
        bdobj=None, 
        select_indices=None) -> None:

        def _proc(self, args):
            return self._biargs_sep(args, \
                deco=_to_array, sepper=lambda v: np.split(v, 1, axis=line_axis-1))

        super().__init__(*args, arg_proc=_proc, \
            xbound=xbound, ybound=ybound, bdobj=bdobj, select_indices=select_indices)

    # def _get_x_coords(self, ybound):
    #     return super()._get_x_coords(_)