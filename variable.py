import numpy as np
from netCDF4 import Dataset as dset, MFDataset as mfdset
from functools import partial

class GridNdarray(np.ndarray):
    """A class to hold a grid-located variable."""
    def __new__(cls,input_array,loc):
        print('new called')
        obj = input_array.view(cls)
        obj.loc = loc
        return obj

    def __array_finalize__(self, obj):
        print('array_finalize called')
        if obj is None: return
        self.loc = getattr(obj, 'loc', None)
        print(self.loc)

    def __array_wrap__(self, out_arr, context=None):
        print('array_wrap called')
        return np.ndarray.__array_wrap__(self, out_arr, context)

class GridVariable():
    """A class to hold a variable."""
    def __init__(self,var,domain,gloc,*fhl):
        for fh in fhl:
            try:
                self._v = fh.variables[var]
            except KeyError:
                print('Trying next file.')
            else:
                self.dom = domain
                self.loc = gloc
                self.sl = domain.sl
                self._slc = domain.sl
                print(self._slc)
                self.map_extend_slice_methods()
                print(self._slc)
                break

    def update_array(self,slc):
        out_array = self._v[getattr(self,slc)]
        if np.ma.isMaskedArray(out_array):
            out_array = out_array.filled(np.nan)
        out_array = GridNdarray(out_array,self.loc)
        self.array = out_array

    @property
    def slc(self):
        return self._slc

    @slc.setter
    def slc(self,slice_operation):
        possible_changes = dict(m=-1, p=1)
        start_end = dict(m='s', p='e')
        possible_dims = dict(t=0, l=1, y=2, x=3)

        dim = slice_operation[0]
        change_by = slice_operation[1]

        if dim not in possible_dims.keys():
            raise KeyError("""{} dimension is not recognized. Possible dimensions are {}."""
                           .format(dim, possible_dims.keys()))
        if change_by not in possible_changes.keys():
            raise KeyError("""{} change is not recognized. Possible changes are {}."""
                           .format(dim, possible_changes.keys()))

        sl = self._slc
        _d = dict(ts = sl[0].start,te = sl[0].stop,
                  ls = sl[1].start,le = sl[1].stop,
                  ys = sl[2].start,ye = sl[2].stop,
                  xs = sl[3].start,xe = sl[3].stop)

        slice_dim_to_change = dim+start_end[change_by]
        if _d[slice_dim_to_change] is not None:
            _d[slice_dim_to_change] += possible_changes[change_by]

        ts, te = _d['ts'],_d['te']
        ls, le = _d['ls'],_d['le']
        ys, ye = _d['ys'],_d['ye']
        xs, xe = _d['xs'],_d['xe']
        sl = np.s_[ts:te,ls:le,ys:ye,xs:xe]
        self._slc =  sl

    def slc_changer(self,string):
        self.slc = string
        return self

    def map_extend_slice_methods(self):
        dims = ['t','l','y','x']
        operations = ['m','p']
        for dim in dims:
            for op in operations:
                setattr(self,dim+op,partial(self.slc_changer,dim+op))

    def purge_array(self):
        self.array = None
        self._slc = self.sl

    def o1diff(self,axis):
        possible_locs = dict(u = ['u','u','q','h'],
                             v = ['v','v','h','q'],
                             h = ['h','h','v','u'],
                             q = ['q','q','u','v'])
        out_arr = np.diff(self.array,1,axis)
        out_arr.loc = possible_locs[self.array.loc][axis]
        self.purge_array()
        return out_arr

    def move_to_neighbor(self,loc):
        if loc == self.array.loc: return
        possible_locs = dict(u = ['u','u','q','h'],
                             v = ['v','v','h','q'],
                             h = ['h','h','v','u'],
                             q = ['q','q','u','v'])
        possible_shifts = dict(u = [0,0,1,-1],
                               v = [0,0,-1,1],
                               h = [0,0,1,1],
                               q = [0,0,-1,-1])
        out_arr = 0.5*self.array
        sl = self.slc
        i = possible_locs[self.array.loc].index(loc)
        out_arr += np.roll(self.array, 1,axis=axis)
        out_arr.loc = possible_locs[self.array.loc][axis]
        return out_arr

    @staticmethod
    def shift_slice(sl,dim,n):
        """Extends the dimension, dim, of slice, sl, by n."""
        dims = np.zeros((4,3))
        dims[dim,:2] = n
        for i, slc in enumerate(sl):
            dims+=
        sl = np.s_[ts:te,ls:le,ys:ye,xs:xe]
        return sl
