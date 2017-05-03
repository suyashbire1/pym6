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
    def __init__(self,var,domain,loc,*fhl,**kwargs):
        for fh in fhl:
            try:
                self._v = fh.variables[var]
            except KeyError:
                print('Trying next file.')
            else:
                self.dom = domain
                self.loc = loc
                self.output_loc = kwargs.get('output_loc',self.loc)
                self._output_slice = self.dom.slices[self.output_loc]
                self._slice = self._slice_array_to_slice(self._output_slice)
                break

    @staticmethod
    def extend_halos(array,axis,boundary_index,**kwargs):
        method = kwargs.get('method')
        if method == 'vorticity':
            slip = kwargs.get('slip')
            slip_multiplyer = 1 if slip else -1
            array_extendor = slip_multiplyer*array.take([boundary_index],axis=axis)
            if boundary_index == 0:
                array1 = array_extendor
                array2 = array
            else:
                array1 = array
                array2 = array_extendor
            array = np.append(array1,array2,axis=axis)
        if method == 'mirror':
            array_extendor = array.take([boundary_index],axis=axis)
            if boundary_index == 0:
                array1 = array_extendor
                array2 = array
            else:
                array1 = array
                array2 = array_extendor
            array = np.append(array1,array2,axis=axis)
        if method == 'zeros':
            array_extendor = np.zeros(array.take([boundary_index],axis=axis).shape)
            if boundary_index == 0:
                array1 = array_extendor
                array2 = array
            else:
                array1 = array
                array2 = array_extendor
            array = np.append(array1,array2,axis=axis)
        return array

    def read_array(self,extend_kwargs=None,**kwargs):
        out_array = self._v[self._slice]
        if self._output_slice[2,0] < 0:
            out_array = self.extend_halos(out_array,axis=2,
                                          boundary_index=0,**extend_kwargs)
        if self._output_slice[2,1] > self.dom.total_ylen:
            out_array = self.extend_halos(out_array,axis=2,
                                          boundary_index=-1,**extend_kwargs)
        if self._output_slice[3,0] < 0:
            out_array = self.extend_halos(out_array,axis=3,
                                          boundary_index=0,**extend_kwargs)
        if self._output_slice[3,1] > self.dom.total_xlen:
            out_array = self.extend_halos(out_array,axis=3,
                                          boundary_index=-1,**extend_kwargs)
        if np.ma.isMaskedArray(out_array):
            filled = kwargs.get('filled',np.nan)
            out_array = out_array.filled(filled)
        out_array = GridNdarray(out_array,self.loc)
        return out_array

    def _modify_slice(self,axis,ns,ne):
        out_slice = self._output_slice[axis,0] + ns
        out_slice = out_slice[axis,1] + ne
        return out_slice

    @staticmethod
    def _slice_array_to_slice(slice_array):
        """Creates a slice object from a slice array"""
        Slice = np.s_[ slice_array[0,0]:slice_array[0,1]:slice_array[0,2],
                       slice_array[1,0]:slice_array[1,1]:slice_array[1,2],
                       slice_array[2,0]:slice_array[2,1]:slice_array[2,2],
                       slice_array[3,0]:slice_array[3,1]:slice_array[3,2] ]
        return Slice

    @staticmethod
    def o1diff(array,axis):
        possible_locs = dict(u = ['u','u','q','h'],
                             v = ['v','v','h','q'],
                             h = ['h','h','v','u'],
                             q = ['q','q','u','v'])
        out_arr = np.diff(array,1,axis)
        out_arr.loc = possible_locs[array.loc][axis]
        return out_arr

    @staticmethod
    def move_to_neighbor(array,new_loc):
        possible_locs = dict(u = ['ul','q','h'],
                             v = ['vl','h','q'],
                             h = ['hl','v','u'],
                             q = ['ql','u','v'])
        loc = array.loc
        average_along_axis = possible_locs[loc].index(new_loc)+1
        out_array = 0.5*(  np.take(array,:-1,axis=average_along_axis)
                         + np.take(array,1:,axis=average_along_axis)  )
        return out_array
