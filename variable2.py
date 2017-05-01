import numpy as np
from netCDF4 import Dataset as dset, MFDataset as mfdset
from functools import partial

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
                self.slice_array = self.dom.slices[self.loc]
                self._slice = self.dom.slice_array_to_slice(self.slice_array)
                break

    def read_array(self):
        out_array = self._v[self._slice]
        if np.ma.isMaskedArray(out_array):
            out_array = out_array.filled(np.nan)
        out_array = GridNdarray(out_array,self.loc)
        self.array = out_array

    def modify_slice(self,axis,ns,ne):
        self.slice_array[axis,0] += ns
        self.slice_array[axis,1] += ne

    def o1diff(self,axis):
        possible_locs = dict(u = ['u','u','q','h'],
                             v = ['v','v','h','q'],
                             h = ['h','h','v','u'],
                             q = ['q','q','u','v'])
        out_arr = np.diff(self.array,1,axis)
        out_arr.loc = possible_locs[self.array.loc][axis]
        self.purge_array()
        return out_arr

    def move_to_neighbor(self,axis):
        possible_locs = dict(u = ['u','u','q','h'],
                             v = ['v','v','h','q'],
                             h = ['h','h','v','u'],
                             q = ['q','q','u','v'])
        out_arr = 0.5*self.array
        out_arr += np.roll(self.array, 1,axis=axis)
        out_arr.loc = possible_locs[self.array.loc][axis]
        return out_arr
