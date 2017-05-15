import numpy as np
from netCDF4 import Dataset as dset, MFDataset as mfdset
from functools import partial

class GridNdarray(np.ndarray):
    """A class to hold a grid-located variable."""
    def __new__(cls,input_array,loc):
        obj = input_array.view(cls)
        obj.loc = loc
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.loc = getattr(obj, 'loc', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

class GridVariable():

    """A class to hold a variable."""
    def __init__(self,var,domain,loc,*fhl,**kwargs):
        for fh in fhl:
            try:
                self._v = fh.variables[var]
            except KeyError:
                print('Variable not found. Trying next file.')
            else:
                self.var = var
                self.dom = domain
                self.loc = loc
                self.plot_loc = kwargs.get('plot_loc',self.loc)[0]
                self._plot_slice = self.dom.slices[self.plot_loc].copy()
                self.implement_syntactic_sugar_for_plot_slice()
                self.Time = fh.variables['Time'][:]
                self.dom.dt = np.diff(self.Time[:2])*3600
                average_DT = fh.variables['average_DT'][:]
                average_DT = average_DT[:,np.newaxis,np.newaxis,np.newaxis]
                self.average_DT = average_DT
                break

        self._divisor = kwargs.get('divisor',None)
        if self._divisor:
            for fh in fhl:
                try:
                    self._div = fh.variables[self._divisor]
                except KeyError:
                    print('Divisor not found. Trying next file.')
                else:
                    self._htol = kwargs.get('htol',1e-3)
                    break

    def __add__(self,other):
        if self.values.loc == other.values.loc:
            new_variable = GridVariable(self.var,self.dom,self.values.loc)
            new_variable.values = self.values + other.values
            return new_variable
        else:
            raise ValueError('The two variables are not co-located.')

    def __sub__(self,other):
        if self.values.loc == other.values.loc:
            new_variable = GridVariable(self.var,self.dom,self.values.loc)
            new_variable.values = self.values - other.values
            return new_variable
        else:
            raise ValueError('The two variables are not co-located.')

    def __mul__(self,other):
        if self.values.loc == other.values.loc:
            new_variable = GridVariable(self.var,self.dom,self.values.loc)
            new_variable.values = self.values * other.values
            return new_variable
        else:
            raise ValueError('The two variables are not co-located.')

    def __div__(self,other):
        if self.values.loc == other.values.loc:
            new_variable = GridVariable(self.var,self.dom,self.values.loc)
            new_variable.values = self.values / other.values
            return new_variable
        else:
            raise ValueError('The two variables are not co-located.')

    def __neg__(self):
        self.values *= -1
        return self

    @property
    def plot_slice(self):
        return self._plot_slice

    @plot_slice.setter
    def plot_slice(self,operation_string):
        axis,limit,operation = list(operation_string)

        axes = ['t','l','y','x']
        limits = ['s','e']
        operations = { 'm': -1,
                       'p': +1 }
        self._plot_slice[axes.index(axis)][limits.index(limit)] += operations[operation]

    def plot_slice_modifier(self,string):
        self.plot_slice = string
        return self

    def implement_syntactic_sugar_for_plot_slice(self):
        axes = ['t','l','y','x']
        limits = ['s','e']
        operations = ['m','p']
        for axis in axes:
            for limit in limits:
                for op in operations:
                    string = axis+limit+op
                    setattr(self,string,partial(self.plot_slice_modifier,string))


    @staticmethod
    def extend_halos(array,axis,boundary_index,**kwargs):
        method = kwargs.get('method')
        if method == 'vorticity':
            slip = kwargs.get('slip',False)
            slip_multiplyer = 1 if slip else -1
            array_extendor = slip_multiplyer*array.take([boundary_index],axis=axis)
        elif method == 'mirror':
            array_extendor = array.take([boundary_index],axis=axis)
        elif method == 'zeros':
            array_extendor = np.zeros(array.take([boundary_index],axis=axis).shape)

        if boundary_index == 0:
            array1 = array_extendor
            array2 = array
        else:
            array1 = array
            array2 = array_extendor
        array = np.append(array1,array2,axis=axis)
        return array

    def read_array(self,extend_kwargs={},**kwargs):
        self._slice = self._slice_array_to_slice(self._plot_slice)
        out_array = self._v[self._slice]
        if np.ma.isMaskedArray(out_array):
            filled = kwargs.get('filled',np.nan)
            out_array = out_array.filled(filled)

        if self._divisor:
            divisor = self._div[self._slice]
            if np.ma.isMaskedArray(divisor):
                filled = kwargs.get('filled',np.nan)
                divisor = divisor.filled(filled)
            divisor[divisor<self._htol] = np.nan
            out_array /= divisor

        divide_by_dx = kwargs.get('divide_by_dx',False)
        if divide_by_dx:
            out_array /= self.dom.dxCv[self._slice[2:]]

        divide_by_dy = kwargs.get('divide_by_dy',False)
        if divide_by_dx:
            out_array /= self.dom.dyCu[self._slice[2:]]

        if self._plot_slice[2,0] < 0:
            out_array = self.extend_halos(out_array,axis=2,
                                          boundary_index=0,**extend_kwargs)
        if self._plot_slice[2,1] > self.dom.total_ylen:
            out_array = self.extend_halos(out_array,axis=2,
                                          boundary_index=-1,**extend_kwargs)
        if self._plot_slice[3,0] < 0:
            out_array = self.extend_halos(out_array,axis=3,
                                          boundary_index=0,**extend_kwargs)
        if self._plot_slice[3,1] > self.dom.total_xlen:
            out_array = self.extend_halos(out_array,axis=3,
                                          boundary_index=-1,**extend_kwargs)
        tmean = kwargs.get('tmean',True)
        if tmean:
            dt = self.average_DT
            out_array = np.apply_over_axes(np.sum,out_array*dt,0)/np.sum(dt)
        out_array = GridNdarray(out_array,self.loc)
        self.values = out_array
        return self

    def _modify_slice(self,axis,ns=0,ne=0):
        out_slice = self._plot_slice.copy()
        out_slice[axis,0] += ns
        out_slice[axis,1] += ne
        return out_slice

    @staticmethod
    def _slice_array_to_slice(slice_array):
        """Creates a slice object from a slice array"""
        Slice = np.s_[ slice_array[0,0]:slice_array[0,1]:slice_array[0,2],
                       slice_array[1,0]:slice_array[1,1]:slice_array[1,2],
                       slice_array[2,0]:slice_array[2,1]:slice_array[2,2],
                       slice_array[3,0]:slice_array[3,1]:slice_array[3,2] ]
        return Slice

    def o1diff(self,axis):
        possible_hlocs = dict(u = ['u','u','q','h'],
                              v = ['v','v','h','q'],
                              h = ['h','h','v','u'],
                              q = ['q','q','u','v'])
        possible_vlocs = dict(l = ['l','i','l','l'],
                              i = ['i','l','i','i'])

        out_array = np.diff(self.values,n=1,axis=axis)
        out_array.loc = (  possible_hlocs[self.values.loc[0]][axis]
                           + possible_vlocs[self.values.loc[1]][axis] )
        self.values = out_array
        return self


    def ddx(self,axis):
        possible_divisors = dict(u = [self.dom.dt, self.dom.db,
                                      self.dom.dyBu, self.dom.dxT],
                                 v = [self.dom.dt, self.dom.db,
                                      self.dom.dyT,  self.dom.dxBu],
                                 h = [self.dom.dt, self.dom.db,
                                      self.dom.dyCv, self.dom.dxCu],
                                 q = [self.dom.dt, self.dom.db,
                                      self.dom.dyCu, self.dom.dxCv])
        possible_ns = dict(u = [0,0,0,1],
                           v = [0,0,1,0],
                           h = [0,0,0,0],
                           q = [0,0,-1,-1])
        possible_ne = dict(u = [0,0,-1,0],
                           v = [0,0,0,-1],
                           h = [0,0,-1,-1],
                           q = [0,0,0,0])
        if axis == 1:
            divisor = possible_divisors[self.values.loc[0]][axis]
        else:
            ns = possible_ns[self.values.loc[0]][axis]
            ne = possible_ne[self.values.loc[0]][axis]
            slice_array = self._modify_slice(axis,ns,ne)
            _slice = self._slice_array_to_slice(slice_array)
            divisor = possible_divisors[self.values.loc[0]][axis][_slice[2:]]
        ddx = self.o1diff(axis).values/divisor
        self.values = ddx
        return self

    def move_to(self,new_loc):
        possible_hlocs = dict(u = ['q','h'],
                              v = ['h','q'],
                              h = ['v','u'],
                              q = ['u','v'])
        loc = self.values.loc
        if new_loc[0] in possible_hlocs[loc[0]]:
            axis = possible_hlocs[loc[0]].index(new_loc[0])+2
        elif new_loc[1] is not loc[1]:
            axis = 1
        out_array = 0.5*(  np.take(self.values,range(self.values.shape[axis]-1),axis=axis)
                         + np.take(self.values,range(1,self.values.shape[axis]),axis=axis)  )
        out_array.loc = new_loc
        self.values = out_array
        return self
