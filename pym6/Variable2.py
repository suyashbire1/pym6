import numpy as np
from netCDF4 import Dataset as dset, MFDataset as mfdset
from functools import partial
import copy
from .Plotter import plotter, rhotoz
from functools import wraps, partialmethod


def find_index_limits(dimension, start, end):
    """Finds the extreme indices of the any given dimension of the domain."""
    useful_index = np.nonzero((dimension >= start) & (dimension <= end))[0]
    lims = useful_index[0], useful_index[-1] + 1
    return lims


class MeridionalDomain():
    def __init__(self, **initializer):
        """Initializes meridional domain limits."""
        fhgeo = initializer.get('fhgeo')
        stride = initializer.get('stridey', 1)
        lath = fhgeo.variables['lath'][:]
        south_lat = initializer.get('south_lat', lath[0])
        north_lat = initializer.get('north_lat', lath[-1])
        if hasattr(self, 'indices') is False:
            self.indices = {}
        if hasattr(self, 'dim_arrays') is False:
            self.dim_arrays = {}
        self.indices['yh'] = *find_index_limits(lath, south_lat,
                                                north_lat), stride
        self.dim_arrays['yh'] = lath
        latq = fhgeo.variables['latq'][:]
        south_lat = initializer.get('south_lat', latq[0])
        north_lat = initializer.get('north_lat', latq[-1])
        self.indices['yq'] = *find_index_limits(latq, south_lat,
                                                north_lat), stride
        self.dim_arrays['yq'] = latq


class ZonalDomain():
    def __init__(self, **initializer):
        """Initializes zonal domain limits."""
        fhgeo = initializer.get('fhgeo')
        stride = initializer.get('stridex', 1)
        lonh = fhgeo.variables['lonh'][:]
        west_lon = initializer.get('west_lon', lonh[0])
        east_lon = initializer.get('east_lon', lonh[-1])
        if hasattr(self, 'indices') is False:
            self.indices = {}
        if hasattr(self, 'dim_arrays') is False:
            self.dim_arrays = {}
        self.indices['xh'] = *find_index_limits(lonh, west_lon,
                                                east_lon), stride
        self.dim_arrays['xh'] = lonh
        lonq = fhgeo.variables['lonq'][:]
        west_lon = initializer.get('west_lon', lonq[0])
        east_lon = initializer.get('east_lon', lonq[-1])
        self.indices['xq'] = *find_index_limits(lonq, west_lon,
                                                east_lon), stride
        self.dim_arrays['xq'] = lonq


class HorizontalDomain(MeridionalDomain, ZonalDomain):
    def __init__(self, **initializer):
        MeridionalDomain.__init__(self, **initializer)
        ZonalDomain.__init__(self, **initializer)


class VerticalDomain():
    def __init__(self, **initializer):
        fh = initializer.get('fh')
        stride = initializer.get('strider', 1)
        if hasattr(self, 'indices') is False:
            self.indices = {}
        if hasattr(self, 'dim_arrays') is False:
            self.dim_arrays = {}
        try:
            zl = fh.variables['zl'][:]
            low_density = initializer.get('low_density', zl[0])
            high_density = initializer.get('high_density', zl[-1])
            self.indices['zl'] = *find_index_limits(zl, low_density,
                                                    high_density), stride
            self.dim_arrays['zl'] = zl
        except:
            pass
        try:
            zi = fh.variables['zi'][:]
            low_density = initializer.get('low_density', zi[0])
            high_density = initializer.get('high_density', zi[-1])
            self.indices['zi'] = *find_index_limits(zi, low_density,
                                                    high_density), stride
            self.dim_arrays['zi'] = zi
        except:
            pass


class TemporalDomain():
    def __init__(self, **initializer):
        fh = initializer.get('fh')
        Time = fh.variables['Time'][:]
        initial_time = initializer.get('initial_time', Time[0])
        final_time = initializer.get('final_time', Time[-1])
        stride = initializer.get('stridet', 1)
        if hasattr(self, 'indices') is False:
            self.indices = {}
        self.indices['Time'] = *find_index_limits(Time, initial_time,
                                                  final_time), stride
        if hasattr(self, 'dim_arrays') is False:
            self.dim_arrays = {}
        self.dim_arrays['Time'] = Time


class Domain(TemporalDomain, VerticalDomain, HorizontalDomain):
    def __init__(self, **initializer):
        TemporalDomain.__init__(self, **initializer)
        VerticalDomain.__init__(self, **initializer)
        HorizontalDomain.__init__(self, **initializer)


class BoundaryCondition():
    def __init__(self, bc_type, axis, start_or_end):
        self.bc_type = bc_type
        self.axis = axis
        self.start_or_end = start_or_end

    def set_halo_indices(self):
        if self.array is None:
            raise AttributeError('Array does not exist.')
        if self.bc_type == 'circsymq':
            self.take_index = 1 if self.start_or_end == 0 else -2
        else:
            self.take_index = self.start_or_end

    def create_halo(self):
        self.set_halo_indices()
        self.halo = np.take(self.array, [self.take_index], axis=self.axis)
        if self.bc_type == 'zeros':
            self.halo = np.zeros(self.halo.shape)
        return self

    def boudary_condition_type(self):
        if self.bc_type != 'mirror':
            self.halo = -self.halo

    def append_halo_to_array(self):
        self.boudary_condition_type()
        if self.start_or_end == 0:
            array1 = self.halo
            array2 = self.array
        elif self.start_or_end == -1:
            array1 = self.array
            array2 = self.halo
        self.array = np.concatenate((array1, array2), axis=self.axis)
        return self


class GridVariable2(Domain):
    _loc_registry_hor = dict(
        u=['yh', 'xq'], v=['yq', 'xh'], h=['yh', 'xh'], q=['yq', 'xq'])
    _loc_registry_ver = dict(l='zl', i='zi')
    BC = dict(
        u=[('circsymh', 2, 1), ('circsymq', 3, 1), ('circsymh', 2, 0),
           ('circsymq', 3, 0), ('mirror', 1, 0), ('zeros', 1, 1)],
        v=[('circsymq', 2, 1), ('circsymh', 3, 1), ('circsymq', 2, 0),
           ('circsymh', 3, 0), ('mirror', 1, 0), ('zeros', 1, 1)],
        h=[('circsymh', 2, 1), ('circsymh', 3, 1), ('circsymh', 2, 0),
           ('circsymh', 3, 0), ('mirror', 1, 0), ('zeros', 1, 1)],
        q=[('circsymq', 2, 1), ('circsymq', 3, 1), ('circsymq', 2, 0),
           ('circsymq', 3, 0), ('mirror', 1, 0), ('zeros', 1, 1)])

    def __init__(self, var, fh, final_loc=None, **initializer):
        self._v = fh.variables[var]
        self._dimensions = self._v.dimensions
        self.determine_location()
        initializer['fh'] = fh
        Domain.__init__(self, **initializer)
        if final_loc:
            self._final_loc = final_loc
            self.get_final_location_dimensions()
        else:
            self._final_loc = self._hloc + self._vloc
            self._final_dimensions = self._dimensions
        self.operations = []

    def determine_location(self):
        dims = self._dimensions
        if 'xh' in dims and 'yh' in dims:
            self._hloc = 'h'
        elif 'xq' in dims and 'yq' in dims:
            self._hloc = 'q'
        elif 'xq' in dims and 'yh' in dims:
            self._hloc = 'u'
        elif 'xh' in dims and 'yq' in dims:
            self._hloc = 'v'
        if 'zl' in dims:
            self._vloc = 'l'
        elif 'zi' in dims:
            self._vloc = 'i'

    def get_final_location_dimensions(self):
        self._final_hloc = self._final_loc[0]
        self._final_vloc = self._final_loc[1]
        final_vdim = self._loc_registry_ver[self._final_vloc]
        final_hdims = self._loc_registry_hor[self._final_hloc]
        self._final_dimensions = tuple(['Time', final_vdim, *final_hdims])

    def modify_index(self, axis, startend, value):
        dim = self._final_dimensions[axis]
        axis_indices = list(self.indices[dim])
        axis_indices[startend] += value
        self.indices[dim] = tuple(axis_indices)
        return self

    xsm = partialmethod(modify_index, 3, 0, -1)
    xep = partialmethod(modify_index, 3, 1, 1)
    ysm = partialmethod(modify_index, 2, 0, -1)
    yep = partialmethod(modify_index, 2, 1, 1)
    zsm = partialmethod(modify_index, 1, 0, -1)
    zep = partialmethod(modify_index, 1, 1, 1)

    def get_slice(self):
        self._slice = []
        dims = self._final_dimensions
        actual_dims = self._dimensions
        for i, dim in enumerate(dims):
            indices = list(self.indices[dim])
            if indices[0] < 0:
                indices[0] = 0
            actual_dim = actual_dims[i]
            if indices[1] > self.dim_arrays[actual_dim].size:
                indices[1] = self.dim_arrays[actual_dim].size
            self._slice.append(slice(*indices))
        self._slice = tuple(self._slice)
        return self

    def read(self):
        self.array = self._v[self._slice]
        return self

    def update_boundaries(self):
        dims = self._dimensions
        for i, dim in enumerate(dims):
            indices = self.indices[dim]
            if indices[0] < 0:
                BoundaryCondition.__init__(self, *BC[self._hloc])
            if indices[1] > self.dim_arrays[actual_dim].size:
                BoundaryCondition.__init__(self, *BC[self._hloc])

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def final_dimensions(self):
        return self._final_dimensions

    @property
    def hloc(self):
        """Horizontal location of the variable on the grid.
        h : tracer,
        q : vorticity,
        u : zonal velocity,
        v : meridional velocity."""
        return self._hloc

    @property
    def vloc(self):
        """Vertical location of the variable on the grid.
        l : layer,
        i : interface."""
        return self._vloc

    def lazily_evaluate(self, func):
        @wraps(func)
        def decorated(*args, **kwargs):
            self.operations.append(func, args, kwargs)

        return decorated

    def compute(self):
        for ops in self.operations:
            self.array = ops()


class GridVariable():
    """A class to hold a variable."""

    def __init__(self, var, domain, fh, **kwargs):
        self._v = fh.variables[var]
        self._dims = self._v.dimensions
        self._name = kwargs.get('name', var)
        self._units = kwargs.get('units', None)
        self._math = kwargs.get('math', None)
        self.var = var
        self.domain = domain
        self.determine_location()

    def determine_location(self):
        dims = self._dims
        if 'xh' in dims and 'yh' in dims:
            self._hloc = 'h'
        elif 'xq' in dims and 'yq' in dims:
            self._hloc = 'q'
        elif 'xq' in dims and 'yh' in dims:
            self._hloc = 'u'
        elif 'xh' in dims and 'yq' in dims:
            self._hloc = 'v'
        if 'zl' in dims:
            self._vloc = 'l'
        elif 'zi' in dims:
            self._vloc = 'i'

    @property
    def shape(self):
        return self.values.shape if hasattr(self, 'values') else None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        assert isinstance(name, str)
        self._name = name

    @property
    def math(self):
        return self._math

    @math.setter
    def math(self, math):
        assert isinstance(math, str)
        self._math = math

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units):
        assert isinstance(units, str)
        self._units = units

    @property
    def hloc(self):
        """Horizontal location of the variable on the grid.
        h : tracer,
        q : vorticity,
        u : zonal velocity,
        v : meridional velocity."""
        return self._hloc

    @property
    def vloc(self):
        """Vertical location of the variable on the grid.
        l : layer,
        i : interface."""
        return self._vloc
