import numpy as np
from functools import partial, partialmethod
from collections import OrderedDict


def find_index_limits(dimension, start, end):
    """Finds the extreme indices of the any given dimension of the domain."""
    useful_index = np.nonzero((dimension >= start) & (dimension <= end))[0]
    lims = useful_index[0], useful_index[-1] + 1
    return lims


class MeridionalDomain():
    def __init__(self, **initializer):
        """Initializes meridional domain limits."""
        fh = initializer.get('fh')
        stride = initializer.get('stridey', 1)
        if hasattr(self, 'indices') is False:
            self.indices = {}
        if hasattr(self, 'dim_arrays') is False:
            self.dim_arrays = {}
        try:
            yh = fh.variables['yh'][:]
            south_lat = initializer.get('south_lat', yh[0])
            north_lat = initializer.get('north_lat', yh[-1])
            self.indices['yh'] = *find_index_limits(yh, south_lat,
                                                    north_lat), stride
            self.dim_arrays['yh'] = yh
        except KeyError:
            pass
        try:
            yq = fh.variables['yq'][:]
            south_lat = initializer.get('south_lat', yq[0])
            north_lat = initializer.get('north_lat', yq[-1])
            self.indices['yq'] = *find_index_limits(yq, south_lat,
                                                    north_lat), stride
            self.dim_arrays['yq'] = yq
        except KeyError:
            pass


class ZonalDomain():
    def __init__(self, **initializer):
        """Initializes zonal domain limits."""
        fh = initializer.get('fh')
        stride = initializer.get('stridex', 1)
        if hasattr(self, 'indices') is False:
            self.indices = {}
        if hasattr(self, 'dim_arrays') is False:
            self.dim_arrays = {}
        try:
            xh = fh.variables['xh'][:]
            west_lon = initializer.get('west_lon', xh[0])
            east_lon = initializer.get('east_lon', xh[-1])
            self.indices['xh'] = *find_index_limits(xh, west_lon,
                                                    east_lon), stride
            self.dim_arrays['xh'] = xh
        except KeyError:
            pass
        try:
            xq = fh.variables['xq'][:]
            west_lon = initializer.get('west_lon', xq[0])
            east_lon = initializer.get('east_lon', xq[-1])
            self.indices['xq'] = *find_index_limits(xq, west_lon,
                                                    east_lon), stride
            self.dim_arrays['xq'] = xq
        except KeyError:
            pass


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
        except KeyError:
            pass
        try:
            zi = fh.variables['zi'][:]
            low_density = initializer.get('low_density', zi[0])
            high_density = initializer.get('high_density', zi[-1])
            self.indices['zi'] = *find_index_limits(zi, low_density,
                                                    high_density), stride
            self.dim_arrays['zi'] = zi
        except KeyError:
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


class LazyNumpyOperation():
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, array):
        return self.func(array, *self.args, **self.kwargs)


class BoundaryCondition():
    def __init__(self, bc_type, axis, start_or_end):
        self.bc_type = bc_type
        self.axis = axis
        self.start_or_end = start_or_end

    def set_halo_indices(self):
        if self.bc_type == 'circsymq':
            take_index = 1 if self.start_or_end == 0 else -2
        else:
            take_index = self.start_or_end
        return take_index

    def create_halo(self, array):
        take_index = self.set_halo_indices()
        self.halo = np.take(array, [take_index], axis=self.axis)
        if self.bc_type == 'zeros':
            self.halo = np.zeros(self.halo.shape)

    def boudary_condition_type(self):
        if self.bc_type != 'mirror':
            self.halo = -self.halo

    def append_halo_to_array(self, array):
        self.boudary_condition_type()
        if self.start_or_end == 0:
            array1 = self.halo
            array2 = array
        elif self.start_or_end == -1:
            array1 = array
            array2 = self.halo
        array = np.concatenate((array1, array2), axis=self.axis)
        return array

    def __call__(self, array):
        self.create_halo(array)
        return self.append_halo_to_array(array)


class GridVariable2(Domain):
    def __init__(self,
                 var,
                 fh,
                 final_loc=None,
                 bc_type=None,
                 fillvalue=0,
                 **initializer):
        self._v = fh.variables[var]
        self._current_dimensions = list(self._v.dimensions)
        self._fillvalue = fillvalue
        self.determine_location()
        initializer['fh'] = fh
        Domain.__init__(self, **initializer)
        if final_loc:
            self._final_loc = final_loc
            self.get_final_location_dimensions()
        else:
            self._final_loc = self._current_hloc + self._current_vloc
            self._final_dimensions = tuple(self._current_dimensions)
        self._bc_type = bc_type
        self.array = None
        self.operations = []

    def determine_location(self):
        dims = self._current_dimensions
        if 'xh' in dims and 'yh' in dims:
            self._current_hloc = 'h'
        elif 'xq' in dims and 'yq' in dims:
            self._current_hloc = 'q'
        elif 'xq' in dims and 'yh' in dims:
            self._current_hloc = 'u'
        elif 'xh' in dims and 'yq' in dims:
            self._current_hloc = 'v'
        if 'zl' in dims:
            self._current_vloc = 'l'
        elif 'zi' in dims:
            self._current_vloc = 'i'

    def return_dimensions(self):
        dims = self._current_dimensions
        return_dims = OrderedDict()
        for dim in dims:
            start, stop, stride = self.indices[dim]
            return_dims[dim] = self.dim_arrays[dim][start:stop:stride]
        return return_dims

    @staticmethod
    def get_dimensions_by_location(loc):
        loc_registry_hor = dict(
            u=['yh', 'xq'], v=['yq', 'xh'], h=['yh', 'xh'], q=['yq', 'xq'])
        loc_registry_ver = dict(l='zl', i='zi')
        hloc = loc[0]
        vloc = loc[1]
        vdim = loc_registry_ver[vloc]
        hdims = loc_registry_hor[hloc]
        return tuple(['Time', vdim, *hdims])

    def get_current_location_dimensions(self, loc):
        self._current_hloc = loc[0]
        self._current_vloc = loc[1]
        self._current_dimensions = list(self.get_dimensions_by_location(loc))

    def get_final_location_dimensions(self):
        self._final_hloc = self._final_loc[0]
        self._final_vloc = self._final_loc[1]
        self._final_dimensions = self.get_dimensions_by_location(
            self._final_loc)

    def modify_index(self, axis, startend, value):
        dim = self._final_dimensions[axis]
        axis_indices = list(self.indices[dim])
        axis_indices[startend] += value
        self.indices[dim] = tuple(axis_indices)

    def modify_index_return_self(self, axis, startend, value):
        self.modify_index(axis, startend, value)
        return self

    xsm = partialmethod(modify_index_return_self, 3, 0, -1)
    xep = partialmethod(modify_index_return_self, 3, 1, 1)
    ysm = partialmethod(modify_index_return_self, 2, 0, -1)
    yep = partialmethod(modify_index_return_self, 2, 1, 1)
    zsm = partialmethod(modify_index_return_self, 1, 0, -1)
    zep = partialmethod(modify_index_return_self, 1, 1, 1)

    def get_slice(self):
        self._slice = []
        dims = self._final_dimensions
        current_dims = self._current_dimensions
        for i, dim in enumerate(dims):
            indices = list(self.indices[dim])
            if indices[0] < 0:
                indices[0] = 0
            current_dim = current_dims[i]
            if indices[1] > self.dim_arrays[current_dim].size:
                indices[1] = self.dim_arrays[current_dim].size
            self._slice.append(slice(*indices))
        self._slice = tuple(self._slice)
        return self

    def read(self):
        def lazy_read_and_fill(array):
            array = self._v[self._slice]
            if np.ma.isMaskedArray(array):
                array = array.filled(self._fillvalue)
            return array

        self.operations.append(lazy_read_and_fill)
        self.implement_BC_if_necessary()
        return self

    BoundaryCondition = BoundaryCondition
    _default_bc_type = dict(
        u=['mirror', 'circsymh', 'circsymh', 'circsymh', 'zeros', 'circsymq'],
        v=['mirror', 'circsymh', 'zeros', 'circsymq', 'circsymh', 'circsymh'],
        h=['mirror', 'circsymh', 'mirror', 'mirror', 'mirror', 'mirror'],
        q=['mirror', 'circsymh', 'zeros', 'circsymq', 'zeros', 'circsymq'])

    def implement_BC_if_necessary(self):
        dims = self._current_dimensions
        if self._bc_type is None:
            self._bc_type = self._default_bc_type
        for i, dim in enumerate(dims[1:]):
            indices = self.indices[dim]
            loc = self._current_hloc
            if indices[0] < 0:
                bc_type = self._bc_type[loc][2 * i]
                self.operations.append(
                    self.BoundaryCondition(bc_type, i + 1, 0))
            if indices[1] > self.dim_arrays[dim].size:
                bc_type = self._bc_type[loc][2 * i + 1]
                self.operations.append(
                    self.BoundaryCondition(bc_type, i + 1, -1))

    LazyNumpyOperation = LazyNumpyOperation

    def np_ops(self, npfunc, *args, **kwargs):
        self.operations.append(
            self.LazyNumpyOperation(npfunc, *args, **kwargs))
        return self

    @staticmethod
    def vertical_move(array):
        return 0.5 * (array[:, :-1, :, :] + array[:, 1:, :, :])

    @staticmethod
    def check_possible_movements_for_move(current_loc, new_loc):
        possible_from_to = dict(
            u=['q', 'h'], v=['h', 'q'], h=['v', 'u'], q=['u', 'v'])
        possible_ns = dict(
            u=[0, 0, 0, 1], v=[0, 0, 1, 0], h=[0, 0, 0, 0], q=[0, 0, -1, -1])
        possible_ne = dict(
            u=[0, -1, -1, 0],
            v=[0, -1, 0, -1],
            h=[0, -1, -1, -1],
            q=[0, -1, 0, 0])

        axis = possible_from_to[current_loc].index(new_loc) + 2
        ns = possible_ns[current_loc][axis]
        ne = possible_ne[current_loc][axis]
        return (axis, ns, ne)

    @staticmethod
    def horizontal_move(axis, array):
        return 0.5 * (np.take(array, range(array.shape[axis] - 1), axis=axis) +
                      np.take(array, range(1, array.shape[axis]), axis=axis))

    def adjust_dimensions_and_indices_for_vertical_move(self):
        self.modify_index(1, 1, -1)
        if self._current_vloc == 'l':
            self.modify_index(1, 0, 1)
            self._current_dimensions[1] = 'zi'
        else:
            self._current_dimensions[1] = 'zl'
        self.determine_location()

    def adjust_dimensions_and_indices_for_horizontal_move(self, axis, ns, ne):
        self.modify_index(axis, 0, ns)
        self.modify_index(axis, 1, ne)
        current_dimension = list(self._current_dimensions[axis])
        if current_dimension[1] == 'h':
            current_dimension[1] = 'q'
        elif current_dimension[1] == 'q':
            current_dimension[1] = 'h'
        self._current_dimensions[axis] = "".join(current_dimension)
        self.determine_location()

    def move_to(self, new_loc):
        if new_loc in ['l', 'i'] and new_loc != self._current_vloc:
            self.adjust_dimensions_and_indices_for_vertical_move()
            self.operations.append(self.vertical_move)
        elif new_loc in ['u', 'v', 'h', 'q'] and new_loc != self._current_hloc:
            axis, ns, ne = self.check_possible_movements_for_move(
                self._current_hloc, new_loc)
            self.adjust_dimensions_and_indices_for_horizontal_move(
                axis, ns, ne)
            move = partial(self.horizontal_move, axis)
            self.operations.append(move)
        return self

    @property
    def dimensions(self):
        return tuple(self._current_dimensions)

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
        return self._current_hloc

    @property
    def vloc(self):
        """Vertical location of the variable on the grid.
        l : layer,
        i : interface."""
        return self._current_vloc

    def compute(self):
        for ops in self.operations:
            self.array = ops(self.array)
        self.operations = []
        return self

    @property
    def shape(self):
        return self.array.shape if hasattr(self, 'array') else None

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
