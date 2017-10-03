import numpy as np
from netCDF4 import Dataset as dset, MFDataset as mfdset
from functools import partial
import copy
from .Plotter import plotter, rhotoz


def find_index_limits(dimension, start, end):
    """Finds the extreme indices of the any given dimension of the domain."""
    useful_index = np.nonzero((dimension >= start) & (dimension <= end))[0]
    lims = useful_index[0], useful_index[-1] + 1
    return lims


class MeridionalDomain():
    def __init__(self, fhgeo, south_lat, north_lat, stride=1):
        """Initializes meridional domain limits."""
        lath = fhgeo.variables['lath'][:]
        latq = fhgeo.variables['latq'][:]
        self.indices = {}
        self.indices['h'] = *find_index_limits(lath, south_lat,
                                               north_lat), stride
        self.indices['q'] = *find_index_limits(latq, south_lat,
                                               north_lat), stride


class ZonalDomain():
    def __init__(self, fhgeo, west_lon, east_lon, stride=1):
        """Initializes zonal domain limits."""
        lonh = fhgeo.variables['lonh'][:]
        lonq = fhgeo.variables['lonq'][:]
        self.indices = {}
        self.indices['h'] = *find_index_limits(lonh, west_lon,
                                               east_lon), stride
        self.indices['q'] = *find_index_limits(lonq, west_lon,
                                               east_lon), stride


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
