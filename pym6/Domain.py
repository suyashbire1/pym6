import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dset, MFDataset as mfdset

class Initializer():
    """A class to hold names of files, domain limits, etc"""

    def __init__(self,**kwargs):
        """ geofil:  geometry file
            vgeofil: vertical geometry file"""
        self.geofil = kwargs.get('geofil','ocean_geometry.nc')
        self.vgeofil = kwargs.get('vgeofil','Vertical_coordinate.nc')

        self.fil = kwargs.get('fil')
        self.fil2 = kwargs.get('fil2')
        self.fil3 = kwargs.get('fil3')
        self.wlon = kwargs.get('wlon')
        self.elon = kwargs.get('elon')
        self.slat = kwargs.get('slat')
        self.nlat = kwargs.get('nlat')
        self.ls = kwargs.get('ls',0)
        self.le = kwargs.get('le',None)
        self.ts = kwargs.get('ts',0)
        self.te = kwargs.get('te',None)
        self.stride_x = kwargs.get('stride_x',1)
        self.stride_y = kwargs.get('stride_y',1)
        self.stride_l = kwargs.get('stride_l',1)
        self.stride_t = kwargs.get('stride_t',1)
        self.meanax = kwargs.get('meanax',(0,2))

        for key, value in kwargs.items():
            if not hasattr(self,key):
                setattr(self,key,value)

class Grid():

    """ A class to hold grid metrics from MOM6."""

    def __init__(self, geofil, vgeofil):
        """ geofil:  geometry file
            vgeofil: vertical geometry file"""
        with dset(geofil) as _fhg, dset(vgeofil) as _fhgv:
            for var in _fhg.variables:
                setattr(self,var,_fhg.variables[var][:])
            for var in _fhgv.variables:
                setattr(self,var,_fhgv.variables[var][:])

        self.g = 9.8
        self.Rho0 = self.R[-1]
        b = self.g/self.Rho0*(self.Layer)
        self.db = (b[0]-b[1])
        self.R_earth = 6.378e6
        self.total_xlen = self.lonh.size
        self.total_ylen = self.lath.size
        self.total_zlen = self.Layer.size+1

    def __repr__(self):
        print('Grid metrics found: {}'.format(vars(self).keys()))
        return "MOM6 grid object."

class Domain2(Grid):
    """This class is supposed to be a subset of grid."""
    def __init__(self,geofil,vgeofil,
                      wlon,elon,slat,nlat,
                      **kwargs):
        super(Domain2, self).__init__(geofil,vgeofil)
        self.wlon = wlon
        self.elon = elon
        self.slat = slat
        self.nlat = nlat
        self.ls = kwargs.get('ls',0)
        self.le = kwargs.get('le',None)
        self.ts = kwargs.get('ts',0)
        self.te = kwargs.get('te',None)
        self.stride_x = kwargs.get('stride_x',1)
        self.stride_y = kwargs.get('stride_y',1)
        self.stride_l = kwargs.get('stride_l',1)
        self.stride_t = kwargs.get('stride_t',1)
        self.slices = {}
        self.populate_slices()

    @staticmethod
    def find_lims(dim,start,end):
        """Finds the extreme indices of the domain."""
        usefulindx = np.nonzero((dim >= start) & (dim <= end))[0]
        lims = usefulindx[0], usefulindx[-1]+1
        return lims

    def populate_slices(self):
        """Creates slices from the wlon,elon,slat,nlat,ls,le,ts,te inputs"""
        ls = self.ls
        le = self.le
        ts = self.ts
        te = self.te
        stride_x = self.stride_x
        stride_y = self.stride_y
        stride_l = self.stride_l
        stride_t = self.stride_t
        gridloc = ['u','v','h','q']  # Four vertices of an Arakawa C-grid cell
        xloc = ['q','h','h','q']     # X Location on the grid for variable in gridloc
        yloc = ['h','q','h','q']     # Y Location on the grid for variable in gridloc
        for i, loc in enumerate(gridloc):
            x = getattr(self,'lon'+xloc[i])
            y = getattr(self,'lat'+yloc[i])
            xs, xe = self.find_lims(x,self.wlon,self.elon)
            ys, ye = self.find_lims(y,self.slat,self.nlat)
            slice_array = np.array([[ts,te,stride_t],
                                    [ls,le,stride_l],
                                    [ys,ye,stride_y],
                                    [xs,xe,stride_x]])
            self.slices[gridloc[i]] = slice_array

class Domain(Grid):
    """This class is supposed to be a subset of grid."""
    def __init__(self,initializer):
        super(Domain, self).__init__(initializer.geofil,
                                     initializer.vgeofil)
        for key,value in initializer.__dict__.items():
            setattr(self,key,value)
        self.slices = {}
        self.populate_slices()

    @staticmethod
    def find_lims(dim,start,end):
        """Finds the extreme indices of the domain."""
        usefulindx = np.nonzero((dim >= start) & (dim <= end))[0]
        lims = usefulindx[0], usefulindx[-1]+1
        return lims

    def populate_slices(self):
        """Creates slices from the wlon,elon,slat,nlat,ls,le,ts,te inputs"""
        ls = self.ls
        le = self.le
        ts = self.ts
        te = self.te
        stride_x = self.stride_x
        stride_y = self.stride_y
        stride_l = self.stride_l
        stride_t = self.stride_t
        gridloc = ['u','v','h','q']  # Four vertices of an Arakawa C-grid cell
        xloc = ['q','h','h','q']     # X Location on the grid for variable in gridloc
        yloc = ['h','q','h','q']     # Y Location on the grid for variable in gridloc
        for i, loc in enumerate(gridloc):
            x = getattr(self,'lon'+xloc[i])
            y = getattr(self,'lat'+yloc[i])
            xs, xe = self.find_lims(x,self.wlon,self.elon)
            ys, ye = self.find_lims(y,self.slat,self.nlat)
            slice_array = np.array([[ts,te,stride_t],
                                    [ls,le,stride_l],
                                    [ys,ye,stride_y],
                                    [xs,xe,stride_x]])
            self.slices[gridloc[i]] = slice_array
