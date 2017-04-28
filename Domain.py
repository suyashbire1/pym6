import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as dset, MFDataset as mfdset

class Grid:

    """ A class to hold grid metrics from MOM6."""

    def __init__(self, geofil, vgeofil):
        """ geofil:  geometry file
            vgeofil: vertical geometry file"""
        with dset(geofil) as _fhg, dset(vgeofil) as _fhgv:
            for var in _fhg.variables:
                setattr(self,var,_fhg.variables[var][:])
            for var in _fhgv.variables:
                setattr(self,var,_fhgv.variables[var][:])
        self.R_earth = 6.378e6
        self.Rho0 = self.R[-1]

    def __repr__(self):
        print('Grid metrics found: {}'.format(vars(self).keys()))
#        for key in vars(self):
#            print(key)
        return "MOM6 grid object."

class Domain(Grid):
    """This class is supposed to be a subset of grid."""
    def __init__(self,geofil,vgeofil,
                      wlon,elon,slat,nlat,
                      ls,le,ts,te):
        super(Domain, self).__init__(geofil,vgeofil)
        self.wlon = wlon
        self.elon = elon
        self.slat = slat
        self.nlat = nlat
        self.ls = ls
        self.le = le
        self.ts = ts
        self.te = te
        self.populateslices()

    @staticmethod
    def findlims(dim,start,end):
        """Finds the extreme indices of the domain."""
        usefulindx = np.nonzero((dim >= start) & (dim <= end))[0]
        lims = usefulindx[0], usefulindx[-1]+1
        return lims

    @staticmethod
    def extendslice(sl,dims,n):
        """Extends the dimension, dim, of slice, sl, by n."""
        _d = dict(ts = sl[0].start,te = sl[0].stop,
                  ls = sl[1].start,le = sl[1].stop,
                  ys = sl[2].start,ye = sl[2].stop,
                  xs = sl[3].start,xe = sl[3].stop)

        for i, dim in enumerate(dims):
            _d[dim] = _d[dim]+n[i]

        ts, te = _d['ts'],_d['te']
        ls, le = _d['ls'],_d['le']
        ys, ye = _d['ys'],_d['ye']
        xs, xe = _d['xs'],_d['xe']
        sl = np.s_[ts:te,ls:le,ys:ye,xs:xe]
        return sl

    def extendslices(self,sl):
        """Extends slices in x and y directions. Useful for differentiation."""
        slices = ['my','py','mx','px','mpy','mpx']
        exdims = [['ys'],['ye'],['xs'],['xe'],['ys','ye'],['xs','xe']]
        n = [[-1],[1],[-1],[1],[-1,1],[-1,1]]
        for i, slc in enumerate(slices):
            setattr(self,'sl'+slc,self.extendslice(sl,exdims[i],n[i]))

    def populateslices(self):
        """Creates slices from the wlon,elon,slat,nlat,ls,le,ts,te inputs"""
        ls = self.ls
        le = self.le
        ts = self.ts
        te = self.te
        gridloc = ['u','v','h','q']  # Four vertices of an Arakawa C-grid cell
        xloc = ['q','h','h','q']     # X Location on the grid for variable in gridloc
        yloc = ['h','q','h','q']     # Y Location on the grid for variable in gridloc
        for i, loc in enumerate(gridloc):
            x = getattr(self,'lon'+xloc[i])
            y = getattr(self,'lat'+yloc[i])
            xs, xe = self.findlims(x,self.wlon,self.elon)
            ys, ye = self.findlims(y,self.slat,self.nlat)
            sl = np.s_[ts:te, ls:le, ys:ye, xs:xe]
            setattr(self,'sl',sl)
            self.extendslices(sl)
