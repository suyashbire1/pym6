from pym6 import Variable, Domain, Plotter
gv = Variable.GridVariable
Initializer = Domain.Initializer
from netCDF4 import Dataset as dset
import numpy as np

def test_location():
    initializer = Initializer(geofil = '/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc',
            vgeofil = '/home/sbire/pym6/pym6/tests/data/Vertical_coordinate.nc',
            fil = '/home/sbire/pym6/pym6/tests/data/output__0001_12_009.nc',
            wlon = -25,
            elon = 0,
            slat = 10,
            nlat = 60)
    dom = Domain.Domain(initializer)
    with dset(initializer.fil) as fh:
        e = gv('e',dom,'hi',fh).read_array(tmean=False).values
        e1 = fh.variables['e'][:]

        u = gv('u',dom,'ul',fh).read_array(tmean=False).values
        u1 = fh.variables['u'][:]

        v = gv('v',dom,'vl',fh).read_array(tmean=False).values
        v1 = fh.variables['v'][:]

        wparam = gv('wparam',dom,'hl',fh).read_array(tmean=False).values
        wparam1 = fh.variables['wparam'][:]

        rv = gv('RV',dom,'ql',fh).read_array(tmean=False).values
        rv1 = fh.variables['RV'][:]

    assert np.all(e == e1)
    assert np.all(u == u1)
    assert np.all(v == v1)
    assert np.all(wparam == wparam1)
    assert np.all(rv == rv1)
