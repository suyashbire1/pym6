from pym6 import Variable, Variable2, Domain
from netCDF4 import Dataset as dset
import numpy as np
import unittest
gv = Variable.GridVariable
gv3 = Variable2.GridVariable2
geom = Variable2.GridGeometry
Initializer = Domain.Initializer


def test_location():
    initializer = Initializer(
        geofil='/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc',
        vgeofil='/home/sbire/pym6/pym6/tests/data/Vertical_coordinate.nc',
        fil='/home/sbire/pym6/pym6/tests/data/output__0001_12_009.nc',
        wlon=-25,
        elon=0,
        slat=10,
        nlat=60)
    dom = Domain.Domain(initializer)
    with dset(initializer.fil) as fh:
        e = gv('e', dom, 'hi', fh).read_array(tmean=False).values
        e1 = fh.variables['e'][:]

        u = gv('u', dom, 'ul', fh).read_array(tmean=False).values
        u1 = fh.variables['u'][:]

        v = gv('v', dom, 'vl', fh).read_array(tmean=False).values
        v1 = fh.variables['v'][:]

        wparam = gv('wparam', dom, 'hl', fh).read_array(tmean=False).values
        wparam1 = fh.variables['wparam'][:]

        rv = gv('RV', dom, 'ql', fh).read_array(tmean=False).values
        rv1 = fh.variables['RV'][:]

    assert np.all(e == e1)
    assert np.all(u == u1)
    assert np.all(v == v1)
    assert np.all(wparam == wparam1)
    assert np.all(rv == rv1)


class test_variable(unittest.TestCase):
    def setUp(self):
        self.south_lat, self.north_lat = 30, 40
        self.west_lon, self.east_lon = -10, -5
        self.fh = dset(
            '/home/sbire/pym6/pym6/tests/data/output__0001_12_009.nc')
        self.geom = geom('/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc')
        self.initializer = dict(
            south_lat=self.south_lat,
            north_lat=self.north_lat,
            west_lon=self.west_lon,
            east_lon=self.east_lon)
        self.vars = ['e', 'u', 'v', 'wparam', 'RV']

    def tearDown(self):
        self.fh.close()

    def test_locations(self):
        hlocs = ['h', 'u', 'v', 'h', 'q']
        vlocs = ['i', 'l', 'l', 'l', 'l']
        for i, var in enumerate(self.vars):
            gvvar = gv3(var, self.fh, **self.initializer)
            self.assertEqual(gvvar.hloc, hlocs[i])
            self.assertEqual(gvvar.vloc, vlocs[i])

    def test_has_indices(self):
        for i, var in enumerate(self.vars):
            gvvar = gv3(var, self.fh, **self.initializer)
            for dim in gvvar.dimensions:
                self.assertIn(dim, gvvar.indices)

    def test_array(self):
        for var in self.vars:
            gvvar = gv3(var, self.fh,
                        **self.initializer).get_slice().read().compute().array
            self.assertIsInstance(gvvar, np.ndarray)

    def test_array_full(self):
        for var in self.vars:
            gvvar = gv3(var, self.fh).get_slice().read().compute().array
            var_array = self.fh.variables[var][:]
            self.assertTrue(np.allclose(gvvar, var_array))

    def test_array_full_fillvalue(self):
        for i, fill in enumerate([np.nan, 0]):
            gvvar = gv3(
                'u', self.fh,
                fillvalue=fill).get_slice().read().compute().array
            if i == 0:
                self.assertTrue(np.all(np.isnan(gvvar[:, :, :, -1])))
            else:
                self.assertTrue(
                    np.all(gvvar[:, :, :, -1] == 0),
                    msg=f'{gvvar[:, :, :, -1]}')

    def test_numpy_func(self):
        for var in self.vars:
            gvvar = gv3(
                var, self.fh, fillvalue=0).get_slice().read().np_ops(
                    np.mean, keepdims=True).compute().array
            var_array = self.fh.variables[var][:]
            if np.ma.isMaskedArray(var_array):
                var_array = var_array.filled(0)
            var_array = np.mean(var_array, keepdims=True)
            self.assertTrue(
                np.allclose(gvvar, var_array), msg=f'{gvvar,var_array}')

    def test_boundary_conditions(self):
        for var in self.vars:
            gvvar = gv3(var, self.fh).xsm().xep().ysm().yep().get_slice().read(
            ).compute().array
            var_array = self.fh.variables[var][:]
            shape1 = gvvar.shape
            shape2 = var_array.shape
            self.assertTrue(shape1[0] == shape2[0])
            self.assertTrue(shape1[1] == shape2[1])
            self.assertTrue(shape1[2] == shape2[2] + 2)
            self.assertTrue(shape1[3] == shape2[3] + 2)

    def test_final_locs(self):
        hlocs = ['h', 'u', 'v', 'q']
        vlocs = ['l', 'i']
        vdims = ['zl', 'zi']
        ydims = ['yh', 'yh', 'yq', 'yq']
        xdims = ['xh', 'xq', 'xh', 'xq']
        for var in self.vars:
            for i, hloc in enumerate(hlocs):
                for j, vloc in enumerate(vlocs):
                    gvvar = gv3(
                        var,
                        self.fh,
                        final_loc=hloc + vloc,
                        **self.initializer)
                    dims = gvvar.final_dimensions
                    self.assertTrue(dims[0] == 'Time')
                    self.assertTrue(dims[1] == vdims[j])
                    self.assertTrue(dims[2] == ydims[i])
                    self.assertTrue(dims[3] == xdims[i])

    def test_modify_indices(self):
        plusminus = [-1, 1]
        for i, dim in enumerate(['z', 'y', 'x']):
            for j, op in enumerate(['sm', 'ep']):
                for var in self.vars:
                    gvvar = gv3(var, self.fh, **self.initializer)
                    a = gvvar.indices[gvvar.final_dimensions[i + 1]]
                    gvvar = getattr(gvvar, dim + op)()
                    b = gvvar.indices[gvvar.final_dimensions[i + 1]]
                    self.assertEqual(a[j] + plusminus[j], b[j])

    def test_diff(self):
        for var in self.vars:
            gvvar = gv3(
                var, self.fh, geometry=self.geom,
                **self.initializer).get_slice().read().compute().array
            self.assertIsInstance(gvvar, np.ndarray)
