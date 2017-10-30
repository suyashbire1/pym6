from pym6 import Variable, Variable2, Domain, Plotter
gv = Variable.GridVariable
gv3 = Variable2.GridVariable2
Initializer = Domain.Initializer
from netCDF4 import Dataset as dset
import numpy as np
import unittest
import pytest


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


class test_domain(unittest.TestCase):
    def setUp(self):
        self.slat, self.nlat = 30, 40
        self.wlon, self.elon = -10, -5
        self.fh = dset(
            '/home/sbire/pym6/pym6/tests/data/output__0001_12_009.nc')
        self.initializer = dict(
            fh=self.fh,
            south_lat=self.slat,
            north_lat=self.nlat,
            west_lon=self.wlon,
            east_lon=self.elon)

    def tearDown(self):
        self.fh.close()

    def test_meridional_domain(self):
        for loc in ['h', 'q']:
            var = 'y' + loc
            lat = self.fh.variables[var][:]
            lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
            a, b, c = Variable2.MeridionalDomain(
                **self.initializer).indices['y' + loc]
            self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

    def test_zonal_domain(self):
        for loc in ['h', 'q']:
            var = 'x' + loc
            lon = self.fh.variables[var][:]
            lon_restricted = lon[(lon >= self.wlon) & (lon <= self.elon)]
            a, b, c = Variable2.ZonalDomain(**self.initializer).indices['x'
                                                                        + loc]
            self.assertTrue(np.allclose(lon_restricted, lon[a:b:c]))

    def test_stride_meridional_domain(self):
        for stride in range(2, 4):
            self.initializer['stridey'] = stride
            for loc in ['h', 'q']:
                var = 'y' + loc
                lat = self.fh.variables[var][:]
                lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
                lat_restricted = lat_restricted[::stride]
                a, b, c = Variable2.MeridionalDomain(
                    **self.initializer).indices['y' + loc]
                self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

    def test_stride_zonal_domain(self):
        for stride in range(2, 4):
            self.initializer['stridex'] = stride
            for loc in ['h', 'q']:
                var = 'x' + loc
                lon = self.fh.variables[var][:]
                lon_restricted = lon[(lon >= self.wlon) & (lon <= self.elon)]
                lon_restricted = lon_restricted[::stride]
                a, b, c = Variable2.ZonalDomain(
                    **self.initializer).indices['x' + loc]
                self.assertTrue(np.allclose(lon_restricted, lon[a:b:c]))

    def test_horizontal_domain(self):
        hdomain = Variable2.HorizontalDomain(**self.initializer)
        self.initializer['stridex'] = 1
        self.initializer['stridey'] = 1
        for loc in ['h', 'q']:
            var = 'y' + loc
            lat = self.fh.variables[var][:]
            lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
            a, b, c = hdomain.indices['y' + loc]
            self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

            var = 'x' + loc
            lon = self.fh.variables[var][:]
            lon_restricted = lon[(lon >= self.wlon) & (lon <= self.elon)]
            a, b, c = hdomain.indices['x' + loc]
            self.assertTrue(np.allclose(lon_restricted, lon[a:b:c]))

    def test_vertical_domain(self):
        zl = self.fh.variables['zl'][:]
        zi = self.fh.variables['zi'][:]
        a, b, c = Variable2.VerticalDomain(**self.initializer).indices['zl']
        self.assertEqual(a, 0)
        self.assertEqual(b, len(zl))
        self.assertEqual(c, 1)
        self.assertTrue(np.allclose(zl, zl[a:b:c]))
        a, b, c = Variable2.VerticalDomain(**self.initializer).indices['zi']
        self.assertEqual(a, 0)
        self.assertEqual(b, len(zi))
        self.assertEqual(c, 1)
        self.assertTrue(np.allclose(zi, zi[a:b:c]))

    def test_temporal_domain(self):
        Time = self.fh.variables['Time'][:]
        a, b, c = Variable2.TemporalDomain(**self.initializer).indices['Time']
        self.assertEqual(a, 0)
        self.assertEqual(b, len(Time))
        self.assertEqual(c, 1)
        self.assertTrue(np.allclose(Time, Time[a:b:c]))

    def test_domain(self):
        zl = self.fh.variables['zl'][:]
        zi = self.fh.variables['zi'][:]
        Time = self.fh.variables['Time'][:]
        domain = Variable2.Domain(**self.initializer)
        self.assertEqual(domain.indices['Time'], (0, len(Time), 1))
        self.assertEqual(domain.indices['zl'], (0, len(zl), 1))
        self.assertEqual(domain.indices['zi'], (0, len(zi), 1))
        for loc in ['h', 'q']:
            var = 'y' + loc
            lat = self.fh.variables[var][:]
            lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
            a, b, c = domain.indices[var]
            self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

            var = 'x' + loc
            lon = self.fh.variables[var][:]
            lon_restricted = lon[(lon >= self.wlon) & (lon <= self.elon)]
            a, b, c = domain.indices[var]
            self.assertTrue(np.allclose(lon_restricted, lon[a:b:c]))


class test_variable(unittest.TestCase):
    def setUp(self):
        self.south_lat, self.north_lat = 30, 40
        self.west_lon, self.east_lon = -10, -5
        self.fh = dset(
            '/home/sbire/pym6/pym6/tests/data/output__0001_12_009.nc')
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


@pytest.fixture(params=range(4))
def axis(request):
    return request.param


@pytest.fixture(params=[0, -1])
def start_or_end(request):
    return request.param


@pytest.fixture(params=['zeros', 'mirror', 'circsymh', 'circsymq'])
def bc_type(request):
    return request.param


def test_create_halo(bc_type, axis, start_or_end):
    dummy_array = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    dummy_BC = Variable2.BoundaryCondition(bc_type, axis, start_or_end)
    dummy_BC.create_halo(dummy_array)
    if dummy_BC.bc_type == 'circsymq':
        take_index = 1 if start_or_end == 0 else -2
        compare_array = dummy_array.take([take_index], axis=axis)
    elif dummy_BC.bc_type == 'zeros':
        compare_array = np.zeros(
            dummy_array.take([start_or_end], axis=axis).shape)
    else:
        compare_array = dummy_array.take([start_or_end], axis=axis)
    assert np.all(dummy_BC.halo == compare_array)


def test_dummy_BC_append_halo(bc_type, axis, start_or_end):
    dummy_array = np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    dummy_BC = Variable2.BoundaryCondition(bc_type, axis, start_or_end)
    dummy_BC.create_halo(dummy_array)
    array = dummy_BC.append_halo_to_array(dummy_array)
    if start_or_end == 0:
        if bc_type == 'circsymq':
            array1 = -dummy_array.take([1], axis=axis)
        elif bc_type == 'circsymh':
            array1 = -dummy_array.take([start_or_end], axis=axis)
        elif dummy_BC.bc_type == 'zeros':
            array1 = np.zeros(
                dummy_array.take([start_or_end], axis=axis).shape)
        else:
            array1 = dummy_array.take([start_or_end], axis=axis)
        array2 = dummy_array
    elif start_or_end == -1:
        array1 = dummy_array
        if bc_type == 'circsymq':
            array2 = -dummy_array.take([-2], axis=axis)
        elif bc_type == 'circsymh':
            array2 = -dummy_array.take([start_or_end], axis=axis)
        elif dummy_BC.bc_type == 'zeros':
            array2 = np.zeros(
                dummy_array.take([start_or_end], axis=axis).shape)
        else:
            array2 = dummy_array.take([start_or_end], axis=axis)
    dummy_array = np.concatenate((array1, array2), axis=axis)
    assert np.all(array == dummy_array)


if __name__ == '__main__':
    unittest.main()
