from pym6 import Variable, Variable2, Domain, Plotter
gv = Variable.GridVariable
gv2 = Variable2.GridVariable
gv3 = Variable2.GridVariable2
Initializer = Domain.Initializer
from netCDF4 import Dataset as dset
import numpy as np
import unittest


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


def test_get_location():
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
        e = gv2('e', dom, fh)  #.read_array().values
        assert e.hloc == 'h'
        assert e.vloc == 'i'
        u = gv2('u', dom, fh)  #.read_array().values
        assert u.hloc == 'u'
        assert u.vloc == 'l'
        v = gv2('v', dom, fh)  #.read_array().values
        assert v.hloc == 'v'
        assert v.vloc == 'l'
        wparam = gv2('wparam', dom, fh)  #.read_array().values
        assert wparam.hloc == 'h'
        assert wparam.vloc == 'l'
        rv = gv2('RV', dom, fh)  #.read_array().values
        assert rv.hloc == 'q'
        assert rv.vloc == 'l'


class test_domain(unittest.TestCase):
    def setUp(self):
        self.slat, self.nlat = 30, 40
        self.wlon, self.elon = -10, -5
        self.fhgeo = dset('/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc')
        self.fh = dset(
            '/home/sbire/pym6/pym6/tests/data/output__0001_12_009.nc')
        self.initializer = dict(
            fhgeo=self.fhgeo,
            fh=self.fh,
            south_lat=self.slat,
            north_lat=self.nlat,
            west_lon=self.wlon,
            east_lon=self.elon)

    def tearDown(self):
        self.fhgeo.close()
        self.fh.close()

    def test_meridional_domain(self):
        for loc in ['h', 'q']:
            var = 'lat' + loc
            lat = self.fhgeo.variables[var][:]
            lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
            a, b, c = Variable2.MeridionalDomain(
                **self.initializer).indices['y' + loc]
            self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

    def test_zonal_domain(self):
        for loc in ['h', 'q']:
            var = 'lon' + loc
            lon = self.fhgeo.variables[var][:]
            lon_restricted = lon[(lon >= self.wlon) & (lon <= self.elon)]
            a, b, c = Variable2.ZonalDomain(**self.initializer).indices['x'
                                                                        + loc]
            self.assertTrue(np.allclose(lon_restricted, lon[a:b:c]))

    def test_stride_meridional_domain(self):
        for stride in range(2, 4):
            self.initializer['stridey'] = stride
            for loc in ['h', 'q']:
                var = 'lat' + loc
                lat = self.fhgeo.variables[var][:]
                lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
                lat_restricted = lat_restricted[::stride]
                a, b, c = Variable2.MeridionalDomain(
                    **self.initializer).indices['y' + loc]
                self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

    def test_stride_zonal_domain(self):
        for stride in range(2, 4):
            self.initializer['stridex'] = stride
            for loc in ['h', 'q']:
                var = 'lon' + loc
                lon = self.fhgeo.variables[var][:]
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
            var = 'lat' + loc
            lat = self.fhgeo.variables[var][:]
            lat_restricted = lat[(lat >= self.slat) & (lat <= self.nlat)]
            a, b, c = hdomain.indices['y' + loc]
            self.assertTrue(np.allclose(lat_restricted, lat[a:b:c]))

            var = 'lon' + loc
            lon = self.fhgeo.variables[var][:]
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
        self.fhgeo = dset('/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc')
        self.fh = dset(
            '/home/sbire/pym6/pym6/tests/data/output__0001_12_009.nc')
        self.initializer = dict(
            fhgeo=self.fhgeo,
            south_lat=self.south_lat,
            north_lat=self.north_lat,
            west_lon=self.west_lon,
            east_lon=self.east_lon)
        self.vars = ['e', 'u', 'v', 'wparam', 'RV']

    def tearDown(self):
        self.fh.close()
        self.fhgeo.close()

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
                        **self.initializer).get_slice().read().array
            self.assertIsInstance(gvvar, np.ndarray)

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
