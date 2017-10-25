from pym6 import Variable, Variable2, Domain, Plotter
gv = Variable.GridVariable
gv2 = Variable2.GridVariable
Initializer = Domain.Initializer
from netCDF4 import Dataset as dset
import numpy as np


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


def test_meridional_domain():
    slat, nlat = 30, 40
    with dset('/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc') as fh:
        for loc in ['h', 'q']:
            var = 'lat' + loc
            lat = fh.variables[var][:]
            lat_restricted = lat[(lat >= slat) & (lat <= nlat)]
            a, b, c = Variable2.MeridionalDomain(fh, slat, nlat,
                                                 1).indices['y' + loc]
            assert np.allclose(lat_restricted, lat[a:b:c])


def test_zonal_domain():
    wlon, elon = -10, -5
    with dset('/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc') as fh:
        for loc in ['h', 'q']:
            var = 'lon' + loc
            lon = fh.variables[var][:]
            lon_restricted = lon[(lon >= wlon) & (lon <= elon)]
            a, b, c = Variable2.ZonalDomain(fh, wlon, elon, 1).indices['x'
                                                                       + loc]
            assert np.allclose(lon_restricted, lon[a:b:c])


def test_stride_meridional_domain():
    slat, nlat = 30, 40
    with dset('/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc') as fh:
        for stride in range(2, 4):
            for loc in ['h', 'q']:
                var = 'lat' + loc
                lat = fh.variables[var][:]
                lat_restricted = lat[(lat >= slat) & (lat <= nlat)]
                lat_restricted = lat_restricted[::stride]
                a, b, c = Variable2.MeridionalDomain(fh, slat, nlat,
                                                     stride).indices['y' + loc]
                assert np.allclose(lat_restricted, lat[a:b:c])


def test_stride_zonal_domain():
    wlon, elon = -10, -5
    with dset('/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc') as fh:
        for stride in range(2, 4):
            for loc in ['h', 'q']:
                var = 'lon' + loc
                lon = fh.variables[var][:]
                lon_restricted = lon[(lon >= wlon) & (lon <= elon)]
                lon_restricted = lon_restricted[::stride]
                a, b, c = Variable2.ZonalDomain(fh, wlon, elon,
                                                stride).indices['x' + loc]
                assert np.allclose(lon_restricted, lon[a:b:c])


def test_horizontal_domain():
    slat, nlat = 30, 40
    wlon, elon = -10, -5
    with dset('/home/sbire/pym6/pym6/tests/data/ocean_geometry.nc') as fh:
        initializer = dict(
            fhgeo=fh,
            south_lat=slat,
            north_lat=nlat,
            west_lon=wlon,
            east_lon=elon)
        hdomain = Variable2.HorizontalDomain(initializer)
        for loc in ['h', 'q']:
            var = 'lat' + loc
            lat = fh.variables[var][:]
            lat_restricted = lat[(lat >= slat) & (lat <= nlat)]
            a, b, c = hdomain.indices['y' + loc]
            assert np.allclose(lat_restricted, lat[a:b:c])

            var = 'lon' + loc
            lon = fh.variables[var][:]
            lon_restricted = lon[(lon >= wlon) & (lon <= elon)]
            a, b, c = hdomain.indices['x' + loc]
            assert np.allclose(lon_restricted, lon[a:b:c])
