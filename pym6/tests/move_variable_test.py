from pym6 import Variable, Variable2, Domain
from netCDF4 import Dataset as dset
import numpy as np
import unittest
import pytest
gv = Variable.GridVariable
gv3 = Variable2.GridVariable2
Initializer = Domain.Initializer


class test_move(unittest.TestCase):
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

    def test_move_u(self):
        ops = ['yep', 'xsm']
        new_loc = ['q', 'h']
        for i, op in enumerate(ops):
            gvvar = getattr(gv3('u', self.fh), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'u')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.return_dimensions()
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(value.size == gvvar.shape[j])
            u = self.fh.variables['u'][:]
            if np.ma.isMaskedArray(u):
                u = u.filled(0)
            if i == 0:
                u = np.concatenate((u, -u[:, :, -1:, :]), axis=2)
                u = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
            elif i == 1:
                u = np.concatenate((np.zeros(u[:, :, :, :1].shape), u), axis=3)
                u = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
            self.assertTrue(np.allclose(u, gvvar))

    def test_move_v(self):
        ops = ['ysm', 'xep']
        new_loc = ['h', 'q']
        for i, op in enumerate(ops):
            gvvar = getattr(gv3('v', self.fh), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'v')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.return_dimensions()
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(value.size == gvvar.shape[j])
            v = self.fh.variables['v'][:]
            if np.ma.isMaskedArray(v):
                v = v.filled(0)
            if i == 0:
                v = np.concatenate((np.zeros(v[:, :, :1, :].shape), v), axis=2)
                v = 0.5 * (v[:, :, :-1] + v[:, :, 1:])
            elif i == 1:
                v = np.concatenate((v, -v[:, :, :, -1:]), axis=3)
                v = 0.5 * (v[:, :, :, :-1] + v[:, :, :, 1:])
            self.assertTrue(np.allclose(v, gvvar))

    def test_move_h(self):
        ops = ['xep', 'yep']
        new_loc = ['u', 'v']
        for i, op in enumerate(ops):
            gvvar = getattr(gv3('wparam', self.fh), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'h')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.return_dimensions()
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(value.size == gvvar.shape[j])
            w = self.fh.variables['wparam'][:]
            if np.ma.isMaskedArray(w):
                w = w.filled(0)
            if i == 0:
                w = np.concatenate((w, w[:, :, :, -1:]), axis=3)
                w = 0.5 * (w[:, :, :, :-1] + w[:, :, :, 1:])
            elif i == 1:
                w = np.concatenate((w, w[:, :, -1:, :]), axis=2)
                w = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
            self.assertTrue(np.allclose(w, gvvar))

    def test_move_q(self):
        ops = ['xsm', 'ysm']
        new_loc = ['v', 'u']
        for i, op in enumerate(ops):
            gvvar = getattr(gv3('RV', self.fh), op)().get_slice().read()
            self.assertTrue(gvvar.hloc == 'q')
            gvvar = gvvar.move_to(new_loc[i])
            self.assertTrue(gvvar.hloc == new_loc[i])
            dims = gvvar.return_dimensions()
            gvvar = gvvar.compute().array
            for j, (key, value) in enumerate(dims.items()):
                self.assertTrue(value.size == gvvar.shape[j])
            rv = self.fh.variables['RV'][:]
            if np.ma.isMaskedArray(rv):
                rv = rv.filled(0)
            if i == 0:
                rv = np.concatenate(
                    (np.zeros(rv[:, :, :, :1].shape), rv), axis=3)
                rv = 0.5 * (rv[:, :, :, :-1] + rv[:, :, :, 1:])
            elif i == 1:
                rv = np.concatenate(
                    (np.zeros(rv[:, :, :1, :].shape), rv), axis=2)
                rv = 0.5 * (rv[:, :, :-1] + rv[:, :, 1:])
            self.assertTrue(np.allclose(rv, gvvar))

    def test_move_vertical_e(self):
        gvvar = gv3('e', self.fh).get_slice().read()
        self.assertTrue(gvvar.vloc == 'i')
        gvvar = gvvar.move_to('l')
        self.assertTrue(gvvar.vloc == 'l')
        dims = gvvar.return_dimensions()
        gvvar = gvvar.compute().array
        for j, (key, value) in enumerate(dims.items()):
            self.assertTrue(value.size == gvvar.shape[j])
        e = self.fh.variables['e'][:]
        e = 0.5 * (e[:, :-1] + e[:, 1:])
        self.assertTrue(np.allclose(e, gvvar))

    def test_move_vertical_u(self):
        gvvar = gv3('u', self.fh).zsm().zep().get_slice().read()
        self.assertTrue(gvvar.vloc == 'l')
        gvvar = gvvar.move_to('i')
        self.assertTrue(gvvar.vloc == 'i')
        dims = gvvar.return_dimensions()
        gvvar = gvvar.compute().array
        for j, (key, value) in enumerate(dims.items()):
            self.assertTrue(value.size == gvvar.shape[j])
        u = self.fh.variables['u'][:]
        if np.ma.isMaskedArray(u):
            u = u.filled(0)
        u = np.concatenate((u[:, :1, :, :], u, -u[:, -1:, :, :]), axis=1)
        u = 0.5 * (u[:, :-1] + u[:, 1:])
        self.assertTrue(np.allclose(u, gvvar))

    def test_move_u_twice(self):
        gvvar = gv3('u', self.fh).xsm().yep().get_slice().read()
        self.assertTrue(gvvar.hloc == 'u')
        gvvar = gvvar.move_to('h')
        self.assertTrue(gvvar.hloc == 'h')
        gvvar = gvvar.move_to('v')
        self.assertTrue(gvvar.hloc == 'v')
        dims = gvvar.return_dimensions()
        gvvar = gvvar.compute().array
        for j, (key, value) in enumerate(dims.items()):
            self.assertTrue(value.size == gvvar.shape[j])
        u = self.fh.variables['u'][:]
        if np.ma.isMaskedArray(u):
            u = u.filled(0)
        u = np.concatenate((np.zeros(u[:, :, :, :1].shape), u), axis=3)
        u = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
        u = np.concatenate((u, -u[:, :, -1:, :]), axis=2)
        u = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
        self.assertTrue(np.allclose(u, gvvar))
