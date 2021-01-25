# -*- coding: utf-8 -*-
"""
Created on Fri Nov 8 20:27:29 2019

@author: Luiz Votto
"""


from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.special import (riccati_jn, riccati_yn, lpmn)

import field_components as fcs

from utils import (
    plane_wave_coefficient, ltaumn, lpimn,
    mie_ans, mie_bns, mie_cns, mie_dns
)

MAX_IT = 1000

class Field(ABC):
    def __init__(self, wave_number, bscs={'TM': {}, 'TE': {}}, f0=1):
        self.tm_bscs = bscs['TM']
        self.te_bscs = bscs['TE']
        self.degrees = set()
        self.wave_number = wave_number
        self.f0 = f0
        for _, m in self.tm_bscs:
            self.degrees.add(m)
    
    @property
    def k(self):
        return self.wave_number

    @property
    def coordinate_system_name(self):
        if self.__class__ in [SphericalElectricField]:
            return "spherical"
        else:
            return "cartesian"
    
    @classmethod
    def max_n(cls, x):
        return int(x + 4.05 * np.power(x, 1 / 3) + 2) if x >= .02 else 2

    def max_it(self, radius):
        x = self.wave_number * radius
        return Field.max_n(x)

    @abstractmethod
    def field_i(self, x1, x2, x3, **kwargs):
        pass

    @abstractmethod
    def field_s(self, x1, x2, x3, **kwargs):
        pass

    @abstractmethod
    def field_sp(self, x1, x2, x3, **kwargs):
        pass

    def field_t(self, x1, x2, x3, **kwargs):
        return sum([
            self.field(x1, x2, x3, which=w, **kwargs) for w in ("i", "s", "sp")
        ])

    def field(self, x1, x2, x3, which="i", **kwargs):
        if which.lower() == "i":
            return self.field_i(x1, x2, x3, **kwargs)
        if which.lower() == "s":
            return self.field_s(x1, x2, x3, **kwargs)
        if which.lower() == "sp":
            return self.field_sp(x1, x2, x3, **kwargs)
        if which.lower() == "t":
            return self.field_t(x1, x2, x3, **kwargs)
        else:
            raise ValueError("I don't recognize field type: " + str(which))

    def norm(self, x1, x2, x3, which="i", **kwargs):
        return np.linalg.norm(self.field(x1, x2, x3, which=which, **kwargs))
    
    @classmethod
    def sph2car_matrix(cls, theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        return np.array([
            [st * cp,   ct * cp,    -sp ],
            [st * sp,   ct * sp,    cp  ],
            [ct,        -st,        0   ]
        ])
    
    @classmethod
    def sph2car(cls, v, theta, phi):
        return np.matmul(Field.sph2car_matrix(theta, phi), v)
    
    @classmethod
    def car2sph(cls, x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        t = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
        p = np.arctan2(y, x)
        return np.array([r, t, p])

class SphericalField(Field):
    def component(self, radial, theta, phi,
                  riccati_terms, legendre_terms, pre_mul, nm_func,
                  max_it, mode="tm", mies=None):
        n, res = 1, 0
        
        if mode.lower() == "tm":
            bscs = self.tm_bscs
        if mode.lower() == "te":
            bscs = self.te_bscs
        if mode.lower() not in ["tm", "te"]:
            raise KeyError(
                "I only recognize modes TM and TE. Received: " + str(mode)
            )

        if mies is None:
            mies = [1 for _ in range(max_it + 1)]

        while n <= max_it:
            for m in self.degrees:
                inc = plane_wave_coefficient(n, self.wave_number) \
                    * bscs[(n, m)] \
                    * riccati_terms[n] \
                    * legendre_terms[np.abs(m)][n] \
                    * np.exp(1j * m * phi) \
                    * nm_func(n, m) \
                    * mies[n]
                res += inc
            n += 1
        return pre_mul * res


    def max_it(self, radius):
        x = self.wave_number * radius
        return int(x + 4.05 * np.power(x, 1 / 3) + 2) if x >= .02 else 2

    def _radial_i(self, radial, theta, phi, radius=None, mode="TM"):
        if radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, self.wave_number * radial)[0]
        legendre_terms = lpmn(max_it, max_it, np.cos(theta))[0]
        nm_func = lambda n, m: n * (n + 1)
        pre_mul = self.f0 / self.wave_number / np.power(radial, 2)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mode=mode
        )
    
    def _theta_tx_i(self, radial, theta, phi, radius=None, mode="TM"):
        if radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, self.wave_number * radial)[1]
        legendre_terms = ltaumn(max_it, max_it, theta)
        nm_func = lambda n, m: 1
        pre_mul = self.f0 / radial

        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mode=mode
        )

    def _theta_ty_i(self, radial, theta, phi, radius=None, mode="TE"):
        if radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, self.wave_number * radial)[0]
        legendre_terms = lpimn(max_it, max_it, theta)
        nm_func = lambda n, m: m
        pre_mul = self.f0 / radial
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mode=mode
        )
    
    def _phi_tx_i(self, radial, theta, phi, radius=None, mode="TM"):
        if radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, self.wave_number * radial)[1]
        legendre_terms = lpimn(max_it, max_it, theta)
        nm_func = lambda n, m: m
        pre_mul = 1j * self.f0 / radial
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mode=mode
        )

    def _phi_ty_i(self, radial, theta, phi, radius=None, mode="TE"):
        if radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, self.wave_number * radial)[0]
        legendre_terms = ltaumn(max_it, max_it, theta)
        nm_func = lambda n, m: 1
        pre_mul = 1j * self.f0 / radial
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mode=mode
        )

    @abstractmethod
    def radial_i(self, *args, **kwargs):
        pass

    @abstractmethod
    def theta_tm_i(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def theta_te_i(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def phi_tm_i(self, *args, **kwargs):
        pass

    @abstractmethod
    def phi_te_i(self, *args, **kwargs):
        pass

    def theta_i(self, radial, theta, phi):
        return self.theta_tm_i(radial, theta, phi) \
             + self.theta_te_i(radial, theta, phi)

    def phi_i(self, radial, theta, phi):
        return self.phi_tm_i(radial, theta, phi) \
             + self.phi_te_i(radial, theta, phi)

    def field_i(self, radial, theta, phi, **kwargs):
        args = [radial, theta, phi]
        return np.array([
            self.radial_i(*args),
            self.theta_i(*args),
            self.phi_i(*args)
        ])
    
    @abstractmethod
    def field_s(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def field_sp(self, *args, **kwargs):
        pass


class SphericalElectricField(SphericalField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e0 = self.f0

    def component(self, radial, theta, phi, 
                  riccati_terms, legendre_terms, pre_mul, nm_func,
                  max_it, mode="tm", mies=None):
        n, res = 1, 0
        
        if mode.lower() == "tm":
            bscs = self.tm_bscs
        if mode.lower() == "te":
            bscs = self.te_bscs
        if mode.lower() not in ["tm", "te"]:
            raise KeyError(
                "I only recognize modes TM and TE. Received: " + str(mode)
            )

        if mies is None:
            mies = [1 for _ in range(max_it + 1)]

        while n <= max_it:
            for m in self.degrees:
                inc = plane_wave_coefficient(n, self.wave_number) \
                    * bscs[(n, m)] \
                    * riccati_terms[n] \
                    * legendre_terms[np.abs(m)][n] \
                    * np.exp(1j * m * phi) \
                    * nm_func(n, m) \
                    * mies[n]
                res += inc
            n += 1
        
        return (pre_mul * res)

    def radial_i(self, radial, theta, phi, radius=None):
        return self._radial_i(radial, theta, phi, radius=radius, mode="TM")
    
    def radial_s(self, radial, theta, phi,
                 radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        if radial <= radius:
            return 0

        riccati_terms = riccati_yn(max_it, self.wave_number * radial)[0]
        legendre_terms = lpmn(max_it, max_it, np.cos(theta))[0]
        nm_func = lambda n, m: n * (n + 1)
        pre_mul = -self.e0 / self.wave_number / np.power(radial, 2)
        mies = mie_ans(max_it, self.wave_number, radius, M, mu, musp)

        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies
        )

    def radial_sp(self, radial, theta, phi,
                  radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, M * self.wave_number * radial)[0]
        legendre_terms = lpmn(max_it, max_it, np.cos(theta))[0]
        nm_func = lambda n, m: n * (n + 1)
        pre_mul = self.e0 / np.power(radial, 2) / M ** 2 / self.wave_number
        mies = mie_cns(max_it, self.wave_number, radius, M, mu, musp)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies
        )
        
    def theta_tm_i(self, radial, theta, phi, radius=None):
        return self._theta_tx_i(radial, theta, phi, radius=radius, mode="TM")

    def theta_te_i(self, radial, theta, phi, radius=None):
        return self._theta_ty_i(radial, theta, phi, radius=radius, mode="TE")

    def theta_i(self, radial, theta, phi):
        return self.theta_tm_i(radial, theta, phi) \
             + self.theta_te_i(radial, theta, phi)
    
    def theta_tm_s(self, radial, theta, phi,
                   radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_yn(max_it, self.wave_number * radial)[1]
        legendre_terms = ltaumn(max_it, max_it, theta)
        nm_func = lambda n, m: 1
        pre_mul = -self.e0 / radial
        mies = mie_ans(max_it, self.wave_number, radius, M, mu, musp)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies
        )

    def theta_te_s(self, radial, theta, phi,
                   radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_yn(max_it, self.wave_number * radial)[0]
        legendre_terms = lpimn(max_it, max_it, theta)
        nm_func = lambda n, m: m
        pre_mul = -self.e0 / radial
        mies = mie_bns(max_it, self.wave_number, radius, M, mu, musp)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mode="te", mies=mies
        )

    def theta_s(self, radial, theta, phi,
                **scatt_params):    
        if radial <= scatt_params["radius"]:
            return 0
        return self.theta_tm_s(radial, theta, phi, **scatt_params) \
             + self.theta_te_s(radial, theta, phi, **scatt_params)
        
    def theta_tm_sp(self, radial, theta, phi,
                    radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, M * self.wave_number * radial)[1]
        legendre_terms = ltaumn(max_it, max_it, theta)
        nm_func = lambda n, m: 1
        pre_mul = self.e0 / radial / M
        mies = mie_cns(max_it, self.wave_number, radius, M, mu, musp)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies
        )
        
    def theta_te_sp(self, radial, theta, phi,
                    radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, M * self.wave_number * radial)[0]
        legendre_terms = lpimn(max_it, max_it, theta)
        nm_func = lambda n, m: m
        pre_mul = self.e0 / radial / M ** 2 * mu / musp
        mies = mie_dns(max_it, self.wave_number, radius, M, mu, musp)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies, mode="te"
        )

    def theta_sp(self, radial, theta, phi, **scatt_params):
        args = [radial, theta, phi]
        return self.theta_tm_sp(*args, **scatt_params) \
             + self.theta_te_sp(*args, **scatt_params)

    def phi_tm_i(self, radial, theta, phi, radius=None):
        return self._phi_tx_i(radial, theta, phi, radius=radius, mode="TM")

    def phi_te_i(self, radial, theta, phi, radius=None):
        return self._phi_ty_i(radial, theta, phi, radius=radius, mode="TE")

    def phi_i(self, radial, theta, phi):
        return self.phi_tm_i(radial, theta, phi) \
             + self.phi_te_i(radial, theta, phi)

    def phi_tm_s(self, radial, theta, phi,
                 radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_yn(max_it, self.wave_number * radial)[1]
        legendre_terms = lpimn(max_it, max_it, theta)
        nm_func = lambda n, m: m
        pre_mul = -1j * self.e0 / radial
        mies = mie_ans(max_it, self.wave_number, radius, M, mu, musp)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies
        )

    def phi_te_s(self, radial, theta, phi,
                 radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_yn(max_it, self.wave_number * radial)[0]
        legendre_terms = ltaumn(max_it, max_it, theta)
        nm_func = lambda n, m: 1
        pre_mul = -1j * self.e0 / radial
        mies = mie_bns(max_it, self.wave_number, radius, M, mu, musp)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies, mode="te"
        )
        
    def phi_s(self, radial, theta, phi,
                **scatt_params):    
        if radial < scatt_params["radius"]:
            return 0
        return self.phi_tm_s(radial, theta, phi, **scatt_params) \
             + self.phi_te_s(radial, theta, phi, **scatt_params)
    
    def phi_tm_sp(self, radial, theta, phi,
                 radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, M * self.wave_number * radial)[1]
        legendre_terms = lpimn(max_it, max_it, theta)
        nm_func = lambda n, m: m
        pre_mul = 1j * self.e0 / radial / M
        mies = mie_cns(max_it, self.wave_number, radius, M, mu, musp)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies
        )

    def phi_te_sp(self, radial, theta, phi,
                 radius=None, M=2, mu=1, musp=1, small=False):
        if not small or radius is None:
            max_it = self.max_it(radial)
        else:
            max_it = self.max_it(radius)

        riccati_terms = riccati_jn(max_it, M * self.wave_number * radial)[0]
        legendre_terms = ltaumn(max_it, max_it, theta)
        nm_func = lambda n, m: 1
        pre_mul = 1j * self.e0 / radial / M ** 2 * musp / mu
        mies = mie_dns(max_it, self.wave_number, radius, M, mu, musp)
        
        return self.component(
            radial, theta, phi,
            riccati_terms, legendre_terms, pre_mul, nm_func,
            max_it, mies=mies, mode="te"
        )
    
    def phi_sp(self, radial, theta, phi, **scatt_params):
        args = [radial, theta, phi]
        return self.phi_tm_sp(*args, **scatt_params) \
             + self.phi_te_sp(*args, **scatt_params)

    def field_i(self, radial, theta, phi, **kwargs):
        args = [radial, theta, phi]
        return np.array([
            self.radial_i(*args),
            self.theta_i(*args),
            self.phi_i(*args)
        ])

    def field_s(self, radial, theta, phi, **scatt_params):
        args = [radial, theta, phi]
        return np.array([
            self.radial_s(*args, **scatt_params),
            self.theta_s(*args, **scatt_params),
            self.phi_s(*args, **scatt_params)
        ])
    
    def field_sp(self, radial, theta, phi, **scatt_params):
        args = [radial, theta, phi]
        return np.array([
            self.radial_sp(*args, **scatt_params),
            self.theta_sp(*args, **scatt_params),
            self.phi_sp(*args, **scatt_params)
        ])

    def cs_sca(self, max_it=None):
        if max_it is None: max_it = MAX_IT
        
        pre = 4 * np.pi / self.wave_number
        res = 0

        for n in range(1, max_it):
            for m in self.degrees:
                inc = (2 * n + 1) / n / (n + 1) \
                    * mp.gammaprod([n + np.abs(m)], [n - np.abs(m)])
                    ######## NEED MIE COEFFS

    
    def plot_r(self, r_max, sample=200):
        rs = np.linspace(-r_max, r_max, sample)
        xs, zs = np.meshgrid(rs, rs)
        r = lambda x, z: np.sqrt(x ** 2 + z ** 2)
        t = lambda x, z: np.arctan2(z, x)
        grid = np.abs(np.vectorize(self.norm)(r(xs, zs), t(xs, zs), 0)) ** 2
        grid = np.nan_to_num(grid)
        rr = r_max * 1E6
        plt.xlabel(r'x [$\mu$m]')
        plt.ylabel(r'z [$\mu$m]')
        plt.imshow(grid.transpose(), extent=[-rr, rr, -rr, rr], cmap='inferno')
        plt.colorbar()
        plt.show()

    def make_cartesian_field(self):
        bscs = {
            "TM": self.tm_bscs,
            "TE": self.te_bscs
        }
        return CartesianField(
            self, self.k, bscs=bscs, f0=self.f0
        )

class SphericalMagneticField(SphericalField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h0 = self.f0

    def radial_i(self, radial, theta, phi, radius=None):
        return self._radial_i(radial, theta, phi, radius=radius, mode="TE")
    
    def theta_te_i(self, radial, theta, phi, radius=None):
        return self._theta_tx_i(radial, theta, phi, radius=radius, mode="TE")

    def theta_tm_i(self, radial, theta, phi, radius=None):
        return -self._theta_ty_i(radial, theta, phi, radius=radius, mode="TM")

    def phi_te_i(self, radial, theta, phi, radius=None):
        return self._phi_tx_i(radial, theta, phi, radius=radius, mode="TE")

    def phi_tm_i(self, radial, theta, phi, radius=None):
        return -self._phi_ty_i(radial, theta, phi, radius=radius, mode="TM")

    def field_s(self):
        pass
    
    def field_sp(self):
        pass
    

class CartesianField(Field):
    def __init__(self, spherical, *args, **kwargs):
        self.spherical = spherical
        super().__init__(spherical.k,*args, **kwargs)
    
    @classmethod
    def cartesian_at_coord(cls, x, y, z, sph_field, **kwargs):
        rtp = Field.car2sph(x, y, z)
        return Field.sph2car(
            sph_field(*rtp, **kwargs),
            np.arctan2(np.sqrt(x ** 2 + y ** 2), z),
            np.arctan2(y, x)
        )
    
    def field_i(self, x, y, z, **kwargs):
        sph_field = self.spherical.field_i
        return CartesianField.cartesian_at_coord(x, y, z, sph_field, **kwargs)
    
    def field_s(self, x, y, z, **kwargs):
        sph_field = self.spherical.field_s
        return CartesianField.cartesian_at_coord(x, y, z, sph_field, **kwargs)
    
    
    def field_sp(self, x, y, z, **kwargs):
        sph_field = self.spherical.field_sp
        return CartesianField.cartesian_at_coord(x, y, z, sph_field, **kwargs)
    