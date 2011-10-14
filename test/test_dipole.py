#!/usr/bin/env python

import healpy as hp
import numpy as np

nside = 16
xyz   = hp.pix2vec(nside, np.arange(0, 12*nside**2))

c, [dx, dy, dz] = hp.fit_dipole(np.ones(12*nside**2))
print 'c' + (' dipole map c, [dx, dy, dz] = %+2.2e, [%+2.2e, %+2.2e, %+2.2e]' % (c, dx, dy, dz))
for i, f in zip( (0, 1, 2), ('x', 'y', 'z') ):
    c, [dx, dy, dz] = hp.fit_dipole(xyz[i])
    print f + (' dipole map c, [dx, dy, dz] = %+2.2e, [%+2.2e, %+2.2e, %+2.2e]' % (c, dx, dy, dz))

alm    = np.zeros(3, dtype=np.complex)
alm[0] = np.random.standard_normal(1)[0]
alm[1] = np.random.standard_normal(1)[0]
alm[2] = np.random.standard_normal(1)[0] + np.random.standard_normal(1)[0]*1.j

tmap   = hp.alm2map(alm, nside)

c, [dx, dy, dz] = hp.fit_dipole(tmap)

alm_c  = +alm[0].real / np.sqrt(4.*np.pi)
alm_dz = +alm[1].real / np.sqrt(4.*np.pi/3.)
alm_dx = -alm[2].real / np.sqrt(2.*np.pi/3.)
alm_dy = +alm[2].imag / np.sqrt(2.*np.pi/3.)

print '    c = ',     c, '     dx = ',     dx, '     dy = ',     dy, '     dz = ',     dz
print 'alm_c = ', alm_c, ' alm_dx = ', alm_dx, ' alm_dy = ', alm_dy, ' alm_dz = ', alm_dz
