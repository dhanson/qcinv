#!/usr/bin/env python

import numpy as np
import pylab as pl

import qcinv

lmax1  = 10
lmax2  = 6

#

talm1 = np.random.standard_normal( qcinv.util.lmax2nlm(lmax1) ) + np.random.standard_normal( qcinv.util.lmax2nlm(lmax1) )*1.j
talm2 = qcinv.util.alm_copy(talm1, lmax=lmax2)

pl.plot( qcinv.spectra.alm_cl(talm1), label='talm1' )
pl.plot( qcinv.spectra.alm_cl(talm2), label='talm2' )

pl.axvline(x=lmax2)

pl.legend()
pl.show()
