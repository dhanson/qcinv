#!/usr/bin/env python

import numpy as np
import pylab as pl

import qcinv

lmax1  = 10
lmax2  = 6

#

talm1 = np.random.standard_normal( qcinv.util_alm.lmax2nlm(lmax1) ) + np.random.standard_normal( qcinv.util_alm.lmax2nlm(lmax1) )*1.j
talm2 = qcinv.util_alm.alm_copy(talm1, lmax=lmax2)

pl.plot( qcinv.util_alm.alm_cl(talm1), label='talm1' )
pl.plot( qcinv.util_alm.alm_cl(talm2), label='talm2' )

pl.axvline(x=lmax2, color='k', linestyle='--')

pl.legend()
pl.show()
