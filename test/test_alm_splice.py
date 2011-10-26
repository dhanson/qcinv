#!/usr/bin/env python

import numpy as np
import pylab as pl

import qcinv

lmax   = 10
lsplit = 5

#

talm_lo = np.random.standard_normal( qcinv.util_alm.lmax2nlm(lmax) ) + np.random.standard_normal( qcinv.util_alm.lmax2nlm(lmax) )*1.j
talm_hi = np.random.standard_normal( qcinv.util_alm.lmax2nlm(lmax) ) + np.random.standard_normal( qcinv.util_alm.lmax2nlm(lmax) )*1.j

pl.plot( qcinv.util_alm.alm_cl(talm_lo), label='lo' )
pl.plot( qcinv.util_alm.alm_cl(talm_hi), label='hi' )

pl.plot( qcinv.util_alm.alm_cl( qcinv.util_alm.alm_splice(talm_lo, talm_hi, lsplit) ), linestyle='--' )

pl.axvline(x=lsplit, color='k', linestyle='--')

pl.legend()
pl.show()
