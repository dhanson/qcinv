#!/usr/bin/env python

import numpy as np
import pylab as pl

import qcinv

lmax   = 10
lsplit = 5

#

talm_lo = np.random.standard_normal( qcinv.util.lmax2nlm(lmax) ) + np.random.standard_normal( qcinv.util.lmax2nlm(lmax) )*1.j
talm_hi = np.random.standard_normal( qcinv.util.lmax2nlm(lmax) ) + np.random.standard_normal( qcinv.util.lmax2nlm(lmax) )*1.j

pl.plot( qcinv.spectra.alm_cl(talm_lo), label='lo' )
pl.plot( qcinv.spectra.alm_cl(talm_hi), label='hi' )

pl.plot( qcinv.spectra.alm_cl( qcinv.util.alm_splice(talm_lo, talm_hi, lsplit) ), linestyle='--' )

pl.legend()
pl.show()
