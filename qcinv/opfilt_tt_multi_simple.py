# opfilt_tt_multi_simple
#
# operations and filters for temperature only c^-1
# with several input maps, each with perfectly
# correlated signal, and perfectly uncorrelated
# inhomogeneous noise, with independent levels.
#

import os, hashlib
import numpy  as np
import healpy as hp
import pickle as pk

import util
import util_alm
import template_removal

import opfilt_tt
from opfilt_tt import apply_fini, dot_op

def calc_prep(maps, s_cls, n_inv_filts):
    alm = opfilt_tt.calc_prep( maps[0], s_cls, n_inv_filts[0] )
    for m, n_inv_filt in zip(maps[1:], n_inv_filts[1:]):
        alm += opfilt_tt.calc_prep( m, s_cls, n_inv_filt )
    return alm

class fwd_op():
    def __init__(self, s_cls, n_inv_filts):
        cltt = s_cls.cltt[:]
        self.cltt_inv = np.zeros(len(cltt))
        self.cltt_inv[np.where(cltt != 0)] = 1.0/cltt[np.where(cltt != 0)]

        self.n_inv_filts = n_inv_filts

    def hashdict(self):
        return { 'cltt_inv'    : hashlib.sha1( self.cltt_inv.view(np.uint8) ).hexdigest(),
                 'n_inv_filts' : self.n_inv_filts.hashdict() }

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        if ( np.all(talm == 0) ): # do nothing if zero
            return talm

        ret = hp.almxfl(talm, self.cltt_inv)
        for n_inv_filt in self.n_inv_filts:
            alm = np.copy(talm)
            n_inv_filt.apply_alm(alm)
            ret += alm

        return ret

class pre_op_diag():
    def __init__(self, s_cls, n_inv_filts):
        cltt = s_cls.cltt[:]

        lmax = len(n_inv_filts[0].b_transf)-1
        assert( lmax <= (len(cltt)-1) )
        
        filt = np.zeros(lmax+1)
        filt[ np.where(cltt[0:lmax+1] != 0) ] += 1.0/cltt[np.where(s_cls.cltt[0:lmax+1] != 0)]

        for n_inv_filt in n_inv_filts:
            assert( len(cltt) >= len(n_inv_filt.b_transf) )
        
            n_inv_cl = np.sum( n_inv_filt.n_inv ) / (4.0*np.pi)

            tlmax = len(n_inv_filt.b_transf)-1
            assert( tlmax == lmax )

            filt[ np.where(n_inv_filt.b_transf[0:lmax+1] != 0) ] += n_inv_cl * n_inv_filt.b_transf[np.where(n_inv_filt.b_transf[0:lmax+1] != 0)]**2

        filt[np.where(filt != 0)] = 1.0/filt[np.where(filt != 0)]
        self.filt = filt

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return hp.almxfl(talm, self.filt)

class pre_op_dense():
    def __init__(self, lmax, fwd_op, cache_fname=None):
        # construct a low-l, low-nside dense preconditioner by brute force.
        # order of operations is O(nside**2 lmax**3) ~ O(lmax**5), so doing
        # by brute force is still comparable to matrix inversion, with
        # benefit of being very simple to implement.

        if (cache_fname != None) and (os.path.exists(cache_fname)):
            [cache_lmax, cache_hashdict, cache_minv] = pk.load( open(cache_fname, 'r') )
            self.minv = cache_minv

            if ( (lmax != cache_lmax) or (self.hashdict(lmax, fwd_op) != cache_hashdict) ):
                print "WARNING: PRE_OP_DENSE CACHE: hashcheck failed. recomputing."
                os.remove(cache_fname)
                self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)
        else:
            self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)

    def compute_minv(self, lmax, fwd_op, cache_fname=None):
        if cache_fname != None:
            assert(not os.path.exists(cache_fname))

        nrlm = (lmax+1)**2
        trlm = np.zeros( nrlm )
        tmat = np.zeros( ( nrlm, nrlm ) )

        ntmpl = 0
        for n_inv_filt in fwd_op.n_inv_filts:
            tntmpl = 0.0
            for t in n_inv_filt.templates:
                tntmpl += t.nmodes
            ntmpl = max(tntmpl, ntmpl)

        print "computing dense preconditioner:"
        print "     lmax  =", lmax
        print "     ntmpl =", ntmpl

        for i in np.arange(0, nrlm):
            if np.mod(i, int( 0.1 * nrlm) ) == 0: print ("   filling M: %4.1f" % (100. * i / nrlm)), "%"
            trlm[i]   = 1.0
            tmat[:,i] = util_alm.alm2rlm( fwd_op( util_alm.rlm2alm(trlm) ) )
            trlm[i]   = 0.0

        print "   inverting M..."
        eigv, eigw = np.linalg.eigh( tmat )

        eigv_inv = 1.0 / eigv

        if ntmpl > 0:
            # do nothing to the ntmpl eigenmodes
            # with the lowest eigenvalues.
            print "     eigv[ntmpl-1] = ", eigv[ntmpl-1]
            print "     eigv[ntmpl]   = ", eigv[ntmpl]
            eigv_inv[0:ntmpl] = 1.0

        self.minv = np.dot( np.dot( eigw, np.diag(eigv_inv)), np.transpose(eigw) )

        if cache_fname != None:
            pk.dump( [lmax, self.hashdict(lmax, fwd_op), self.minv], open(cache_fname, 'w') )

    def hashdict(self, lmax, fwd_op):
        return { 'lmax'   : lmax,
                 'fwd_op' : fwd_op.hashdict() }

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return util_alm.rlm2alm( np.dot( self.minv, util_alm.alm2rlm(talm) ) )

# ===

class alm_filter_ninv_filts(object):
    def __init__(self, n_inv_filts, degrade_single=False):
        self.n_inv_filts    = n_inv_filts
        self.degrade_single = degrade_single

    def hashdict(self):
        return { 'degrade_single' : self.degrade_single,
                 'n_inv_filts'    : [ n_inv_filt.hashdict() for n_inv_filt in self.n_inv_filts ] }

    def degrade(self, nside):
        degraded_filts = [ n_inv_filt.degrade(nside) for n_inv_filt in self.n_inv_filts ]

        if self.degrade_single == True:
            n_inv    = np.zeros( 12*nside**2 )
            b_transf = np.zeros( len(degraded_filts[0].b_transf - 1) )

            for filt in degraded_filts:
                n_inv    += filt.n_inv
                b_transf += filt.b_transf

                assert(filt.marge_dipole   == True)
                assert(filt.marge_monopole == True)

            b_transf /= len(degraded_filts)

            return alm_filter_ninv_filts( [opfilt_tt.alm_filter_ninv( n_inv, b_transf, marge_monopole=True, marge_dipole=True )] )
        else:
            return alm_filter_ninv_filts( degraded_filts )

    def __iter__(self):
        for n_inv_filt in self.n_inv_filts:
            yield n_inv_filt

    def __getitem__(self, i):
        return self.n_inv_filts[i]
