# opfilt_pp_multi_simple
#
# operations and filters for polarization only c^-1
# with several input maps, each with perfectly
# correlated signal, and perfectly uncorrelated
# inhomogeneous noise, with independent levels.
#

import opfilt_pp
from opfilt_pp import apply_fini, dot_op, pre_op_dense

def calc_prep(maps, s_cls, n_inv_filts):
    alm = opfilt_pp.calc_prep( maps[0], s_cls, n_inv_filts[0] )
    for m, n_inv_filt in zip(maps[1:], n_inv_filts[1:]):
        alm += opfilt_pp.calc_prep( m, s_cls, n_inv_filt )
    return alm

class fwd_op():
    def __init__(self, s_cls, n_inv_filts):
        self.s_inv_filt = alm_filter_sinv(s_cls)
        self.n_inv_filts = n_inv_filts

    def hashdict(self):
        return { 's_inv_filt'  : self.s_inv_filt.hashdict(),
                 'n_inv_filts' : self.n_inv_filts.hashdict() }

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, alm):
        nlm = alm*1.0
        self.n_inv_filt.apply_alm( nlm )

        ret = self.s_inv_filt.calc( alm )

        for n_inv_filt in self.n_inv_filts:
            nlm = alm*1.
            n_inv_filt.apply_alm(nlm)
            ret += nlm
        return ret

class pre_op_diag():
    def __init__(self, s_cls, n_inv_filts):
        s_inv_filt = alm_filter_sinv(s_cls)

        flmat = s_inv_filt.slinv[0:lmax+1,:,:]

        for n_inv_filt in n_inv_filts:
            assert( (s_inv_filt.lmax+1) >= len(n_inv_filt.b_transf) )

            ninv_fel, ninv_fbl = n_inv_filt.get_febl()

            for l in xrange(0,lmax+1):
                flmat[l,0,0] += ninv_fel[l]
                flmat[l,1,1] += ninv_fbl[l]

        for l in xrange(0,lmax+1):
            flmat[l,:,:] = np.linalg.pinv( flmat[l,:,:].reshape((2,2)) )
        self.flmat = flmat

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        tmat = self.flmat
        
        relm = hp.almxfl( alm.elm, tmat[:,0,0], inplace=False ) + hp.almxfl( alm.blm, tmat[:,0,1], inplace=False )
        rblm = hp.almxfl( alm.elm, tmat[:,1,0], inplace=False ) + hp.almxfl( alm.blm, tmat[:,1,1], inplace=False )
        return eblm( [relm, rblm] )

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

        if (self.degrade_single == True):
            n_inv    = np.zeros( 12*nside**2 )
            b_transf = np.zeros( len(degraded_filts[0].b_transf - 1) )

            for filt in degraded_filts:
                n_inv    += filt.n_inv
                b_transf += filt.b_transf

            b_transf /= len(degraded_filts)

            return alm_filter_ninv_filts( [opfilt_pp.alm_filter_ninv( n_inv, b_transf )] )
        else:
            return alm_filter_ninv_filts( degraded_filts )

    def __iter__(self):
        for n_inv_filt in self.n_inv_filts:
            yield n_inv_filt

    def __getitem__(self, i):
        return self.n_inv_filts[i]
