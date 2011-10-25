# opfilt_tt
#
# operations and filters for temperature only c^-1
# S^{-1} (S^{-1} + Y^t N^{-1} Y)^{-1} Y^t N^{-1}

import os
import numpy  as np
import healpy as hp

import util
import spectra
import template_removal

def dot_op(alm1, alm2):
    lmax1 = util.nlm2lmax(len(alm1))
    lmax2 = util.nlm2lmax(len(alm2))

    assert(lmax1 == lmax2)
    lmax = lmax1
    
    tcl = spectra.alm_cl_cross(alm1, alm2)

    return np.sum( tcl * (2.*np.arange(0, lmax+1) + 1) )

def calc_prep_map(map, s_cls, n_inv_filt):
    tmap = np.copy(map)
    n_inv_filt.apply_map(tmap)

    lmax  = len(n_inv_filt.b_transf) - 1
    npix  = len(map)

    alm  = hp.map2alm(tmap, lmax=lmax, iter=0, regression=False)
    alm *= npix / (4.*np.pi)

    hp.almxfl( alm, n_inv_filt.b_transf, inplace=True )
    return alm

def apply_fini(alm, s_cls, n_inv_filt):
    cltt_inv = np.zeros(len(s_cls.cltt))
    cltt_inv[np.where(s_cls.cltt != 0)] = 1.0/s_cls.cltt[np.where(s_cls.cltt != 0)]
    
    hp.almxfl( alm, cltt_inv, inplace=True )

class fwd_op():
    def __init__(self, s_cls, n_inv_filt):
        # s_cls - a cl object
        # n_inv - a healpix map array

        self.cltt_inv = np.zeros(len(s_cls.cltt))
        self.cltt_inv[np.where(s_cls.cltt != 0)] = 1.0/s_cls.cltt[np.where(s_cls.cltt != 0)]

        self.n_inv_filt = n_inv_filt

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        if ( np.all(talm == 0) ): # do nothing if zero
            return talm

        alm = np.copy(talm)
        self.n_inv_filt.apply_alm(alm)
        alm += hp.almxfl(talm, self.cltt_inv)

        return alm

class pre_op_diag():
    def __init__(self, s_cls, n_inv_filt):
        assert( len(s_cls.cltt) >= len(n_inv_filt.b_transf) )
        
        n_inv_cl = np.sum( n_inv_filt.n_inv ) / (4.0*np.pi)

        lmax = len(n_inv_filt.b_transf)-1
        assert( lmax <= (len(s_cls.cltt)-1) )
        
        filt = np.zeros(lmax+1)

        filt[ np.where(s_cls.cltt[0:lmax+1] != 0) ] += 1.0/s_cls.cltt[np.where(s_cls.cltt[0:lmax+1] != 0)]
        filt[ np.where(n_inv_filt.b_transf[0:lmax+1] != 0) ] += n_inv_cl * n_inv_filt.b_transf[np.where(n_inv_filt.b_transf[0:lmax+1] != 0)]**2

        filt[np.where(filt != 0)] = 1.0/filt[np.where(filt != 0)]

        self.filt = filt

    def __call__(self, talm):
        return self.calc(talm)
        
    def calc(self, talm):
        return hp.almxfl(talm, self.filt)

class pre_op_dense():
    def __init__(self, lmax, fwd_op):
        # construct a low-l, low-nside dense preconditioner by brute force.
        # order of operations is O(nside**2 lmax**3) ~ O(lmax**5), so doing
        # by brute force is still comparable to matrix inversion, with
        # benefit of being very simple to implement.

        nrlm = (lmax+1)**2
        tmat = np.zeros( ( nrlm, nrlm ) )
        trlm = np.zeros( nrlm )

        fwd_op.n_inv_filt.load_mem()
        ntmpl = 0
        for t in fwd_op.n_inv_filt.templates:
            ntmpl += t.nmodes

        print "initializing dense preconditioner:"
        print "     lmax  =", lmax
        print "     ntmpl =", ntmpl

        for i in np.arange(0, nrlm):
            if np.mod(i, int( 0.1 * nrlm) ) == 0: print ("   filling M: %4.1f" % (100. * i / nrlm)), "%"
            trlm[i]   = 1.0
            tmat[:,i] = util.alm2rlm( fwd_op( util.rlm2alm(trlm) ) )
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

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return util.rlm2alm( np.dot( self.minv, util.alm2rlm(talm) ) )


class alm_filter_ninv():
    def __init__(self, n_inv, b_transf, marge_monopole=False, marge_dipole=False, marge_maps=[]):
        self.par_n_inv      = n_inv
        self.b_transf       = b_transf
        self.marge_monopole = marge_monopole
        self.marge_dipole   = marge_dipole
        self.par_marge_maps = marge_maps
        
        self.memloaded = False

    def load_mem(self):
        if self.memloaded == True:
            return
        
        # load marge maps.
        n_inv = self.par_n_inv
        if isinstance( n_inv, basestring ):
            n_inv = hp.read_map(self.n_inv)
    
        marge_maps = []
        for i, v in enumerate(self.par_marge_maps):
            if isinstance( v, basestring ):
                marge_maps.append( hp.read_map(v) )
            else:
                marge_maps.append(v)
        # --

        for tmap in marge_maps:
            assert( len(n_inv) == len(tmap) )
        
        templates = [template_removal.template_map(tmap) for tmap in marge_maps]
        if (self.marge_monopole): templates.append(template_removal.template_monopole())
        if (self.marge_dipole)  : templates.append(template_removal.template_dipole())

        if (len(templates) != 0):
            nmodes = np.sum([t.nmodes for t in templates])
            modes_idx_t = np.concatenate(([t.nmodes*[int(im)] for im, t in enumerate(templates)]))
            modes_idx_i = np.concatenate(([range(0,t.nmodes) for t in templates]))
            
            Pt_Nn1_P = np.zeros((nmodes, nmodes))
            for ir in range(0, nmodes):
                tmap = np.copy(n_inv)
                templates[modes_idx_t[ir]].apply_mode(tmap, int(modes_idx_i[ir]))

                ic = 0
                for tc in templates[0:modes_idx_t[ir]+1]:
                    Pt_Nn1_P[ir, ic:(ic+tc.nmodes)] = tc.dot(tmap)
                    Pt_Nn1_P[ic:(ic+tc.nmodes), ir] = Pt_Nn1_P[ir, ic:(ic+tc.nmodes)]
                    ic += tc.nmodes

            self.Pt_Nn1_P_inv     = np.linalg.inv(Pt_Nn1_P)

        self.n_inv      = n_inv
        self.marge_maps = marge_maps
        self.templates  = templates

        self.memloaded = True

    def degrade(self, nside):
        self.load_mem()
        
        if nside == hp.npix2nside(len(self.n_inv)):
            return self
        else:
            marge_maps = []
            n_marge_maps = len(self.templates) - (self.marge_monopole + self.marge_dipole)
            if ( n_marge_maps > 0 ):
                marge_maps = [hp.ud_grade(ti.map, nside) for ti in self.templates[0:n_marge_maps]]
            return alm_filter_ninv(hp.ud_grade(self.n_inv, nside, power=-2), self.b_transf, self.marge_monopole, self.marge_dipole, marge_maps)

    def apply_alm(self, alm):
        self.load_mem()
        
        # applies Y^T N^{-1} Y
        npix = len(self.n_inv)
        
        hp.almxfl(alm, self.b_transf, inplace=True)

        tmap = hp.alm2map(alm, hp.npix2nside(npix))

        self.apply_map(tmap)

        alm[:]  = hp.map2alm(tmap, lmax=util.nlm2lmax(len(alm)), iter=0, regression=False)
        alm[:] *= (npix / (4.*np.pi))
        
        hp.almxfl(alm, self.b_transf, inplace=True)
        
    def apply_map(self, map):
        self.load_mem()
        
        # applies N^{-1}
        map *= self.n_inv

        if len(self.templates) != 0:
            coeffs = np.concatenate(([t.dot(map) for t in self.templates]))
            coeffs = np.dot( self.Pt_Nn1_P_inv, coeffs )

            pmodes = np.zeros( len(self.n_inv) )
            im = 0
            for t in self.templates:
                t.accum(pmodes, coeffs[im:(im+t.nmodes)])
                im += t.nmodes
            pmodes *= self.n_inv
            map -= pmodes
