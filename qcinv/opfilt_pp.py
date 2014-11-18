# opfilt_pp
#
# operations and filters for polarization only c^-1
# S^{-1} (S^{-1} + Y^t N^{-1} Y)^{-1} Y^t N^{-1}

import os, hashlib
import numpy  as np
import healpy as hp
import pickle as pk

import util
import util_alm
import template_removal

class eblm():
    def __init__(self, alm ):
        [elm, blm] = alm
        assert(len(elm) == len(blm))
        
        self.lmax = util_alm.nlm2lmax(len(elm))
        
        self.elm = elm
        self.blm = blm

    def alm_copy(self, lmax=None):
        return eblm( [ util_alm.alm_copy( self.elm, lmax=lmax ),
                       util_alm.alm_copy( self.blm, lmax=lmax ) ] )

    def alm_splice(self, alm_hi, lsplit):
        return eblm( [ util_alm.alm_splice( self.elm, alm_hi.elm, lsplit ),
                       util_alm.alm_splice( self.blm, alm_hi.blm, lsplit ) ] )

    def __add__(self, other):
        assert( self.lmax == other.lmax )
        return eblm( [self.elm + other.elm, self.blm + other.blm] )

    def __sub__(self, other):
        assert( self.lmax == other.lmax )
        return eblm( [self.elm - other.elm, self.blm - other.blm] )

    def __iadd__(self, other):
        assert( self.lmax == other.lmax )
        self.elm += other.elm; self.blm += other.blm
        return self

    def __isub__(self, other):
        assert( self.lmax == other.lmax )
        self.elm -= other.elm; self.blm -= other.blm
        return self

    def __mul__(self, other):
        return eblm( [self.elm * other, self.blm * other] )
# ===

def calc_prep(maps, s_cls, n_inv_filt):
    qmap, umap = np.copy(maps[0]), np.copy(maps[1])
    assert(len(qmap) == len(umap))
    npix  = len(qmap)
    
    n_inv_filt.apply_map([qmap, umap])

    lmax  = len(n_inv_filt.b_transf) - 1

    tlm, elm, blm = hp.map2alm( [qmap, qmap, umap], lmax=lmax, iter=0, regression=False, use_weights=False, pol=True )
    del tlm
    elm *= npix / (4.*np.pi); blm *= npix / (4.*np.pi)

    hp.almxfl( elm, n_inv_filt.b_transf, inplace=True )
    hp.almxfl( blm, n_inv_filt.b_transf, inplace=True )
    return eblm([elm, blm])

def apply_fini(alm, s_cls, n_inv_filt):
    sfilt = alm_filter_sinv(s_cls)
    ret = sfilt.calc(alm)
    alm.elm[:] = ret.elm[:]; alm.blm[:] = ret.blm[:]

# ===

class dot_op():
    def __init__(self, lmax=None):
        self.lmax = lmax

    def __call__(self, alm1, alm2):
        lmax1 = alm1.lmax
        lmax2 = alm2.lmax

        if self.lmax != None:
            lmax = self.lmax
        else:
            assert(lmax1 == lmax2)
            lmax = lmax1

        tcl = util_alm.alm_cl_cross(alm1.elm, alm2.elm) + util_alm.alm_cl_cross(alm1.blm, alm2.blm)

        return np.sum( tcl[2:] * (2.*np.arange(2, lmax+1) + 1) )

class fwd_op():
    def __init__(self, s_cls, n_inv_filt):
        self.s_inv_filt = alm_filter_sinv(s_cls)
        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return { 's_inv_filt' : self.s_inv_filt.hashdict(),
                 'n_inv_filt' : self.n_inv_filt.hashdict() }

    def __call__(self, alm):
        return self.calc(alm)

    def calc(self, alm):
        nlm = alm*1.0
        self.n_inv_filt.apply_alm( nlm )

        slm = self.s_inv_filt.calc( alm )

        return nlm+slm

# ===

class pre_op_diag():
    def __init__(self, s_cls, n_inv_filt):
        s_inv_filt = alm_filter_sinv(s_cls)
        assert( (s_inv_filt.lmax+1) >= len(n_inv_filt.b_transf) )

        ninv_fel, ninv_fbl = n_inv_filt.get_febl()

        lmax = len(n_inv_filt.b_transf)-1

        flmat = s_inv_filt.slinv[0:lmax+1,:,:]

        for l in xrange(0,lmax+1):
            flmat[l,0,0] += ninv_fel[l]
            flmat[l,1,1] += ninv_fbl[l]

            flmat[l,:,:] = np.linalg.pinv( flmat[l,:,:].reshape((2,2)) )
        
        self.flmat = flmat

    def __call__(self, talm):
        return self.calc(talm)
        
    def calc(self, alm):
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

    def alm2rlm(self, alm):
        rlm = np.zeros( 2*(alm.lmax+1)**2 )
        rlm[0*(alm.lmax+1)**2:1*(alm.lmax+1)**2] = util_alm.alm2rlm(alm.elm)
        rlm[1*(alm.lmax+1)**2:2*(alm.lmax+1)**2] = util_alm.alm2rlm(alm.blm)
        return rlm

    def rlm2alm(self, rlm):
        lmax = int(np.sqrt(len(rlm)/2)-1)
        return eblm( [ util_alm.rlm2alm( rlm[0*(lmax+1)**2:1*(lmax+1)**2] ),
                       util_alm.rlm2alm( rlm[1*(lmax+1)**2:2*(lmax+1)**2] ) ] )

    def compute_minv(self, lmax, fwd_op, cache_fname=None):
        if cache_fname != None:
            assert(not os.path.exists(cache_fname))

        nrlm = 2*(lmax+1)**2
        trlm = np.zeros( nrlm )
        tmat = np.zeros( ( nrlm, nrlm ) )

        ntmpl = 0
        for t in fwd_op.n_inv_filt.templates_p:
            ntmpl += t.nmodes
        ntmpl += 8 # (1 mono + 3 dip) * (e+b)

        print "computing dense preconditioner:"
        print "     lmax  =", lmax
        print "     ntmpl =", ntmpl

        for i in np.arange(0, nrlm):
            if np.mod(i, int( 0.1 * nrlm) ) == 0: print ("   filling M: %4.1f" % (100. * i / nrlm)), "%"
            trlm[i]   = 1.0
            tmat[:,i] = self.alm2rlm( fwd_op( self.rlm2alm(trlm) ) )
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
        return self.rlm2alm( np.dot( self.minv, self.alm2rlm(talm) ) )

# ===

class alm_filter_sinv():
    def __init__(self, s_cls):
        lmax = s_cls.lmax
        zs   = np.zeros(lmax+1)

        slmat = np.zeros( (lmax+1, 2, 2) ) # matrix of EB correlations at each l.
        slmat[:,0,0] = getattr(s_cls, 'clee', zs.copy())[:]
        slmat[:,0,1] = getattr(s_cls, 'cleb', zs.copy())[:]; slmat[:,1,0] = slmat[:,0,1]
        slmat[:,1,1] = getattr(s_cls, 'clbb', zs.copy())[:]
        
        slinv = np.zeros( (lmax+1, 2, 2) )
        for l in xrange(0, lmax+1):
            slinv[l,:,:] = np.linalg.pinv( slmat[l] )
            
        self.lmax  = lmax
        self.slinv = slinv

    def calc(self, alm):
        tmat = self.slinv
        
        relm = hp.almxfl( alm.elm, tmat[:,0,0], inplace=False ) + hp.almxfl( alm.blm, tmat[:,0,1], inplace=False )
        rblm = hp.almxfl( alm.elm, tmat[:,1,0], inplace=False ) + hp.almxfl( alm.blm, tmat[:,1,1], inplace=False )
        return eblm([relm, rblm])

    def hashdict(self):
        return { 'slinv' : hashlib.sha1( self.slinv.flatten().view(np.uint8) ).hexdigest() }
    
class alm_filter_ninv():
    def __init__(self, n_inv, b_transf, marge_maps=[]):
        self.n_inv = []
        for i, tn in enumerate(n_inv):
            if isinstance(tn, list):
                n_inv_prod = util.load_map(tn[0][:])
                if (len(tn) > 1):
                    for n in tn[1:]:
                        n_inv_prod = n_inv_prod * util.load_map(n[:])
                self.n_inv.append(n_inv_prod)
                assert( np.std( self.n_inv[i][ np.where(self.n_inv[i][:] != 0.0) ] ) / np.average( self.n_inv[i][ np.where(self.n_inv[i][:] != 0.0) ] ) < 1.e-7 )
            else:
                self.n_inv.append(util.load_map(n_inv[i]))
        n_inv = self.n_inv

        assert( len(n_inv) == 1 )
        npix  = len(n_inv[0])
        nside = hp.npix2nside(npix)
        for n in n_inv[1:]:
            assert( len(n) == npix )

        templates_p = []; templates_p_hash = []
        for tmap in [util.load_map(m) for m in marge_maps]:
            assert( npix == len(tmap) )
            templates_p.append( template_removal.template_map_p(tmap) )
            templates_p_hash.append( hashlib.sha1( tmap.view(np.uint8) ).hexdigest() )

        if (len(templates_p) != 0):
            nmodes = np.sum([t.nmodes for t in templates_p])
            modes_idx_t = np.concatenate(([t.nmodes*[int(im)] for im, t in enumerate(templates_p)]))
            modes_idx_i = np.concatenate(([range(0,t.nmodes) for t in templates_p]))
            
            Pt_Nn1_P = np.zeros((nmodes, nmodes))
            for ir in range(0, nmodes):
                tmap = np.copy(n_inv[0])
                templates_p[modes_idx_t[ir]].apply_mode(tmap, int(modes_idx_i[ir]))

                ic = 0
                for tc in templates_p[0:modes_idx_t[ir]+1]:
                    Pt_Nn1_P[ir, ic:(ic+tc.nmodes)] = tc.dot(tmap)
                    Pt_Nn1_P[ic:(ic+tc.nmodes), ir] = Pt_Nn1_P[ir, ic:(ic+tc.nmodes)]
                    ic += tc.nmodes

            self.Pt_Nn1_P_inv     = np.linalg.inv(Pt_Nn1_P)

        self.n_inv            = n_inv
        self.b_transf         = b_transf[:]

        self.templates_p      = templates_p
        self.templates_p_hash = templates_p_hash

        self.npix  = npix
        self.nside = nside

    def get_febl(self):
        if (False):
            pass
        elif (len(self.n_inv) == 1): # TT, 1/2(QQ+UU)
            n_inv_cl_p = np.sum( self.n_inv[0] ) / (4.0*np.pi) * self.b_transf**2

            return n_inv_cl_p, n_inv_cl_p
        elif (len(self.n_inv) == 3): # TT, QQ, QU, UU
            n_inv_cl_p = np.sum( 0.5*(self.n_inv[0] + self.n_inv[2]) ) / (4.0*np.pi) * self.b_transf**2

            return n_inv_cl_p, n_inv_cl_p
        else:
            assert(0)

    def hashdict(self):
        return { 'n_inv'            : [ hashlib.sha1( n.view(np.uint8) ).hexdigest() for n in self.n_inv ],
                 'b_transf'         : hashlib.sha1( self.b_transf.view(np.uint8) ).hexdigest(),
                 'templates_p_hash' : self.templates_p_hash }
        
    def degrade(self, nside):
        if nside == self.nside:
            return self
        else:
            marge_maps_p = []
            n_marge_maps_p = len(self.templates_p)
            if ( n_marge_maps_p > 0 ):
                marge_maps_p = [hp.ud_grade(ti.map, nside) for ti in self.templates_p[0:n_marge_maps_p]]
            
            return alm_filter_ninv( [hp.ud_grade(n, nside, power=-2) for n in self.n_inv], self.b_transf, marge_maps_p )

    def apply_alm(self, alm):
        # applies Y^T N^{-1} Y
        lmax = alm.lmax
        
        hp.almxfl(alm.elm, self.b_transf, inplace=True)
        hp.almxfl(alm.blm, self.b_transf, inplace=True)

        tmap, qmap, umap = hp.alm2map( (alm.elm, alm.elm, alm.blm), self.nside, pol=True, inplace=False ) #FIXME: hack! need a P-only sht in healpy.
        del tmap

        self.apply_map( [qmap, umap] )
        npix = len(qmap)

        ttlm, telm, tblm = hp.map2alm( [qmap, qmap, umap], lmax=lmax, iter=0, regression=False, use_weights=False, pol=True )
        del ttlm
        alm.elm[:] = telm * (npix / (4.*np.pi)); alm.blm[:] = tblm * (npix / (4.*np.pi))
        
        hp.almxfl(alm.elm, self.b_transf, inplace=True)
        hp.almxfl(alm.blm, self.b_transf, inplace=True)
        
    def apply_map(self, amap):
        [qmap, umap] = amap
        
        # applies N^{-1}
        if (False):
            pass
        elif (len(self.n_inv) == 1): # TT, QQ=UU
            qmap *= self.n_inv[0]
            umap *= self.n_inv[0]
        elif (len(self.n_inv) == 3):  # TT, QQ, QU, UU
            qmap_copy = qmap.copy()
            
            qmap *= self.n_inv[0]
            qmap += self.n_inv[1] * umap
            
            umap *= self.n_inv[2]
            umap += self.n_inv[1] * qmap_copy

            del qmap_copy
        else:
            assert(0)

        if len(self.templates_p) != 0:
            coeffs = np.concatenate(([t.dot(tmap) for t in self.templates_p]))
            coeffs = np.dot( self.Pt_Nn1_P_inv, coeffs )

            pmodes = np.zeros( len(self.n_inv[0]) )
            im = 0
            for t in self.templates_p:
                t.accum(pmodes, coeffs[im:(im+t.nmodes)])
                im += t.nmodes
            pmodes *= self.n_inv[0]
            tmap -= pmodes
