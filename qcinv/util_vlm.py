import numpy  as np
import healpy as hp

import util_alm

def map2vlm( m, lmax, iter=0 ):
    alm_re = hp.map2alm( m.real.copy(), lmax=lmax, iter=iter, regression=False )
    alm_im = hp.map2alm( m.imag.copy(), lmax=lmax, iter=iter, regression=False )

    ret = np.zeros( (lmax+1)**2, dtype=np.complex )
    for l in xrange(0, lmax+1):
        ms = np.arange(1,l+1)
        ret[l*l+l]    = alm_re[l] + 1.j * alm_im[l]
        ret[l*l+l+ms] = alm_re[ms * (2*lmax+1-ms)/2 + l] + 1.j * alm_im[ms * (2*lmax+1-ms)/2 + l]
        ret[l*l+l-ms] = (-1)**ms * ( np.conj( alm_re[ms * (2*lmax+1-ms)/2 + l] ) + 1.j * np.conj( alm_im[ms * (2*lmax+1-ms)/2 + l] ) )
    return ret

def vlm2alm_gc( vlm ):
    lmax = int(np.sqrt(len(vlm))-1)

    glm = np.zeros( util_alm.lmax2nlm(lmax), dtype=np.complex )
    clm = np.zeros( util_alm.lmax2nlm(lmax), dtype=np.complex )

    for l in xrange(0, lmax+1):
        ms = np.arange(1,l+1)

        glm[l] = vlm[l*l+l].real
        clm[l] = vlm[l*l+l].imag

        glm[ms * (2*lmax+1-ms)/2 + l] = -0.5  * ( vlm[l*l+l+ms] + (-1)**ms * np.conj( vlm[l*l+l-ms] ) )
        clm[ms * (2*lmax+1-ms)/2 + l] =  0.5j * ( vlm[l*l+l+ms] - (-1)**ms * np.conj( vlm[l*l+l-ms] ) )
    return glm, clm

def vlm_cl_cross(vlm1, vlm2, lmax=None):
    vlm1_lmax = int(np.sqrt(len(vlm1))-1)
    vlm2_lmax = int(np.sqrt(len(vlm2))-1)
    
    assert(vlm1_lmax == np.sqrt(len(vlm1))-1)
    assert(vlm2_lmax == np.sqrt(len(vlm2))-1)
    
    if lmax == None:
        lmax = min( vlm1_lmax, vlm2_lmax)
    assert(lmax <= min(vlm1_lmax, vlm2_lmax))

    ret = np.zeros(lmax+1, dtype=np.complex)
    for l in xrange(0, lmax+1):
        ret[l] = np.sum( vlm1[(l*l):(l+1)*(l+1)] * np.conj( vlm2[(l*l):(l+1)*(l+1)]) ) / (2.*l+1.)
    return ret

def vlm_cl(vlm, lmax=None):
    return vlm_cl_cross(vlm, vlm, lmax)

def vlm_cl_cross_gc(vlm1, vlm2, lmax=None):
    vlm1_lmax = int(np.sqrt(len(vlm1))-1)
    vlm2_lmax = int(np.sqrt(len(vlm2))-1)

    assert(vlm1_lmax == np.sqrt(len(vlm1))-1)
    assert(vlm2_lmax == np.sqrt(len(vlm2))-1)

    if lmax == None:
        lmax = min( vlm1_lmax, vlm2_lmax)
    assert(lmax <= min( vlm1_lmax, vlm2_lmax))

    retg = np.zeros(lmax+1)
    retc = np.zeros(lmax+1)
    for l in xrange(0, lmax+1):
        tmon = np.sum( vlm1[(l*l):(l+1)*(l+1)] * np.conj( vlm2[(l*l):(l+1)*(l+1)]) )  / (2.*l+1.) * 0.5
        tdif = np.sum( vlm1[(l*l):(l+1)*(l+1)] * vlm2[(l*l):(l+1)*(l+1)][::-1] * np.array([(-1)**m for m in xrange(-l,l+1)]) ).real / (2.*l+1.) * 0.5

        retg[l] = tmon + tdif
        retc[l] = tmon - tdif
    return retg, retc

def vlm_cl_gc(vlm, lmax=None):
    return vlm_cl_cross_gc(vlm, vlm, lmax)
