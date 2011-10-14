import numpy as np

import util

class camb_clfile(object):
    def __init__(self, tfname, lmax=None):
        tarray = np.loadtxt(tfname)
        assert(tarray[ 0, 0] == 2)

        if lmax == None:
            lmax = np.shape(tarray)[0]+1
            assert(tarray[-1, 0] == lmax)
        assert( (np.shape(tarray)[0]+1) >= lmax )

        ncol = np.shape(tarray)[1]
        ell  = np.arange(2, lmax+1)

        self.ls = np.concatenate( [ [0,0], ell ] )
        if ncol == 5:
            self.cltt = np.concatenate( [ [0,0], tarray[0:(lmax-1),1]*2.*np.pi/ell/(ell+1.)       ] )
            self.clee = np.concatenate( [ [0,0], tarray[0:(lmax-1),2]*2.*np.pi/ell/(ell+1.)        ] )
            self.clbb = np.concatenate( [ [0,0], tarray[0:(lmax-1),3]*2.*np.pi/ell/(ell+1.)        ] )
            self.clte = np.concatenate( [ [0,0], tarray[0:(lmax-1),4]*2.*np.pi/ell/(ell+1.)        ] )

        elif ncol == 6:
            tcmb  = 2.726*1e6 #uK
            
            self.cltt = np.concatenate( [ [0,0], tarray[0:(lmax-1),1]*2.*np.pi/ell/(ell+1.)       ] )
            self.clee = np.concatenate( [ [0,0], tarray[0:(lmax-1),2]*2.*np.pi/ell/(ell+1.)       ] )
            self.clte = np.concatenate( [ [0,0], tarray[0:(lmax-1),3]*2.*np.pi/ell/(ell+1.)       ] )
            self.cldd = np.concatenate( [ [0,0], tarray[0:(lmax-1),4]*(ell+1.)/ell**3/tcmb**2     ] )
            self.cltd = np.concatenate( [ [0,0], tarray[0:(lmax-1),5]*np.sqrt(ell+1.)/ell**3/tcmb ] )

def alm_cl_cross(alm1, alm2, lmax=None):
    alm1_lmax = util.nlm2lmax(len(alm1))
    alm2_lmax = util.nlm2lmax(len(alm2))
    
    if lmax == None:
        lmax = min( alm1_lmax, alm2_lmax)
    assert(lmax <= min(alm1_lmax, alm2_lmax))

    ret    = np.zeros(lmax+1)
    for l in xrange(0, lmax+1):
        ms = np.arange(1,l+1)
        ret[l]  = np.real(alm1[l] * alm2[l])
        ret[l] += 2.*np.sum( np.real(alm1[ms * (2*alm1_lmax+1-ms)/2 + l] * np.conj(alm2[ms * (2*alm2_lmax+1-ms)/2 + l])) )
        ret[l] /= (2.*l+1.)
    return ret

def alm_cl(alm, lmax=None):
    return alm_cl_cross(alm, alm, lmax)

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

def vlm_cl_gc(vlm, lmax=None):
    vlm_lmax = int(np.sqrt(len(vlm))-1)
    assert(vlm_lmax == np.sqrt(len(vlm))-1)

    if lmax == None:
        lmax = vlm_lmax
    assert(lmax <= vlm_lmax)

    retg = np.zeros(lmax+1)
    retc = np.zeros(lmax+1)
    for l in xrange(0, lmax+1):
        tmon = np.sum( vlm[(l*l):(l+1)*(l+1)] * np.conj( vlm[(l*l):(l+1)*(l+1)]) )  / (2.*l+1.) * 0.5
        tdif = np.sum( vlm[(l*l):(l+1)*(l+1)] * vlm[(l*l):(l+1)*(l+1)][::-1] * np.array([(-1)**m for m in xrange(-l,l+1)]) ).real / (2.*l+1.) * 0.5

        retg[l] = tmon + tdif
        retc[l] = tmon - tdif
    return retg, retc
