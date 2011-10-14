import time
import numpy as np

class dt():
    def __init__(self, _dt):
        self.dt = _dt

    def __str__(self):
        return ('%02d:%02d:%02d' % (np.floor(self.dt / 60 / 60),
                                 np.floor(np.mod(self.dt, 60*60) / 60 ),
                                 np.floor(np.mod(self.dt, 60)) ) )
    def __int__(self):
        return int(self.dt)

class stopwatch():
    def __init__(self):
        self.st = time.time()
        self.lt = self.st

    def lap(self):
        lt      = time.time()
        ret     = ( dt(lt - self.st), dt(lt - self.lt) )
        self.lt = lt
        return ret

    def elapsed(self):
        lt      = time.time()
        ret     = dt(lt - self.st)
        self.lt = lt
        return ret

class attr_dict(dict):
    def __init__(self, ini=None):
        if ini is not None:
            for key in ini.keys():
                self[key] = ini[key]

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

# ---

def nlm2lmax(nlm):
    lmax = int(np.floor(np.sqrt(2*nlm)-1))
    assert( (lmax+2)*(lmax+1)/2 == nlm )
    return lmax

def lmax2nlm(lmax):
    return (lmax+1)*(lmax+2)/2

def alm_splice(alm_lo, alm_hi, lsplit):
    alm_lo_lmax = nlm2lmax( len(alm_lo) )
    alm_hi_lmax = nlm2lmax( len(alm_hi) )

    assert( alm_lo_lmax >= lsplit )
    assert( alm_hi_lmax >= lsplit )

    ret = np.copy(alm_hi)
    for m in xrange(0, lsplit):
        ret[(m*(2*alm_hi_lmax+1-m)/2 + m):(m*(2*alm_hi_lmax+1-m)/2+lsplit)] = alm_lo[(m*(2*alm_lo_lmax+1-m)/2 + m):(m*(2*alm_lo_lmax+1-m)/2+lsplit)]
    return ret

def alm_copy(alm, lmax=None):
    alm_lmax = nlm2lmax(len(alm))
    assert(lmax <= alm_lmax)
    
    if (alm_lmax == lmax) or (lmax == None):
        ret = np.copy(alm)
    else:
        ret = np.zeros(lmax2nlm(lmax), dtype=np.complex)
        for m in xrange(0, lmax+1):
            ret[(m*(2*lmax+1-m)/2):(m*(2*lmax+1-m)/2 + m + 1)] = alm[(m*(2*alm_lmax+1-m)/2):(m*(2*alm_lmax+1-m)/2 + m+1)]

    return ret
