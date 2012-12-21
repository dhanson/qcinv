import time
import numpy  as np
import healpy as hp

# ===

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

# ===

class jit:
    # just-in-time instantiation wrapper class.
    def __init__(self, ctype, *cargs, **ckwds):
        self.__dict__['__jit_args'] = [ctype, cargs, ckwds]
        self.__dict__['__jit_obj']  = None

    def instantiate(self):
        [ctype, cargs, ckwds] = self.__dict__['__jit_args']
        print 'jit: instantiating ctype =', ctype
        self.__dict__['__jit_obj'] = ctype( *cargs, **ckwds )
        del self.__dict__['__jit_args']

    def __getattr__(self, attr):
        if self.__dict__['__jit_obj'] == None:
            self.instantiate()
        return getattr(self.__dict__['__jit_obj'], attr)

    def __setattr__(self, attr, val):
        if self.__dict__['__jit_obj'] == None:
            self.instantiate()
        setattr(self.__dict__['__jit_obj'], attr, val)

# ===

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

        self.lmax = lmax
        self.ls   = np.concatenate( [ [0,1], ell ] )
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
            self.cltd = np.concatenate( [ [0,0], tarray[0:(lmax-1),5]*np.sqrt(ell*(ell+1.))/ell**3/tcmb ] )

            self.clpp = np.concatenate( [ [0,0], tarray[0:(lmax-1),4]/ell**4/tcmb**2     ] )

# ===

def load_map(f):
    if type(f) is str:
        return hp.read_map(f)
    else:
        return f
