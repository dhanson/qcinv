import os, sys, urllib, copy
import numpy as np

def hash_check(hash1, hash2):
    if hash1 != hash2:
        print 'ERROR: HASHCHECK FAIL'
        print 'hash1 = ', hash1
        print 'hash2 = ', hash2
        assert(0)

def bl_gauss(fwhm, lmax):
    ls = np.arange(0, lmax+1)
    c  = (fwhm * np.pi/180./60.)**2 / (16.*np.log(2.0))

    return np.exp(-c*ls*(ls+1.))

def download(url, fname):
    print "simcache::util::download. download " + url + " -> " + fname
    pct = 0
    def dlhook(count, block_size, total_size):
        ppct = int(100. * (count-1) * block_size / total_size)
        cpct = int(100. * (count+0) * block_size / total_size)
        if cpct > ppct: sys.stdout.write( "\r simcache::util::download:: " + int(10. * cpct / 100)*"-" + "> " + ("%02d" % cpct) + r"%" ); sys.stdout.flush()

    try:
        urllib.urlretrieve(url, filename=fname, reporthook=dlhook)
        sys.stdout.write( "\n" ); sys.stdout.flush()
    except:
        print "download failed! removing partial."
        os.remove(rfname)

class avg:
    def __init__(self, dovar=True, docov=False, clone=copy.deepcopy):
        self.dovar = dovar
        self.docov = docov
        self.nobj  = 0.
        self.clone = clone

    def __iadd__(self, obj):
        self.add(obj)
        return self

    def add(self, obj):
        if hasattr(self, 'sumavg'):
            self.sumavg += obj
            if (self.dovar): self.sumvar += obj*obj
            if (self.docov): self.sumcov += transpose(matrix(obj))*obj
        else:
            self.sumavg = self.clone(obj)
            if (self.dovar): self.sumvar = obj*obj
            if (self.docov): self.sumcov = transpose(matrix(obj))*obj

        self.nobj += 1

    def avg(self):
        if hasattr(self, 'sumavg'):
            return self.sumavg / self.nobj
        else:
            return 0

    def var(self):
        assert( hasattr(self, 'sumvar') )

        var = (self.sumvar - (self.sumavg)*(self.sumavg) / self.nobj)/self.nobj
        return var

    def cov(self):
        assert( hasattr(self, 'sumcov') )

        avg = self.sumavg / self.nobj
        cov = self.sumcov / self.nobj - (np.transpose(np.matrix(avg)) * avg)
        return cov

    def cor(self):
        print 'cov.size == ', shape(self.cov())
        cov = self.cov()
        var = np.diag(cov)

        cor = np.array(cov.copy())
        icv = 1.0/np.sqrt(var)
        for (vi, v) in enumerate(icv):
            cor[vi,:] *= (icv * v)

        return cor
