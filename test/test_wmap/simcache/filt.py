import os, hashlib
import numpy  as np
import pickle as pk
import healpy as hp

import qcinv

import util
import wmap

class ivf_teb_library(object):
    # a collection of inverse-variance filtered maps.
    
    def __init__(self, sim_lib, lib_dir):
        self.sim_lib = sim_lib
        self.lib_dir = lib_dir

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        if not os.path.exists(lib_dir + "/sim_hash.pk"):
            pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
        util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        return { 'sim_lib' : self.sim_lib.hashdict() }

    def get_dat_tlm(self, det):
        tfname = self.lib_dir + "/dat_det_" + det + "_tlm.npy"
        if not os.path.exists(tfname): self.cache_dat_teb(det)
        return np.load(tfname)

    def get_dat_elm(self, det):
        tfname = self.lib_dir + "/dat_det_" + det + "_elm.npy"
        if not os.path.exists(tfname): self.cache_dat_teb(det)
        return np.load(tfname)

    def get_dat_blm(self, det):
        tfname = self.lib_dir + "/dat_det_" + det + "_blm.npy"
        if not os.path.exists(tfname): self.cache_dat_teb(det)
        return np.load(tfname)

    def get_sim_tlm(self, det, idx):
        tfname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_tlm.npy"
        if not os.path.exists(tfname): self.cache_sim_teb(det, idx)
        return np.load(tfname)

    def get_sim_elm(self, det, idx):
        tfname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_elm.npy"
        if not os.path.exists(tfname): self.cache_sim_teb(det, idx)
        return np.load(tfname)

    def get_sim_blm(self, det, idx):
        tfname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_blm.npy"
        if not os.path.exists(tfname): self.cache_sim_teb(det, idx)
        return np.load(tfname)

    def cache_sim_teb(self, det, idx):
        tlm_fname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_tlm.npy"
        elm_fname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_elm.npy"
        blm_fname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_blm.npy"

        assert( not any(os.path.exists(fname) for fname in [tlm_fname, elm_fname, blm_fname] ) )

        tlm, elm, blm = self.apply_ivf( det, self.sim_lib.get_sim_tmap(det, idx), self.sim_lib.get_sim_pmap(det, idx) )
        
        np.save(tlm_fname, tlm)
        np.save(elm_fname, elm)
        np.save(blm_fname, blm)

    def cache_dat_teb(self, det):
        tlm_fname = self.lib_dir + "/dat_det_" + det + "_tlm.npy"
        elm_fname = self.lib_dir + "/dat_det_" + det + "_elm.npy"
        blm_fname = self.lib_dir + "/dat_det_" + det + "_blm.npy"

        assert( not any(os.path.exists(fname) for fname in [tlm_fname, elm_fname, blm_fname] ) )

        tlm, elm, blm = self.apply_ivf( det, self.sim_lib.get_dat_tmap(det), self.sim_lib.get_dat_pmap(det) )
        
        np.save(tlm_fname, tlm)
        np.save(elm_fname, elm)
        np.save(blm_fname, blm)

    def apply_ivf( self, det, tmap, pmap ):
        assert(0)

    def get_fsky(self):
        mask_t = qcinv.util.load_map(self.mask_t)
        mask_p = qcinv.util.load_map(self.mask_p)
        assert(len(mask_t) == len(mask_p))

        npix = len(mask_t)
        return ( mask_t.sum() / npix,
                 (mask_t * mask_p).sum() / npix,
                 mask_p.sum() / npix )

class ivf_teb_fl_library(ivf_teb_library):
    # library for symmetric inverse-variance.
    
    def __init__(self, lmax, nside, bl, ftl, fel, fbl, mask_t, mask_p, sim_lib, lib_dir):
        # note: ftl, fel, fbl are the effective symmetric inverse-variance filter (after beam deconvolution).
        #       for a sky which has already been beam deconvolved by 1/(bl*pxw)

        self.lmax  = lmax
        self.nside = nside
        self.bl    = bl
        self.ftl   = ftl
        self.fel   = fel
        self.fbl   = fbl

        self.mask_t = mask_t
        self.mask_p = mask_p
        
        super( ivf_teb_fl_library, self ).__init__( sim_lib, lib_dir )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        return { 'lmax'    : self.lmax,
                 'nside'   : self.nside,
                 'bl'      : hashlib.sha1(self.bl.view(np.uint8)).hexdigest(),
                 'ftl'     : hashlib.sha1(self.ftl.view(np.uint8)).hexdigest(),
                 'fel'     : hashlib.sha1(self.fel.view(np.uint8)).hexdigest(),
                 'fbl'     : hashlib.sha1(self.fbl.view(np.uint8)).hexdigest(),
                 'mask_t'  : hashlib.sha1(qcinv.util.load_map(self.mask_t).view(np.uint8)).hexdigest(),
                 'mask_p'  : hashlib.sha1(qcinv.util.load_map(self.mask_p).view(np.uint8)).hexdigest(),
                 'sim_lib' : self.sim_lib.hashdict(), 
                 'super'   : super( ivf_teb_fl_library, self ).hashdict() }

    def apply_ivf(self, det, tmap, pmap):
        mask_t = qcinv.util.load_map(self.mask_t)
        mask_p = qcinv.util.load_map(self.mask_p)

        bl     = self.bl
        pxw    = hp.pixwin(self.nside)[0:self.lmax+1]

        tlm = hp.map2alm( tmap * mask_t, lmax=self.lmax, iter=0, regression=False )
        elm, blm = hp.map2alm_spin( (pmap.real * mask_p, pmap.imag * mask_p), 2, lmax=self.lmax )

        hp.almxfl( tlm, self.ftl / bl / pxw, inplace=True )
        hp.almxfl( elm, self.fel / bl / pxw, inplace=True )
        hp.almxfl( blm, self.fbl / bl / pxw, inplace=True )
    
        return tlm, elm, blm

    def get_ftebl(self, det):
        return self.ftl, self.fel, self.fbl
