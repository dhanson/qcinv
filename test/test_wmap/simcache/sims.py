import os, hashlib, math
import pickle as pk
import numpy  as np
import healpy as hp

import util
import wmap

class tlm_library(object):
    # library of temperature and polarization alms.

    def __init__(self, lib_dir):
        assert(lib_dir != None)
        
        self.lib_dir = lib_dir

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        if not  os.path.exists(lib_dir + "/sim_hash.pk"):
            pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
        util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        return {}

    def get_sim_tlm(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_tlm.npy"

        if not os.path.exists(tfname):
            self.cache_teb(idx)

        return np.load(tfname)

    def get_sim_elm(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_elm.npy"

        if not os.path.exists(tfname):
            self.cache_teb(idx)

        return np.load(tfname)

    def get_sim_blm(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_blm.npy"

        if not os.path.exists(tfname):
            self.cache_teb(idx)

        return np.load(tfname)

    def cache_teb(self, idx):
        tlm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_tlm.npy"
        elm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_elm.npy"
        blm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_blm.npy"

        assert( not any(os.path.exists(fname) for fname in [tlm_fname, elm_fname, blm_fname] ) )

        tlm, elm, blm = self.simulate(idx)

        np.save(tlm_fname, tlm)
        np.save(elm_fname, elm)
        np.save(blm_fname, blm)

    def simulate(self, idx):
        assert(0)

class teblm_library(object):
    # library of temperature and polarization alms.

    def __init__(self, lib_dir):
        assert(lib_dir != None)
        
        self.lib_dir = lib_dir

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        if not  os.path.exists(lib_dir + "/sim_hash.pk"):
            pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
        util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        return {}

    def get_sim_tlm(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_tlm.npy"

        if not os.path.exists(tfname):
            self.cache_teb(idx)

        return np.load(tfname)

    def get_sim_elm(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_elm.npy"

        if not os.path.exists(tfname):
            self.cache_teb(idx)

        return np.load(tfname)

    def get_sim_blm(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_blm.npy"

        if not os.path.exists(tfname):
            self.cache_teb(idx)

        return np.load(tfname)

    def cache_teb(self, idx):
        tlm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_tlm.npy"
        elm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_elm.npy"
        blm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_blm.npy"

        assert( not any(os.path.exists(fname) for fname in [tlm_fname, elm_fname, blm_fname] ) )

        tlm, elm, blm = self.simulate(idx)

        np.save(tlm_fname, tlm)
        np.save(elm_fname, elm)
        np.save(blm_fname, blm)

    def simulate(self, idx):
        assert(0)

class teblm_cmb_library(teblm_library):
    # library of temperature alms, with or without lensing.

    def __init__(self, lmax, cl, lib_dir):
        self.lmax = lmax
        self.cl   = cl

        super( teblm_cmb_library, self ).__init__( lib_dir=lib_dir )

    def hashdict(self):
        return { 'lmax'    : self.lmax,
                 'cltt'    : hashlib.sha1(self.cl.cltt.view(np.uint8)).hexdigest(),
                 'clte'    : hashlib.sha1(self.cl.clte.view(np.uint8)).hexdigest(),
                 'clee'    : hashlib.sha1(self.cl.clee.view(np.uint8)).hexdigest(),
                 'clbb'    : hashlib.sha1(self.cl.clbb.view(np.uint8)).hexdigest(),
                 'lib_dir' : self.lib_dir,
                 'super'   : super( teblm_cmb_library, self ).hashdict() }

    def simulate(self, idx):
        tlm, elm, blm = hp.synalm( [self.cl.cltt, self.cl.clte, self.cl.clee, self.cl.clbb], lmax=self.lmax )
        return tlm, elm, blm

class teblm_cmb_claa_library(teblm_library):
    # library of temperature alms, with or without lensing.

    def __init__(self, claa, nside, sim_lib, lib_dir):
        self.claa    = claa # alpha-alpha power spectrum in units of rad^2.
        self.nside   = nside
        self.sim_lib = sim_lib

        super( teblm_cmb_claa_library, self ).__init__( lib_dir=lib_dir )

    def hashdict(self):
        return { 'claa'    : hashlib.sha1(self.claa.view(np.uint8)).hexdigest(),
                 'nside'   : self.nside,
                 'sim_lib' : self.sim_lib.hashdict(),
                 'lib_dir' : self.lib_dir,
                 'super'   : super( teblm_cmb_claa_library, self ).hashdict() }

    def get_sim_alm(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_alm.npy"

        if not os.path.exists(tfname):
            self.cache_teb(idx)

        return np.load(tfname)

    def cache_teb(self, idx):
        tlm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_tlm.npy"
        elm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_elm.npy"
        blm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_blm.npy"
        alm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_alm.npy"

        assert( not any(os.path.exists(fname) for fname in [tlm_fname, elm_fname, blm_fname, alm_fname] ) )

        tlm, elm, blm, alm = self.simulate(idx)

        np.save(tlm_fname, tlm)
        np.save(elm_fname, elm)
        np.save(blm_fname, blm)
        np.save(alm_fname, alm)

    def simulate(self, idx):
        lmax = self.sim_lib.lmax
        amap, alm = hp.synfast( self.claa, self.nside, lmax=lmax, alm=True )

        print "simulating rot, amap rms = ", np.std(amap)*180./np.pi
        q_nul, u_nul = hp.alm2map_spin( (self.sim_lib.get_sim_elm(idx), self.sim_lib.get_sim_blm(idx)),
                                        self.nside, 2, lmax=lmax )
        p_rot = (q_nul + 1.j*u_nul)*(np.cos(2.*amap) + 1.j*np.sin(2.*amap))
        del q_nul, u_nul

        elm_rot, blm_rot = hp.map2alm_spin( (p_rot.real, p_rot.imag), 2, lmax=lmax )

        return self.sim_lib.get_sim_tlm(idx), elm_rot, blm_rot, alm

class tpmap_library(object):
    # library of sims.

    def __init__(self, lib_dir):
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
        return {}

    def get_sim_tmap(self, det, idx):
        tfname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_tmap.npy"
        if not os.path.exists(tfname): self.cache_sim_tp(det, idx)
        return np.load(tfname)

    def get_sim_pmap(self, det, idx):
        tfname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_pmap.npy"
        if not os.path.exists(tfname): self.cache_tp(idx)
        return np.load(tfname)
    
    def cache_sim_tp(self, det, idx):
        tmap_fname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_tmap.npy"
        pmap_fname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_pmap.npy"

        assert( not any(os.path.exists(fname) for fname in [tmap_fname, pmap_fname] ) )
            
        tmap, pmap = self.simulate(det, idx)
        
        np.save(tmap_fname, tmap)
        np.save(pmap_fname, pmap)
    
    def simulate(self, det, idx):
        assert(0)

# ==

class tpmap_homog_nse_library(tpmap_library):
    def __init__(self, lmax, nside, bl, noiseT_uK_arcmin, noiseP_uK_arcmin, sim_teblm_cmb, lib_dir):
        self.lmax             = lmax
        self.nside            = nside
        self.bl               = bl
        self.noiseT_uK_arcmin = noiseT_uK_arcmin
        self.noiseP_uK_arcmin = noiseP_uK_arcmin
        self.sim_teblm_cmb    = sim_teblm_cmb

        super( tpmap_homog_nse_library, self ).__init__( lib_dir=lib_dir )

    def hashdict(self):
        return { 'lmax'             : self.lmax,
                 'nside'            : self.nside,
                 'bl'               : hashlib.sha1(self.bl.view(np.uint8)).hexdigest(),
                 'noiseT_uK_arcmin' : self.noiseT_uK_arcmin,
                 'noiseP_uK_arcmin' : self.noiseP_uK_arcmin,
                 'sim_teblm_cmb'    : self.sim_teblm_cmb.hashdict(),
                 'lib_dir'          : self.lib_dir,
                 'super'            : super( tpmap_homog_nse_library, self ).hashdict() }

    def get_dat_tmap(self, det):
        return self.get_sim_tmap(det, -1)

    def get_dat_pmap(self, det):
        return self.get_sim_pmap(det, -1)

    def simulate(self, det, idx):
        assert(det == '')
        
        tlm = self.sim_teblm_cmb.get_sim_tlm(idx)
        elm = self.sim_teblm_cmb.get_sim_elm(idx)
        blm = self.sim_teblm_cmb.get_sim_blm(idx)

        beam = self.bl[0:self.lmax+1] * hp.pixwin(self.nside)[0:self.lmax+1]
        hp.almxfl(tlm, beam, inplace=True)
        hp.almxfl(elm, beam, inplace=True)
        hp.almxfl(blm, beam, inplace=True)
        
        tmap = hp.alm2map(tlm, self.nside)
        qmap, umap = hp.alm2map_spin( (elm, blm), self.nside, 2, lmax=self.lmax )

        npix = 12*self.nside**2
        tmap += np.random.standard_normal(npix) * (self.noiseT_uK_arcmin * np.sqrt(npix / 4. / np.pi) * np.pi / 180. / 60.)
        qmap += np.random.standard_normal(npix) * (self.noiseP_uK_arcmin * np.sqrt(npix / 4. / np.pi) * np.pi / 180. / 60.)
        umap += np.random.standard_normal(npix) * (self.noiseP_uK_arcmin * np.sqrt(npix / 4. / np.pi) * np.pi / 180. / 60.)

        return tmap, qmap + 1.j*umap
