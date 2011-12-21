import os, hashlib, math
import pickle as pk
import numpy  as np
import healpy as hp

import util
import wmap
import sims

class tpmap_coadd_library(sims.tpmap_library):
    # a library of noise realizations generated from a map of N^{-1}
    # on top of CMB realizations retrieved from a sim_teblm_cmb,
    # convolved with a symmetric beam.
    #

    def __init__(self, lmax, nside, year, sim_teblm_cmb, forered, ntype, lib_dir):
        assert( lmax  == 1000 )
        assert( nside == 512 )
        
        self.lmax           = lmax
        self.nside          = nside
        
        self.year           = year
        self.sim_teblm_cmb  = sim_teblm_cmb
        self.forered        = forered
        self.ntype          = ntype
    
        super( tpmap_coadd_library, self ).__init__( lib_dir=lib_dir )

    def hashdict(self):
        return { 'lmax'          : self.lmax,
                 'nside'         : self.nside,
                 'year'          : self.year,
                 'sim_teblm_cmb' : self.sim_teblm_cmb.hashdict(),
                 'forered'       : self.forered,
                 'nytpe'         : self.ntype,
                 'lmax'          : self.lmax,
                 'nside'         : self.nside,
                 'super'         : super( tpmap_coadd_library, self ).hashdict() }

    def get_dat_tmap(self, det):
        return hp.read_map( wmap.get_fname_iqumap(self.year, det, self.forered), hdu=1, field=0 ) * 1.e3

    def get_dat_pmap(self, det):
        return ( hp.read_map( wmap.get_fname_iqumap(self.year, det, self.forered), hdu=1, field=1 ) +
                 hp.read_map( wmap.get_fname_iqumap(self.year, det, self.forered), hdu=1, field=2 ) * 1.j ) * 1.e3

    def get_beam(self, det):
        pxw  = hp.pixwin(self.nside)[0:self.lmax+1]
        beam = wmap.get_bl(self.year, det)[0:self.lmax+1]

        return beam * pxw

    def simulate(self, det, idx):
        assert( det in (wmap.wmap_das + wmap.wmap_bands) )
        
        tlm = self.sim_teblm_cmb.get_sim_tlm(idx)
        elm = self.sim_teblm_cmb.get_sim_elm(idx)
        blm = self.sim_teblm_cmb.get_sim_blm(idx)
        
        hp.almxfl(tlm, self.get_beam(det), inplace=True)
        hp.almxfl(elm, self.get_beam(det), inplace=True)
        hp.almxfl(blm, self.get_beam(det), inplace=True)
            
        tmap = hp.alm2map(tlm, self.nside)
        qmap, umap = hp.alm2map_spin( (elm, blm), self.nside, 2, lmax=self.lmax )

        tmap_nse, qmap_nse, umap_nse = self.simulate_noise(det)
        tmap += tmap_nse; del tmap_nse
        qmap += qmap_nse; del qmap_nse
        umap += umap_nse; del umap_nse
        
        return tmap, (qmap + 1.j*umap)

    def simulate_noise(self, det):
        if self.ntype == 'nobs':
            rtnobsinv = 1.0 / np.sqrt( hp.read_map( wmap.get_fname_iqumap(self.year, det, self.forered), hdu=1, field=3 ) )
            npix = 12*self.nside**2
            
            t_nse = rtnobsinv * wmap.sigma0[(self.year, self.forered, 'T')][det] * 1e3 * np.random.standard_normal( npix )
            q_nse = rtnobsinv * wmap.sigma0[(self.year, self.forered, 'P')][det] * 1e3 * np.random.standard_normal( npix ) * np.sqrt(2.)
            u_nse = rtnobsinv * wmap.sigma0[(self.year, self.forered, 'P')][det] * 1e3 * np.random.standard_normal( npix ) * np.sqrt(2.)
            
        elif self.ntype == 'nobs_qucov':
            tfname =  wmap.get_fname_iqumap(self.year, det, False) #NOTE: Nobs counts taken from non-FG reduced maps.
            npix = 12*self.nside**2

            # temperature
            t_nse = (wmap.sigma0[(self.year, self.forered, 'T')][det] * 1e3 /
                     np.sqrt( hp.read_map( tfname, hdu=1, field=3 ) )) * np.random.standard_normal( npix )

            # polarization
            cov_inv_qq = hp.read_map( tfname, hdu=2, field=1 ) / (wmap.sigma0[(self.year, self.forered, 'P')][det] * 1.e3)**2
            cov_inv_qu = hp.read_map( tfname, hdu=2, field=2 ) / (wmap.sigma0[(self.year, self.forered, 'P')][det] * 1.e3)**2
            cov_inv_uu = hp.read_map( tfname, hdu=2, field=3 ) / (wmap.sigma0[(self.year, self.forered, 'P')][det] * 1.e3)**2

            detern = 1.0 / (cov_inv_qq * cov_inv_uu - cov_inv_qu**2)
            cov_qq = +cov_inv_uu * detern
            cov_qu = -cov_inv_qu * detern
            cov_uu = +cov_inv_qq * detern
            del cov_inv_qq, cov_inv_qu, cov_inv_uu, detern

            r1 = np.random.standard_normal( npix )
            r2 = np.random.standard_normal( npix )

            q_nse = r1 * np.sqrt(cov_qq)
            u_nse = r1 * cov_qu / np.sqrt(cov_qq) + r2 * np.sqrt(cov_uu - cov_qu**2 / cov_qq)
            del r1, r2
        else:
            assert(0)

        return t_nse, q_nse, u_nse

