#!/usr/bin/env python

import os
import numpy  as np

import healpy as hp
import qcinv

out_prefix = "outputs/sim_WMAP_inhomog/"
if not os.path.exists(out_prefix):
    os.makedirs(out_prefix)

# --

nside = 512
lmax  = 1000

# --

npix = hp.nside2npix(nside)

if (__name__ == "__main__"):
    mask = hp.read_map("inputs/wmap/wmap_temperature_analysis_mask_r9_7yr_v4.fits")

    #theoretical power spectra.
    cl = qcinv.util.camb_clfile('inputs/wmap/bestfit_lensedCls.dat', lmax=lmax)
    
    # noise level
    # http://lambda.gsfc.nasa.gov/product/map/dr4/skymap_info.cfm
    ninv = hp.read_map("inputs/wmap/wmap_band_forered_iqumap_r9_7yr_V_v4.fits", field=3) / (3.137*1.0e3)**2
    assert( len(ninv) == npix )

    ncov  = np.zeros(npix)
    ncov[ np.where(ninv != 0) ] = 1.0 / ninv[ np.where(ninv != 0) ]

    assert( npix == len(ninv) )

    beam  = 0.5 * (np.loadtxt('inputs/wmap/wmap_ampl_bl_V1_7yr_v4.txt')[:,1][0:(lmax+1)] + np.loadtxt('inputs/wmap/wmap_ampl_bl_V2_7yr_v4.txt')[:,1][0:(lmax+1)])
    beam *= hp.pixwin(nside)[0:(lmax+1)]

    clnn  = np.sum(ncov) * (4.*np.pi) / npix**2 / beam**2

    ncov *= mask
    ninv *= mask

    # noise map
    print "generating noise map"
    nmap  = np.random.standard_normal(npix) * np.sqrt(ncov)

    # cmb map
    print "generating cmb map"
    tmap, talm = hp.synfast(cl.cltt * beam * beam, nside, lmax=lmax, alm=True)
    hp.almxfl(talm, 1.0/beam, inplace=True)

    dmap = (tmap + nmap) * mask

    # write everything out

    print "writing outputs"

    np.savetxt(out_prefix + "cltt.dat", cl.cltt)
    np.savetxt(out_prefix + "clnn.dat", clnn)
    np.savetxt(out_prefix + "beam.dat", beam)

    hp.write_map(out_prefix + "ninv.fits", ninv)
    hp.write_map(out_prefix + "dmap.fits", dmap)
