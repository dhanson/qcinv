#!/usr/bin/env python

import imp, os, sys, getopt

import numpy  as np
import healpy as hp

import qcinv

# -- load parameters
usage = "usage: " + sys.argv[0] + " <params.py>"
try:
    opts, args = getopt.getopt(sys.argv[1:], 'p')
except getopt.GetoptError:
    print usage
    exit()
if (len(args) != 1):
    print usage
    exit()

par = imp.load_source('par', args[0])
# --

# sanity tests.
if os.path.exists(par.out_prefix):
    print 'directory ', par.out_prefix, ' already exists.'
    #assert(0)
os.mkdir(par.out_prefix)

# restore simulation.
cltt  = np.loadtxt(par.sim_prefix + "cltt.dat")
clnn  = np.loadtxt(par.sim_prefix + "clnn.dat")
beam  = np.loadtxt(par.sim_prefix + "beam.dat")

class cl(object):
    pass
s_cls = cl
s_cls.cltt = cltt

ninv  = hp.read_map(par.sim_prefix + "ninv.fits")
dmap  = hp.read_map(par.sim_prefix + "dmap.fits")

lmax  = len(beam)-1
nside = hp.npix2nside(len(ninv))

# construct the chain.
# note: just-in-time instantiation here is useful e.g.
#       if n_inv_filt and chain are defined in a parameter
#       file and don't necessarily need to be loaded.
n_inv_filt = qcinv.util.jit( qcinv.opfilt_tt.alm_filter_ninv, ninv, beam, marge_monopole=True, marge_dipole=True, marge_maps=[] )
chain = qcinv.util.jit( qcinv.multigrid.multigrid_chain, qcinv.opfilt_tt, par.chain_descr, s_cls, n_inv_filt, debug_log_prefix=(par.out_prefix + 'log_') )

soltn = np.zeros( qcinv.util_alm.lmax2nlm(lmax), dtype=np.complex )

print 'RUNNING SOLVE'
chain.solve( soltn, dmap )

hp.almxfl(soltn, cltt, inplace=True)
np.save(par.out_prefix + 'soltn_wf_alm.npy', soltn)
