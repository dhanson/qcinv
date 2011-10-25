import imp
import numpy as np

import qcinv

# --- parameters ---

run_script = imp.load_source('run_script', 'scripts/run_sim_WMAP_inhomog.py')

nside      = run_script.nside
lmax       = run_script.lmax

sim_prefix = "outputs/sim_WMAP_inhomog/"
out_prefix = "outputs/sim_WMAP_inhomog/chain_01/"

# entirely diag.
chain_descr  =  [ [  0,    ["diag_cl"],    lmax,   nside,  np.inf,  1.0e-6,   qcinv.cd_solve.tr_cg,    qcinv.cd_solve.cache_mem()] ]
