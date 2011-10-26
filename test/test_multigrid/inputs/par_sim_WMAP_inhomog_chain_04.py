import imp
import numpy as np

import qcinv

tr_cg   = qcinv.cd_solve.tr_cg

# --- parameters ---

run_script = imp.load_source('run_script', 'scripts/run_sim_WMAP_inhomog.py')

nside      = run_script.nside
lmax       = run_script.lmax

sim_prefix = "outputs/sim_WMAP_inhomog/"
out_prefix = "outputs/sim_WMAP_inhomog/chain_04/"

#                   id     preconditioners                      lmax    nside       im        em      tr      cache
chain_descr  =  [ [  3,    ["split(dense,     64, diag_cl)"],    128,     64,        3,      0.0,    tr_cg,   qcinv.cd_solve.cache_mem()],
                  [  2,    ["split(stage(3), 128, diag_cl)"],    256,    128,        3,      0.0,    tr_cg,   qcinv.cd_solve.cache_mem()],
                  [  1,    ["split(stage(2), 256, diag_cl)"],    512,    256,        3,      0.0,    tr_cg,   qcinv.cd_solve.cache_mem()],
                  [  0,    ["split(stage(1), 512, diag_cl)"],    1000,   512,   np.inf,   1.0e-6,    tr_cg,   qcinv.cd_solve.cache_mem()] ]
