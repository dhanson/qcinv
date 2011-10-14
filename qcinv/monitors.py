import sys
import numpy as np

import util

## loggers
class flat_logger():
    def __init__(self, mesg):
        self.watch = util.stopwatch()
        self.mesg  = mesg

    def __call__(self, iter, eps, **kwargs):
        print self.mesg, \
              ' [', self.watch.elapsed(), '] iter = ', '%03d' % iter, \
              ' flat eps = ', kwargs['flat_delta']/kwargs['d0_flat'], ' cg eps = ', kwargs['cg_delta']/kwargs['d0_cg']

        f_handle = file('outputs/flat_soltns.dat', 'a')
        np.savetxt(f_handle, [[v for (l,v) in kwargs['flat_soltn_cl']]])
        f_handle.close()

        f_handle = file('outputs/flat_resid.dat', 'a')
        np.savetxt(f_handle, [[v for (l,v) in kwargs['flat_resid_cl']]])
        f_handle.close()
