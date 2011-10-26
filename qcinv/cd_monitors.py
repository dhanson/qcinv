import sys
import numpy as np

import util
import util_alm

## monitors
logger_basic = (lambda iter, eps, watch=None, **kwargs : sys.stdout.write( '[' + str(watch.elapsed()) + '] ' + str((iter, eps)) + '\n' ))
logger_none  = (lambda iter, eps, watch=None, **kwargs : 0)
class monitor_basic():
    def __init__(self, dot_op, iter_max=1000, eps_min=1.0e-10, logger=logger_basic):
        self.dot_op   = dot_op
        self.iter_max = iter_max
        self.eps_min  = eps_min
        self.logger   = logger

        self.watch = util.stopwatch()

    def criterion(self, iter, soltn, resid):
        delta = self.dot_op( resid, resid )
        
        if (iter == 0):
            self.d0 = delta

        if (self.logger is not None): self.logger( iter, np.sqrt(delta/self.d0), watch=self.watch,
                                                   soltn_cl=util_alm.alm_cl(soltn), resid_cl=util_alm.alm_cl(resid) )

        if (iter >= self.iter_max) or (delta <= self.eps_min**2 * self.d0):
            return True

        return False

    def __call__(self, *args):
        return self.criterion(*args)
