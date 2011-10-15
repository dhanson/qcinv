import sys, os, re, glob
import numpy as np

import util
import cd_solve
import cd_monitors
    
class pre_op_split():
    def __init__(self, lsplit, lmax, pre_op_low, pre_op_hgh):
        self.lsplit = lsplit
        self.lmax   = lmax
        
        self.pre_op_low = pre_op_low
        self.pre_op_hgh = pre_op_hgh

        self.iter   = 0

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        self.iter += 1

        talm_low = self.pre_op_low(util.alm_copy(talm, lmax=self.lsplit))
        talm_hgh = self.pre_op_hgh(util.alm_copy(talm, lmax=self.lmax))

        return util.alm_splice(talm_low, talm_hgh, self.lsplit)

class pre_op_multigrid():
    def __init__(self, opfilt, lmax, nside, s_cls, n_inv_filt, pre_ops,
                 logger, tr, cache, iter_max, eps_min ):

        self.opfilt   = opfilt
        self.fwd_op   = opfilt.fwd_op(s_cls, n_inv_filt)
        
        self.lmax     = lmax
        self.nside    = nside

        self.s_cls    = s_cls
        self.pre_ops  = pre_ops

        self.logger   = logger

        self.tr       = tr
        self.cache    = cache
        
        self.iter_max = iter_max
        self.eps_min  = eps_min

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        monitor = cd_monitors.monitor_basic(self.opfilt.dot_op, iter_max=self.iter_max, eps_min=self.eps_min, logger=self.logger)

        soltn = np.zeros(util.lmax2nlm(self.lmax), dtype=np.complex)
        
        cd_solve.cd_solve( soltn, util.alm_copy(talm, lmax=self.lmax),
                           self.fwd_op, self.pre_ops, self.opfilt.dot_op,
                           monitor, tr=self.tr, cache=self.cache )

        return util.alm_splice(soltn, talm, self.lmax)

def parse_pre_op_descr(pre_op_descr, **kwargs):
    if re.match("split\((.*),\s*(.*),\s*(.*)\)\Z", pre_op_descr):
        (low_descr, lsplit, hgh_descr) = re.match("split\((.*),\s*(.*),\s*(.*)\)\Z", pre_op_descr).groups()
        print 'creating split preconditioner ', (low_descr, lsplit, hgh_descr)

        kwargs['lsplit'] = int(lsplit)
        pre_op_low = parse_pre_op_descr( low_descr, **kwargs )
        pre_op_hgh = parse_pre_op_descr( hgh_descr, **kwargs )
        
        return pre_op_split( int(lsplit), kwargs['lmax'], pre_op_low, pre_op_hgh )
    elif re.match("diag_cl\Z", pre_op_descr):
        return kwargs['opfilt'].pre_op_diag( kwargs['s_cls'], kwargs['n_inv_filt'] )
    elif re.match("stage\(.*\)\Z", pre_op_descr):
        (stage_id,) = re.match("stage\((.*)\)\Z", pre_op_descr).groups()
        print 'creating multigrid preconditioner: stage_id = ', stage_id
        
        stage    = kwargs['stages'][int(stage_id)]
        logger   = (lambda iter, eps, stage=stage, chain=kwargs['chain'], **kwargs :
                    chain.log(stage, iter, eps, **kwargs))

        assert( stage.lmax == kwargs['lsplit'] )
        
        return pre_op_multigrid(kwargs['opfilt'], stage.lmax, stage.nside,
                                kwargs['s_cls'], kwargs['n_inv_filt'].degrade(stage.nside),
                                stage.pre_ops, logger, stage.tr, stage.cache,
                                stage.iter_max, stage.eps_min)
    else:
        print 'pre_op_descr = ', pre_op_descr, ' is unrecognized!'
        assert(0)

class multigrid_chain():
    def __init__(self, opfilt, chain_descr, s_cls, n_inv_filt, debug_log_prefix=None):
        self.debug_log_prefix = debug_log_prefix

        self.opfilt         = opfilt
        self.chain_descr    = chain_descr

        self.s_cls          = s_cls
        self.n_inv_filt     = n_inv_filt

        self.memloaded      = False
        
    def load_mem( self ):
        if self.memloaded == True:
            return

        self.n_inv_filt.load_mem()
        
        stages = {}
        for [id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache] in self.chain_descr:
            stages[id] = util.attr_dict( dict(
                zip( ['depth', 'pre_ops_descr', 'lmax', 'nside', 'iter_max', 'eps_min', 'tr', 'cache', 'pre_ops'],
                     [id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache, [] ] ) ) )
            
            for pre_op_descr in pre_ops_descr:
                stages[id].pre_ops.append( parse_pre_op_descr(pre_op_descr, opfilt=self.opfilt,
                                                              s_cls=self.s_cls, n_inv_filt=self.n_inv_filt,
                                                              stages=stages, lmax=lmax, chain=self) )
        self.bstage = stages[0]

        self.memloaded = True

    def solve( self, soltn, tpn_map ):
        self.load_mem()

        self.watch = util.stopwatch()

        self.iter_tot   = 0        
        self.prev_eps   = None

        logger = (lambda iter, eps, stage=self.bstage, **kwargs :
                  self.log(stage, iter, eps, **kwargs))

        monitor = cd_monitors.monitor_basic(self.opfilt.dot_op, logger=logger, iter_max=self.bstage.iter_max, eps_min=self.bstage.eps_min)

        tpn_alm = self.opfilt.calc_prep_map(tpn_map, self.s_cls, self.n_inv_filt)

        fwd_op  = self.opfilt.fwd_op(self.s_cls, self.n_inv_filt)

        cd_solve.cd_solve( soltn, tpn_alm,
                           fwd_op, self.bstage.pre_ops, self.opfilt.dot_op, monitor,
                           tr=self.bstage.tr, cache=self.bstage.cache )

        self.opfilt.apply_fini( soltn, self.s_cls, self.n_inv_filt )

    def log(self, stage, iter, eps, **kwargs):
        self.iter_tot += 1
        elapsed = self.watch.elapsed()

        if stage.depth != 0:
            return

        log_str = ('   ')*stage.depth + '(%4d, %04d) [%s] (%d, %f)' % (stage.nside, stage.lmax, str(elapsed), iter, eps) + '\n'
        sys.stdout.write(log_str)

        if (self.debug_log_prefix is not None):
            log = open( self.debug_log_prefix + 'stage_all.dat', 'a')
            log.write( log_str )
            log.close()

            if (stage.depth == 0):
                f_handle = file(self.debug_log_prefix + 'stage_soltn_' + str(stage.depth) + '.dat', 'a')
                np.savetxt(f_handle, [[v for v in kwargs['soltn_cl']]])
                f_handle.close()

                f_handle = file(self.debug_log_prefix + 'stage_resid_' + str(stage.depth) + '.dat', 'a')
                np.savetxt(f_handle, [[v for v in kwargs['resid_cl']]])
                f_handle.close()

            log_str = '%05d %05d %10.6e %05d %s\n' % (self.iter_tot, int(elapsed), eps, iter, str(elapsed))
            log = open( self.debug_log_prefix + 'stage_' + str(stage.depth) + '.dat', 'a' )
            log.write( log_str )
            log.close()

            if ((self.prev_eps is not None) and (self.prev_stage.depth > stage.depth)):
                log_final_str = '%05d %05d %10.6e %s\n' % (self.iter_tot - 1, int(self.prev_elapsed), self.prev_eps, str(self.prev_elapsed))
                
                log = open(self.debug_log_prefix + 'stage_final_' + str(self.prev_stage.depth) + '.dat', 'a')
                log.write( log_final_str )
                log.close()

            self.prev_stage = stage
            self.prev_eps   = eps
            self.prev_elapsed = elapsed
