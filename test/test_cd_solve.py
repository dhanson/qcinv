#!/usr/bin/env python

import sys
import numpy as np

import qcinv

class test_fwd():
    def __init__(self, mat):
        self.mat = mat
        
    def __call__(self, var):
        return np.dot(self.mat, var)

class test_pre_idt(test_fwd):
    #identity matrix preconditioner
    def __init__(self, mat):
        self.mat = np.identity(np.shape(mat)[0])

class test_pre_inv(test_fwd):
    # the best preconditioner ever
    def __init__(self, mat):
        self.mat = np.linalg.inv(mat)

class test_pre_axd():
    def __init__(self, mat):
        self.shape = np.shape(mat)
        self.iter  = 0

    def __call__(self, var):
        self.iter = np.mod( self.iter, self.shape[0] )

        ret = np.zeros( self.shape )
        ret[self.iter, self.iter] = 1.0
        self.iter += 1
        
        return np.dot( ret, var )

# testing.

if (__name__ == "__main__"):
    dim = 100
    np.random.seed(10)

    # random symmetric positive definite matrix
    mat = np.array(np.random.standard_normal([dim,dim]))
    mat = np.dot(np.transpose(mat), mat)
    
    mat = np.diag( np.arange(1.0,dim+1.0) )

    b  = np.array(np.random.standard_normal(dim))
    x = np.zeros(dim)
    
    fwd_op = test_fwd(mat)
    pre_op_idt = test_pre_idt(mat)
    pre_op_inv = test_pre_inv(mat)
    pre_op_axd = test_pre_axd(mat)

    monitor = qcinv.cd_monitors.monitor_basic( np.dot, iter_max=10000, eps_min=1.0e-10, logger=None )

    # test inv. preconditioner
    print 'test 0) perfect preconditioner --'
    tr = (lambda i : i-1)
    x_cd = np.zeros(dim)
    iter = qcinv.cd_solve.cd_solve( x_cd, b, fwd_op, [pre_op_inv], np.dot, monitor, tr, roundoff=1 )
    print 'cd solve: (inv. only) iter = ', iter

    x_cd = np.zeros(dim)
    iter = qcinv.cd_solve.cd_solve( x_cd, b, fwd_op, [pre_op_inv, pre_op_idt], np.dot, monitor, tr, roundoff=1 )
    print 'cd solve: (inv + idt) iter = ', iter

    # test vs. CG
    print 'test 1) CD vs. CG --'
    tr = (lambda i : i-1)
    x_cd = np.zeros(dim)
    iter = qcinv.cd_solve.cd_solve( x_cd, b, fwd_op, [pre_op_idt], np.dot, monitor, tr, roundoff=1 )
    print 'cd solve: iter = ', iter

    x_cg = np.zeros(dim)
    (iter, eps) = qcinv.cg_solve.cg_solve_simple( x_cg, b, fwd_op, pre_op_idt, np.dot,
                                                  iter_max=1000, eps_min=1.0e-10, roundoff=1 )
    print 'cg solve: iter = ', iter
    print 'rms. frac. diff is ', np.sqrt( np.sum( (x_cd - x_cg)**2 ) / 0.5 / ( np.sum( x_cd**2 + x_cg**2 )) )

    # test vs. 1/2 CG
    print 'test 2) 1/2 CD vs. CG --'
    tr = (lambda i : i-1 )
    x_cd = np.zeros(dim)
    iter = qcinv.cd_solve.cd_solve( x_cd, b, fwd_op, [pre_op_idt, fwd_op], np.dot, monitor, tr, roundoff=1 )
    print 'cd_solve: iter = ', iter 
    print 'rms frac. diff. is ', np.sqrt( np.sum( (x_cd - x_cg)**2 ) / 0.5 / ( np.sum( x_cd**2 + x_cg**2 )) )

    # test gram-schmidt.
    print 'test 3) gram-schmidt --'
    tr = (lambda i : 0)
    x_cd = np.zeros(dim)
    iter = qcinv.cd_solve.cd_solve( x_cd, b, fwd_op, [pre_op_axd], np.dot, monitor, tr, roundoff=1 )

    print 'cd_solve completed. iter (should be '+ str(dim)+ ') = ', iter
