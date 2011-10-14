#!/usr/bin/env python

import sys
import numpy as np

import qcinv

class test_fwd():
    def __init__(self, mat):
        self.mat = mat
        
    def do(self, var):
        return np.dot(self.mat, var)

class test_pre_idt(test_fwd):
    #identity matrix preconditioner
    def __init__(self, mat):
        self.mat = np.identity(np.shape(mat)[0])

class test_pre_rnd(test_fwd):
    # a random symmetric +def matrix preconditioner.
    def __init__(self, mat):
        tmat = np.array(np.random.standard_normal(np.shape(mat)))
        self.mat = np.dot(np.transpose(tmat), tmat)

class test_pre_inv(test_fwd):
    # the best preconditioner ever
    def __init__(self, mat):
        self.mat = np.linalg.inv(mat) 

# testing.

if (__name__ == "__main__"):
    dim = 100
    np.random.seed(10)

    # generate a random symmetric positive definite matrix
    mat = np.array(np.random.standard_normal([dim,dim]))
    mat = np.dot(np.transpose(mat), mat) 
    
    fwd_op = test_fwd(mat).do
    pre_op = test_pre_idt(mat).do

    b = np.array(np.random.standard_normal(dim))
    x = np.zeros(dim)
    
    (iter, eps) = qcinv.cg_solve.cg_solve_simple( x, b, fwd_op, pre_op, np.dot,
                                                  iter_max=1000, eps_min=1.0e-10 )
    
    print 'cgsolve completed. iter = ', iter, ' eps = ', eps
    sol = np.dot( np.linalg.inv(mat), b )
    err = np.abs( x - sol )
    print 'max,min soltn. = (', np.max(np.abs(sol)), ', ', np.min(np.abs(sol)), ')'
    print 'max,min error  = (', np.max(err), ', ', np.min(err), ')'
    print '"" frac.error  = (', np.max(err / np.abs(sol)), ', ', np.min(err / np.abs(sol)), ')'
    print 'average error  = ',  np.average(err)
