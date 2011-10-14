import numpy as np

def cg_solve_simple( x, b, fwd_op, pre_op, dot_op, iter_max=1000, eps_min=1.0e-5, roundoff=50 ):
    # simple conjugate gradient loop, demonstrating use of cg_iterator.
    my_cg_iterator = cg_iterator( x, b, fwd_op, pre_op, dot_op, roundoff )

    #initialize.
    (delta, resid) = my_cg_iterator.next()
    d0 = delta

    #loop
    for iter in xrange(1,iter_max+1):
        (delta, resid) = my_cg_iterator.next()

        if (delta < eps_min**2 * d0): break

    return (iter, delta/d0)

def cg_solve(soltn, b, fwd_op, pre_op, dot_op, criterion, apply_prep_op=None, apply_fini_op=None, roundoff=50):
    # fully customizable conjugate gradient loop.
    if (apply_prep_op is not None): apply_prep_op(b)
    
    cg_iter        = cg_iterator( soltn, b, fwd_op, pre_op, dot_op )
    (delta, resid) = cg_iter.next()

    iter = 0
    while criterion(iter, soltn, resid, delta) == False:
        (delta, resid) = cg_iter.next()
        iter += 1

    if (apply_fini_op is not None): apply_fini_op(soltn)

def cg_iterator( x, b, fwd_op, pre_op, dot_op, roundoff=50 ):
    # conjugate gradient iterator for x=[fwd_op]^{-1}b
    # initial value of x is taken as guess
    # fwd_op, pre_op and dot_op must not modify their arguments!
    residual  = b - fwd_op(x)
    searchdir = pre_op(residual)

    delta     = dot_op(residual, searchdir)
    
    iter = 0
    while True:
        assert( delta >= 0.0 ) #sanity check
        yield (delta, residual)
        
        searchfwd = fwd_op(searchdir)
        alpha     = delta / dot_op(searchdir, searchfwd)

        x += (searchdir * alpha)

        iter += 1
        if ( np.mod(iter, roundoff) == 0 ):
            residual = b - fwd_op(x)
        else:
            residual -= searchfwd * alpha

        tsearchdir = pre_op(residual)
        tdelta     = dot_op(residual, tsearchdir)

        searchdir *= (tdelta / delta)
        searchdir += tsearchdir

        delta = tdelta
