# Contains the custom Nelder-Mead algorithm
import numpy as np
import sys
eps = sys.float_info.epsilon # For Amoeba xtol and tfol
import time
import pdb
import copy
from numba import njit, jit, prange
from pdb import set_trace as stop
import numba

# If custom_subspace is set, n_sub_calls is ignored.
# x0 is all parameters (varied and non)
# vlb, vub are also full arrays
# vp is the array of indices for which parameters are varied
def simps(init_pars, foo, max_f_evals=None, xtol=1E-4, ftol=1E-4, n_sub_calls=None, no_improv_break=3, args_to_pass=None):

    # Number of total parameters
    nx = len(init_pars)

    # Unpack arrays
    #x0, vlb, vub, vp = init_pars.unpack(fetch=("value", "minv", "maxv", "vary"))
    v = init_pars.unpack()
    x0, vlb, vub, vp = v['values'], v['minvs'], v['maxvs'], v['varies']

    # This is what's passed to the target function
    #pars = copy.deepcopy(init_pars)
    pars = copy.deepcopy(init_pars)

    # Get the indices
    vpi = np.where(vp)[0]

    # Number of varied parameters
    nxv = len(vpi)

    # Fill in remaining defaults
    if max_f_evals is None:
        max_f_evals = 500 * nxv

    if n_sub_calls is None:
        n_sub_calls = nxv
        
    if args_to_pass is None:
        args_to_pass = numba.types.List

    # Extract the varies params and bounds
    x0v = x0[vpi]
    vlbv = vlb[vpi]
    vubv = vub[vpi]

    # Perform sanity checks
    if nxv == 0:
        raise ValueError("No parameters found to optimize!")

    if np.any(vubv - vlbv < 0):
        raise ValueError("Params Out of Bounds")

    if np.any(x0v > vubv):
        raise ValueError("Params Out of Bounds")

    if np.any(x0v < vlbv):
        raise ValueError("Params Out of Bounds")

    # Constants
    step = 0.5
    nxp1 = nx + 1
    nxvp1 = nxv + 1
    penalty = 1E6

    # Initialize the simplex
    right = np.zeros(shape=(nxv, nxvp1), dtype=float)
    left = np.transpose(np.tile(x0v, (nxvp1, 1)))
    diag = np.diag(0.5 * x0v)
    right[:, :-1] = diag
    simplex = left + right

    # Initialize xmin, fmin and dx
    xmin = np.copy(x0)
    fmin = np.inf # breaks when less than ftol N times, updated
    dx = np.inf # breaks when less than xtol, updated
    fcalls = 0

    # keeps track of full calls to ameoba
    i = 0
    
    while i < n_sub_calls and dx >= xtol:

        # Perform Ameoba call for all parameters
        xdefault = np.copy(xmin) # the new default is the current min (varied and non varied)
        y, fmin, fcallst = ameoba(pars, simplex, foo, xdefault, vp, vlbv, vubv, no_improv_break, max_f_evals, ftol, penalty, args_to_pass=args_to_pass)
        fcalls += fcallst
        xmin[vp] = y # Only update all varied parameters, y only contains varied parameters
        
        # If theres < 2 params, a three-simplex is the smallest simplex used and only used once.
        if nxv < 3:
            break
        
        # Perform Ameoba call for dim=2 ( and thus three simplex) subspaces
        for j in range(nxv-1):
            
            # j1 is the subspace (indices in the simplex)
            j1 = np.array([j, j+1])
            
            # Subspace simplex
            simplex_sub = np.array([[x0v[j1[0]], x0v[j1[1]]], [y[j1[0]], y[j1[1]]], [x0v[j1[0]], y[j1[1]]]]).T
                                   
            # xdefault is the set of parameters used in the target function.
            # When the subspace is updated, xdefault is then updated
            xdefault = np.copy(xmin)
            
            # Call the solver
            y[j1], fmin, fcallst = ameoba(pars, simplex_sub, foo, xdefault, np.array([vpi[j1[0]], vpi[j1[1]]]), vlbv[j1], vubv[j1], no_improv_break, max_f_evals, ftol, penalty, args_to_pass=args_to_pass)
            
            # Update fcalls
            fcalls += fcallst
            
            # Update the full simplex with the new best fit pars
            simplex[:, -1] = y
            xmin[vp] = y

        # Perform the last pairs of points (first and last), same steps.
        j1 = np.array([0, -1])
        simplex_sub = np.array([[x0v[j1[0]], x0v[j1[1]]], [y[j1[0]], y[j1[1]]], [x0v[j1[0]], y[j1[1]]]]).T
        xdefault = np.copy(xmin)
        y[j1], fmin, fcallst = ameoba(pars, simplex_sub, foo, xdefault, np.array([vpi[j1[0]], vpi[j1[1]]]), vlbv[j1], vubv[j1], no_improv_break, max_f_evals, ftol, penalty, args_to_pass=args_to_pass)
        fcalls += fcallst
        
        simplex[:, -1] = y
        xmin[vp] = y
        
        # Increment the iteration
        i += 1
        
        # Compute the max absolute range of the simplex
        dx = np.max(tolx(np.min(simplex, axis=1), np.max(simplex, axis=1)))
        
    pars.setv(values=xmin, varies=vp)

    # Return best fit params, rms, and N function calls
    return np.array([pars, fmin, fcalls], dtype=object)

# Ameoba assumes that simplex has been modified for any unvaried parameters.
# simplex: The simplex of varied parameters, determined by subspace
# foo: the target function
# xdefault: the default values to use if not varying a parameter
# subspace: the array of par indices being varied
# vlb, vub: the lower and upper bounds, for current varied parameters. only varied parameters are compared against vlb/vub
# no_improv_break: how many times in a row convergence occurs before exiting
def ameoba(pars, simplex, foo, xdefault, subspace, vlb, vub, no_improv_break, max_f_evals, ftol, penalty, args_to_pass=None):

    # Constants
    n = np.min(simplex.shape)
    np1 = n + 1
    alpha = 1
    gamma = 2
    sigma = 0.5
    delta = 0.5

    # Stores the f values
    fvals = np.empty(np1, dtype=float)
    
    # Generate the fvals for the initial simplex
    for i in range(np1):
        fvals[i] = foo_wrapper(pars, simplex[:, i], foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
        
    # Number of functions calls
    fcalls = np1

    # Sort the fvals and then simplex
    ind = np.argsort(fvals)
    simplex = simplex[:, ind]
    fvals = fvals[ind]
    fmin = fvals[0]
    xmin = simplex[:, 0]
    n_small_steps = 0

    # init storage arrays
    x = np.empty(n, dtype=float)
    xr = np.empty(n, dtype=float)
    xbar = np.empty(n, dtype=float)
    xc = np.empty(n, dtype=float)
    xe = np.empty(n, dtype=float)
    xcc = np.empty(n, dtype=float)
    
    # Gradient Estimate
    xs = np.empty(n, dtype=float)
    g = np.empty(n, dtype=float)

    while True:

        # Sort the vertices according from best to worst
        # Define the worst and best vertex, and f(best vertex)
        xnp1 = simplex[:, -1]
        fnp1 = fvals[-1]
        x1 = simplex[:, 0]
        f1 = fvals[0]
        xn = simplex[:, -2]
        fn = fvals[-2]
        
        # Gradient Estimate
        #xs[:] = np.diagonal(simplex[:, :-1])
        #fs = foo_wrapper(xs)
        #g[:] = estimate_gradient(n, simplex, fvals, xs, fs)
            
        
        # Possibly updated
        shrink = 0

        # break after max_iter
        if fcalls >= max_f_evals:
            break
            
        # Break if f tolerance has been met n_small_steps in a row.
        if tolf(fmin, fnp1) > ftol:
            n_small_steps = 0
        else:
            n_small_steps += 1
        if n_small_steps >= no_improv_break:
            break

        # Idea of NM: Given a sorted simplex; N + 1 Vectors of N parameters,
        # We want to replace the worst point with a better point.
        
        # The "average" vector, V_i=par_i_avg
        # We first anchor points off this average Vector
        xbar[:] = np.average(simplex[:, :-1], axis=1)
        
        # The reflection point
        # For alpha = 1, this 
        xr[:] = xbar + alpha * (xbar - xnp1)
        
        x[:] = xr
        
        fr = foo_wrapper(pars, x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
        fcalls += 1

        if fr < f1:
            xe[:] = xbar + gamma * (xbar - xnp1)
            x[:] = xe
            fe = foo_wrapper(pars, x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
            fcalls += 1
            if fe < fr:
                simplex[:, -1] = xe
                fvals[-1] = fe
            else:
                simplex[:, -1] = xr
                fvals[-1] = fr
        elif fr < fn:
            simplex[:, -1] = xr
            fvals[-1] = fr
        else:
            if fr < fnp1:
                xc[:] = xbar + sigma * (xbar - xnp1)
                x[:] = xc
                fc = foo_wrapper(pars, x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
                fcalls += 1
                if fc <= fr:
                    simplex[:, -1] = xc
                    fvals[-1] = fc
                else:
                    shrink = 1
            else:
                xcc[:] = xbar + sigma * (xnp1 - xbar)
                x[:] = xcc
                fcc = foo_wrapper(pars, x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
                fcalls += 1
                if fcc < fvals[-1]:
                    simplex[:, -1] = xcc
                    fvals[-1] = fcc
                else:
                    shrink = 1
        if shrink > 0:
            for j in range(1, np1):
                simplex[:, j] = x1 + delta * (simplex[:, j] - x1)
                fvals[j] = foo_wrapper(pars, simplex[:, j], foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
            fcalls += n

        ind = np.argsort(fvals)
        fvals = fvals[ind]
        simplex = simplex[:, ind]
        fmin = fvals[0]
        xmin = simplex[:, 0]

    # Returns only the best varied parameters
    return xmin, fmin, fcalls


@njit(numba.types.float64[:](numba.types.float64[:], numba.types.float64[:]))
def tolx(a, b):
    c = (np.abs(b) + np.abs(a)) / 2
    c = np.atleast_1d(c)
    ind = np.where(c < eps)[0]
    if ind.size > 0:
        c[ind] = 1
    r = np.abs(b - a) / c
    return r

@njit(numba.types.float64(numba.types.float64, numba.types.float64))
def tolf(a, b):
    return np.abs(a - b)

#@njit(parallel=True)

#def estimate_gradient(n, simplex, fvals, xs, fs):
#    g = np.empty(n, dtype=float)
#    for i in prange(n):
#        if i%2 == 0:
#            g[i] = (fvals[i-1] - fs) / (simplex[i-1, i])
#        else:
#            g[i] = (fvals[i+1] - fs) / (simplex[i+1, i])
#    return g

def foo_wrapper(pars, xv, foo, xdefault, vlbv, vubv, subspace, penalty, args_to_pass):

    # Update the current varied subspace from the current simplex
    xdefault[subspace] = xv
    
    # Determine which parameters are being varied in the current simplex.
    varies = np.zeros(xdefault.size, dtype=bool)
    varies[subspace] = True

    v = np.zeros(xdefault.size, dtype=bool)
    v[subspace] = 1
    #pars.setvalues(xdefault)
    #pars.setvaries(v)
    pars.setv(values=xdefault, varies=v)
    
    # Call the target function
    f, c = foo(pars, *args_to_pass) # target function must be given full array of pars
    
    # Penalize the target function if pars are out of bounds or constraint is less than zero
    f += penalty*np.where((xv <= vlbv) | (xv >= vubv))[0].size
    f += penalty*(c < 0)
    return f