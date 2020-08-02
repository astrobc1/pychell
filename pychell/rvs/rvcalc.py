import os
from pdb import set_trace as stop

# Multiprocessing
from joblib import Parallel, delayed

# Maths
import numpy as np
import scipy.interpolate
import scipy.constants as cs

# LLVM
from numba import jit, njit, prange

# Graphics
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")
from robustneldermead.neldermead import NelderMead
import pychell.maths as pcmath
import pychell.utils as pcutils
import copy


def get_nightly_jds(jds, sep=0.5):
    """Computes nightly (average) JDs (or BJDs) for a time-series observation over several nights.

    Args:
        jds (np.ndarray): An array of sorted JDs (or BJDs).
        sep (float): The minimum separation in days between two different nights of data, defaults to 0.5.
    Returns:
        (np.ndarray): The average nightly jds.
        (np.ndarray): The number of observations each night.
    """
    
    # Number of spectra
    n_spec = len(jds)

    prev_i = 0
    # Calculate mean JD date and number of observations per night for nightly
    # coadded RV points; assume that observations on separate nights are
    # separated by at least 0.5 days.
    jds_nightly = []
    n_obs_nights = []
    if n_spec == 1:
        jds_nightly.append(jds[0])
        n_obs_nights.append(1)
    else:
        for i in range(n_spec-1):
            if jds[i+1] - jds[i] > sep:
                jd_avg = np.average(jds[prev_i:i+1])
                n_obs_night = i - prev_i + 1
                jds_nightly.append(jd_avg)
                n_obs_nights.append(n_obs_night)
                prev_i = i + 1
        jds_nightly.append(np.average(jds[prev_i:]))
        n_obs_nights.append(n_spec - prev_i)

    jds_nightly = np.array(jds_nightly) # convert to np arrays
    n_obs_nights = np.array(n_obs_nights).astype(int)

    return jds_nightly, n_obs_nights


def weighted_brute_force(forward_model, iter_index):
    """Performs a pseudo weighted cross-correlation via brute force RMS minimization and estimating the minumum with a quadratic.

    Args:
        forward_model (ForwardModel): A single forward model object.
        iter_index (int): The iteration to use to construct the forward model.
    Returns:
        (np.ndarray): The velocities for the xcorr function.
        (np.ndarray): The rms (xcorr) surface.
        (np.ndarray): The xcorr RV (bary-center velocity subtracted).
        (float): The BIS.
    """
    
    pars = copy.deepcopy(forward_model.best_fit_pars[-1])
    v0 = pars[forward_model.models_dict['star'].par_names[0]].value
    vels = np.linspace(v0 - forward_model.xcorr_options['range'], v0 + forward_model.xcorr_options['range'], num=forward_model.xcorr_options['n_vels'])

    # Stores the rms as a function of velocity
    rmss = np.empty(vels.size, dtype=np.float64) + np.nan
    
    # Starting weights are flux uncertainties and bad pixels. If flux unc are uniform, they have no effect.
    weights_init = np.copy(forward_model.data.badpix * forward_model.data.flux_unc)
    
    # Flag regions of heavy tellurics
    low_flux_regions = np.where(np.log(forward_model.data.flux) < -3)[0]
    if low_flux_regions.size > 0:
        weights_init[low_flux_regions] = 0
        
    for i in range(vels.size):
        
        weights = np.copy(weights_init)
        
        # Set the RV parameter to the current step
        pars[forward_model.models_dict['star'].par_names[0]].setv(value=vels[i])
        
        # Build the model
        _, model_lr = forward_model.build_full(pars, iter_index)
        
        # Weights
        bad = np.where(~np.isfinite(model_lr) | (weights <= 0))[0]
        if bad.size > 0:
            weights[bad] = 0
        
        # Construct the RMS
        rmss[i] = np.sqrt(np.nansum((forward_model.data.flux - model_lr)**2 * weights) / np.nansum(weights))

    # Extract the best rv
    xcorr_star_vel = vels[np.nanargmin(rmss)]
    xcorr_rv_init = xcorr_star_vel + forward_model.data.bc_vel # Actual RV is bary corr corrected
    vels_for_rv = vels + forward_model.data.bc_vel
        
    # Fit with a polynomial
    use = np.where((vels_for_rv > xcorr_rv_init - 150) & (vels_for_rv < xcorr_rv_init + 150))[0]
    pfit = np.polyfit(vels_for_rv[use], rmss[use], 2)
    xcorr_rv = -1 * pfit[1] / (2 * pfit[0])
    
    # Bisector span
    bspan_result = compute_bisector_span(vels_for_rv, rmss, n_bs=forward_model.xcorr_options['n_bs'])
    
    return vels_for_rv, rmss, xcorr_rv, bspan_result[1]


def crude_brute_force(forward_model, iter_index=None):
    """Performs a pseudo cross-correlation via brute force RMS minimization and estimating the minumum with a quadratic.

    Args:
        forward_model (ForwardModel): A single forward model object.
        iter_index (int): The iteration to use to construct the forward model. Not used, is None.
        
    Returns:
        forward_model (ForwardModel): The forward model object with cross-correlation results stored in place.
    """
    
    pars = copy.deepcopy(forward_model.initial_parameters)
    vels = np.arange(-250000, 250000, 500)

    # Stores the rms as a function of velocity
    rmss = np.empty(vels.size, dtype=np.float64) + np.nan
    
    # Weights are bad pixels
    weights_init = np.copy(forward_model.data.badpix)
    
    # Flag very low flux
    low_flux_regions = np.where(np.log(forward_model.data.flux) < -3)[0]
    good = np.where(forward_model)
    if low_flux_regions.size > 0:
        weights_init[low_flux_regions] = 0
        
    for i in range(vels.size):
        
        weights = np.copy(weights_init)
        
        # Set the RV parameter to the current step
        pars[forward_model.models_dict['star'].par_names[0]].setv(value=vels[i])
        
        # Build the model
        _, model_lr = forward_model.build_full(pars, iter_index)
        
        # Weights
        bad = np.where(~np.isfinite(model_lr) | (weights <= 0))[0]
        if bad.size > 0:
            weights[bad] = 0
        
        # Construct the RMS
        rmss[i] = np.sqrt(np.nansum((forward_model.data.flux - model_lr)**2 * weights)) / np.nansum(weights)

    # Extract the best rv
    xcorr_star_vel = vels[np.nanargmin(rmss)]
    
    return xcorr_star_vel

def generate_rv_contents(templates_dict, snr=100, blaze=True, ron=0, R=80000):
    """Wrapper to compute the rv content for several templates

    Args:
        templates_dict (dict): The dictionary of templates. Each entry is  2-column array.
        snr (int, optional): The peak snr per 1d-pixel. Defaults to 100.
        blaze (bool, optional): Whether or not to modulate by a pseudo blaze function. The pseudo blaze is a polynomial where the end points are at 50 percent in flux. Defaults to True.
        ron (int, optional): The read out noise of the detector. Defaults to 0.
        R (int, optional): The resolution to convolve the templates. Defaults to 80000.
    Returns:
        dict: A dictionary of RV contents with keys corresponding to the templates_dict dictionary. 
    """
    
    rv_contents = {}
    for t in templates_dict:
        rv_contents[t] = compute_rv_content(templates_dict[t][:, 0], templates_dict[t][:, 1], snr=snr, blaze=blaze, ron=ron, R=R)
        
    return rv_contents

def combine_orders(rvs, rvs_nightly, unc_nightly, weights, n_obs_nights):
    """Combines RVs across orders.

    Args:
        rvs (np.ndarray): The single measurement RVs (shape=(n_ord, n_spec)).
        rvs_nightly (np.ndarray): The rvs dictionary returned by parse_rvs (shape=(n_ord, n_nights))
        bad_rvs_dict (dict): A dictionary of rvs to ignore.
        rvcs (np.ndarray) : The rv contents of each order.
    Returns:
        np.ndarray: The single measurement RVs averaged across orders.
        np.ndarray: The single measurement uncertanties.
        np.ndarray: The nightly RVs averaged across orders.
        np.ndarray: The nightly RV uncertainties.
    """
    
    n_ord, n_spec = rvs.shape
    n_nights = len(n_obs_nights)
        
    # Copy the weights
    w = np.copy(weights)
    
    # Parameters are spectra and orders
    init_pars = np.zeros(n_spec + n_ord, dtype=float) + np.nan
    vlb = np.zeros(n_spec + n_ord, dtype=float) + np.nan
    vub = np.zeros(n_spec + n_ord, dtype=float) + np.nan
    vp = np.ones(n_spec + n_ord, dtype=int)
    
    # Copy rvs
    rvs_flagged = np.copy(rvs)
    
    # Stores single measurement rvs
    rvs_single = np.zeros(n_spec, dtype=float) + np.nan
    unc_single = np.zeros(n_spec, dtype=float) + np.nan

    # Do an initial order offset and flag
    for o in range(n_ord):
        rvs_flagged[o, :] = rvs_flagged[o, :] - np.nanmedian(rvs_flagged[o, :])

    # set bad rvs to nan
    bad = np.where(w == 0)
    if bad[0].size > 0:
        rvs_flagged[bad] = np.nan
    
    # Normalize weights
    w /= np.nansum(w)
    
    # Initialize params
    # Actual RVs with order offsets (above) subtracted off.
    for i in range(n_spec):
        init_pars[i] = 1
        vlb[i] = -np.inf
        vub[i] = np.inf
    
    # Order offsets, weighted average has been subtracted off, so should be nearly zero.
    for i in range(n_ord):
        init_pars[n_spec + i] = 1
        vlb[n_spec + i] = -np.inf
        vub[n_spec + i] = np.inf
    
    # Force last order to have zero offset
    init_pars[-1] = 0
    vp[-1] = 0
    
    # Disable bad spec
    bad = np.where(np.nansum(w, axis=0) == 0)
    vp[bad] = 0

    vpi = np.where(vp)[0] # change to actual indicies
        
    # Do the optimization. result contains the lnightly RVs (n_nights,) and order offsets (n_ord,)
    print('Solving RVs')

    result = NelderMead(rv_solver, init_pars, minvs=vlb, maxvs=vub, varies=vpi, ftol=1E-5, n_iterations=3, no_improve_break=3, args_to_pass=(rvs_flagged, w, n_obs_nights)).solve()
    
    # Best pars
    best_pars = result[0]
    order_offsets = best_pars[n_spec:] # the order offsets
    
    # Offset
    for o in range(n_ord):
        rvs_flagged[o, :] -= order_offsets[o]
    
    # Calculate the individual order averaged RVs here
    # Could use the "Fit" RVs but would rather do a direct weighted formulation with only the offsets.
    for ispec in range(n_spec):
        rr, ww = rvs_flagged[:, ispec], w[:, ispec]
        good_finite = np.where((ww > 0) & np.isfinite(ww))[0]
        ng = good_finite.size
        if ng == 0:
            rvs_single[ispec] = np.nan
            unc_single[ispec] = np.nan
        elif ng == 1:
            rvs_single[ispec] = np.copy(rr[good_finite[0]])
            unc_single[ispec] = np.nan
        elif ng in (2, 3):
            rvs_single[ispec] = pcmath.weighted_mean(rr, ww)
            unc_single[ispec] = pcmath.weighted_stddev(rr, ww) / np.sqrt(ng)
        else:
            wstddev = pcmath.weighted_stddev(rr, ww)
            wmean = pcmath.weighted_mean(rr, ww)
            bad = np.where(np.abs(rr - wmean) > 5*wstddev)
            if bad[0].size > 0:
                ng -= bad[0].size
                ww[bad] = 0
            rvs_single[ispec] = pcmath.weighted_mean(rr, ww)
            unc_single[ispec] = pcmath.weighted_stddev(rr, ww) / np.sqrt(ng)

        
    # Nightly RVs
    rvs_nightly_out = np.zeros(n_nights, dtype=float) + np.nan
    unc_nightly_out = np.zeros(n_nights, dtype=float) + np.nan
    f = 0
    l = n_obs_nights[0]

    for inight in range(n_nights):
        rr, ww = rvs_flagged[:, f:l].flatten(), w[:, f:l].flatten()
        good_finite = np.where((ww > 0) & np.isfinite(ww))[0]
        ng = good_finite.size
        if ng == 0:
            rvs_nightly_out[inight] = np.nan
            unc_nightly_out[inight] = np.nan
        elif ng == 1:
            rvs_nightly_out[inight] = np.copy(rr[good_finite[0]])
            unc_nightly_out[inight] = np.nan
        elif ng in (2, 3):
            rvs_nightly_out[inight] = pcmath.weighted_mean(rr, ww)
            unc_nightly_out[inight] = pcmath.weighted_stddev(rr, ww) / np.sqrt(ng)
        else:
            wstddev = pcmath.weighted_stddev(rr, ww)
            wmean = pcmath.weighted_mean(rr, ww)
            bad = np.where(np.abs(rr - wmean) > 5*wstddev)
            if bad[0].size > 0:
                ng -= bad[0].size
                ww[bad] = 0
            rvs_nightly_out[inight] = pcmath.weighted_mean(rr, ww)
            unc_nightly_out[inight] = pcmath.weighted_stddev(rr, ww) / np.sqrt(ng)
                
            
        if inight < n_nights - 1:
            f += n_obs_nights[inight]
            l += n_obs_nights[inight+1]

    return rvs_single, unc_single, rvs_nightly_out, unc_nightly_out


def combine_orders_fast(rvs, rvs_nightly, unc_nightly, weights, n_obs_nights):
    """Combines RVs across orders using a faster implementation of the above method.

    Args:
        rvs (np.ndarray): The single measurement RVs (shape=(n_ord, n_spec)).
        rvs_nightly (np.ndarray): The rvs dictionary returned by parse_rvs (shape=(n_ord, n_nights))
        bad_rvs_dict (dict): A dictionary of rvs to ignore.
        rvcs (np.ndarray) : The rv contents of each order.
    Returns:
        np.ndarray: The single measurement RVs averaged across orders.
        np.ndarray: The single measurement uncertanties.
        np.ndarray: The nightly RVs averaged across orders.
        np.ndarray: The nightly RV uncertainties.
    """
    n_ord, n_spec = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Generate nightly weights
    f, l = 0, n_obs_nights[0]
    w = np.zeros_like(rvs_nightly)
    for inight in range(n_nights):
        for o in range(n_ord):
            good = np.where(weights[o, f:l] > 0)[0]
            if good.size > 0:
                w[o, inight] = np.nanmean(weights[o, f:l])
        if inight < n_nights - 1:
            f += n_obs_nights[inight]
            l += n_obs_nights[inight+1]

    # Include nightly uncertainties
    w *= unc_nightly
    
    # Parameters are spectra and orders
    init_pars = np.zeros(n_nights + n_ord, dtype=float) + np.nan
    vlb = np.zeros(n_nights + n_ord, dtype=float) + np.nan
    vub = np.zeros(n_nights + n_ord, dtype=float) + np.nan
    vp = np.ones(n_nights + n_ord, dtype=int)
    
    rvs_nightly_flagged = np.copy(rvs_nightly)
    rvs_flagged = np.copy(rvs)
    
    # Stores single measurement rvs
    rvs_single = np.zeros(n_spec, dtype=float) + np.nan
    unc_single = np.zeros(n_spec, dtype=float) + np.nan

    # Do an initial order offset and flag
    for o in range(n_ord):
        rvs_nightly_flagged[o, :] = rvs_nightly_flagged[o, :] - np.nanmedian(rvs_flagged[o, :])

    # set bad rvs to nan
    bad = np.where(w == 0)
    if bad[0].size > 0:
        rvs_nightly_flagged[bad] = np.nan
    
    # Normalize weights
    w /= np.nansum(w)
    
    # Initialize params
    # Actual RVs with order offsets (above) subtracted off.
    for i in range(n_nights):
        init_pars[i] = 1
        vlb[i] = -np.inf
        vub[i] = np.inf
    
    # Order offsets, weighted average has been subtracted off, so should be nearly zero.
    for i in range(n_ord):
        init_pars[n_nights + i] = 1
        vlb[n_nights + i] = -np.inf
        vub[n_nights + i] = np.inf
    
    # Force last order to have zero offset
    init_pars[-1] = 0
    vp[-1] = 0
    
    # Disable bad nights
    bad = np.where(np.nansum(w, axis=0) == 0)
    vp[bad] = 0
    vpi = np.where(vp)[0] # change to actual indicies
        
    # Do the optimization. result contains the lnightly RVs (n_nights,) and order offsets (n_ord,)
    print('Solving RVs')

    result = NelderMead(rv_solver_nightsonly, init_pars, minvs=vlb, maxvs=vub, varies=vpi, ftol=1E-5, n_iterations=3, no_improve_break=3, args_to_pass=(rvs_nightly_flagged, w, n_obs_nights)).solve()
    
    # Best pars
    best_pars = result[0]
    order_offsets = best_pars[n_nights:] # the order offsets
    
    # Calculate the individual order averaged RVs here
    # Could use the "Fit" RVs but would rather do a direct weighted formulation with only the offsets.
    w = np.copy(weights)
    for ispec in range(n_spec):
        good = np.where(np.isfinite(rvs_flagged[:, ispec]))[0]
        ng = good.size
        if ng == 0:
            rvs_single[ispec] = np.nan
            unc_single[ispec] = np.nan
        elif ng == 1:
            rvs_single[ispec] = np.copy(rvs_flagged[good[0], ispec] - order_offsets)
            unc_single[ispec] = np.nan
        else:
            rvs_single[ispec] = pcmath.weighted_mean(rvs_flagged[good, ispec] - order_offsets, w[good, ispec])
            unc_single[ispec] = pcmath.weighted_stddev(rvs_flagged[good, ispec] - order_offsets, w[good, ispec]) / np.sqrt(ng)

    
    # Nightly RVs from above
    rvs_nightly_out = np.zeros(n_nights, dtype=float) + np.nan
    unc_nightly_out = np.zeros(n_nights, dtype=float) + np.nan
    f = 0
    l = n_obs_nights[0]
    if n_ord == 1:
        w = np.ones(n_spec)
    else:
        w = 1 / unc_single**2
    for inight in range(n_nights):
        good = np.where((w[f:l] > 0) & np.isfinite(w[f:l]))[0]
        ng = good.size
        if ng == 0:
            rvs_nightly_out[inight] = np.nan
            unc_nightly_out[inight] = np.nan
        elif ng == 1:
            rvs_nightly_out[inight] = np.copy(rvs_single[f:l][good[0]])
            unc_nightly_out[inight] = np.nan
        else:
            rvs_nightly_out[inight] = pcmath.weighted_mean(rvs_single[f:l][good], w[f:l][good])
            unc_nightly_out[inight] = pcmath.weighted_stddev(rvs_single[f:l][good], w[f:l][good]) / np.sqrt(ng)
            
        if inight < n_nights - 1:
            f += n_obs_nights[inight]
            l += n_obs_nights[inight+1]

    return rvs_single, unc_single, rvs_nightly_out, unc_nightly_out
    
    
 
 
# Wobble Method of combining RVs (Starting from single RVs)
# pars[0:n] = rvs
# pars[n:] = order offsets.
@jit(parallel=True)
def rv_solver(pars, rvs, weights, n_obs_nights):
    """Internal function to optimize the rv offsets between orders.
    """
    
    n_ord = rvs.shape[0]
    n_spec = rvs.shape[1]
    n_nights = n_obs_nights.size
    rvs_individual = pars[:n_spec]
    order_offsets = pars[n_spec:]
    term = np.empty(shape=(n_ord, n_spec), dtype=np.float64)
    
    # MIN[(RV_j - RV_jo + VR_o)^2 * w_jo]
    for o in prange(n_ord):
        for i in range(n_spec):
            term[o, i] = weights[o, i]**2 * (rvs_individual[i] - rvs[o, i] + order_offsets[o])**2
    rms = np.sqrt(np.nansum(term) / term.size)
    
    bad = np.where(term > 5*rms)
    if bad[0].size > 0:
        term[bad] = 0
    rms = np.sqrt(np.nansum(term) / term.size)
    
    return rms, 1

# Wobble Method of combining RVs (Starting from single RVs)
# pars[0:n] = rvs
# pars[n:] = order offsets.
@jit(parallel=True)
def rv_solver_nightsonly(pars, rvs, weights, n_obs_nights):
    n_ord = rvs.shape[0]
    n_spec = rvs.shape[1]
    n_nights = n_obs_nights.size
    rvs_nightly = pars[:n_nights]
    order_offsets = pars[n_nights:]
    term = np.empty(shape=(n_ord, n_nights), dtype=np.float64)
    
    # MIN[(RV_j - RV_jo + VR_o)^2 * w_jo]
    for o in prange(n_ord):
        for i in range(n_nights):
            term[o, i] = weights[o, i]**2 * (rvs_nightly[i] - rvs[o, i] + order_offsets[o])**2
    rms = np.sqrt(np.nansum(term) / term.size)

    return rms, 1
 
# Computes the RV content per pixel.
def compute_rv_content(wave, flux, snr=100, blaze=False, ron=0, R=None):
    """Computes the radial-velocity information content per pixel and for a whole swath.

    Args:
        wave (np.ndarray): The wavelength grid in units of Angstroms.
        flux (np.ndarray): The flux, normalized to ~ 1.
        snr (int, optional): The peak snr per 1d-pixel. Defaults to 100.
        blaze (bool, optional): Whether or not to modulate by a pseudo blaze function. The pseudo blaze is a polynomial where the end points 
        ron (int, optional): The read out noise of the detector. Defaults to 0.
        R (int, optional): The resolution to convolve the templates. Defaults to 80000.
    Returns:
        np.ndarray: The "rv information content" at each pixel, or precisely the uncertainty of measuring the rv of an individual "pixel".
        np.ndarray: The rv content for the whole swath ("uncertainty").
    """
    
    nx = wave.size
    fluxmod = np.copy(flux)
    
    # Convert to PE
    fluxmod *= snr**2
    
    if R is not None:
        fluxmod = pcmath.convolve_flux(wave, fluxmod, R=R)
    if blaze:
        good = np.where(np.isfinite(wave) & np.isfinite(fluxmod))[0]
        ng = good.size
        x, y = np.array([wave[good[0]], wave[good[int(ng/2)]], wave[good[-1]]]), np.array([0.5, 1, 0.5])
        pfit = pcmath.poly_coeffs(x, y)
        blz = np.polyval(pfit, wave)
        fluxmod *= blz
        
    rvc_per_pix = np.zeros(nx, dtype=np.float64) + np.nan
    good = np.where(np.isfinite(wave) & np.isfinite(fluxmod))[0]
    ng = good.size
    flux_spline = scipy.interpolate.CubicSpline(wave[good], fluxmod[good], extrapolate=True, bc_type='clamped')
    for i in range(nx):
        if i in good:
            slope = flux_spline(wave[i], 1)
            if not np.isfinite(slope) or slope == 0:
                continue
            rvc_per_pix[i] = cs.c * np.sqrt(fluxmod[i] + ron**2) / (wave[i] * np.abs(slope))

    rvc_tot = np.nansum(1 / rvc_per_pix**2)**(-0.5)
    
    return rvc_per_pix, rvc_tot
        

def compute_bisector_span(cc_vels, ccf, n_bs=1000):
    """Computes the bisector span of a given cross-correlation (RMS brute force) function.

    Args:
        cc_vels (np.ndarray): The velocities used for cross-correlation.
        ccf (int): The corresponding 1-dimensional RMS curve.
        n_bs (int): The number of depths to use in calculating the BIS, defaults to 1000.
    Returns:
        line_bisectors (np.ndarray): The line bisectors of the ccf.
        bisector_span (float): The bisecor span of the line bisector (commonly referred to as the BIS).
    """
    # B(d) = (v_l(d) + v_r(d)) / 2
    # v_l = velocities located on the left side from the minimum of the CCF peak and v_r are the ones on the right side
    # Mean bisector is computed at two depth ranges:
    # d = (0.1, 0.4), (0.6, 0.85)
    # B_(0.1, 0.4) = E(B(d)) for 0.1 to 0.4
    # B_(0.6, 0.85) = E(B(d)) for 0.6 to 0.85
    # BS = B_(0.1, 0.4) - B_(0.6, 0.85) = E(B(d)) for 0.1 to 0.4 - E(B(d)) for 0.6 to 0.85
    # .. = Average(B(d)) for 0.1 to 0.4 = Average((v_l(d) + v_r(d)) / 2) for 0.1 to 0.4
    
    # The bottom "half"
    drtop = (0.1, 0.4)
    
    # The top "half"
    drbottom = (0.6, 0.8)
    
    # The depths are from 0 to 1 for the normalized CCF
    depths = np.linspace(0, 1, num=n_bs)

    # Initialize the line bisector array (a function of CCF depth)
    line_bisectors = np.empty(depths.size, dtype=np.float64)

    # First normalize the CCF function
    ccf = ccf - np.nanmin(ccf)
    continuum = pcmath.weighted_median(ccf, med_val=0.95)
    ccfn = ccf / continuum
    
    # Get the velocities and offset such that the best vel is at zero
    best_vel = cc_vels[np.nanargmin(ccf)]
    cc_vels = cc_vels - best_vel
    
    # High res version of the ccf
    cc_vels_hr = np.linspace(np.nanmin(cc_vels), np.nanmax(cc_vels), num=int(cc_vels.size*100))
    good = np.where(np.isfinite(cc_vels) & np.isfinite(ccf))[0]
    ccf_hr = scipy.interpolate.CubicSpline(cc_vels[good], ccfn[good], extrapolate=False)(cc_vels_hr)
    
    # The vels on the left and right of the best vel.
    use_left = np.where(cc_vels < 0)[0]
    use_right = np.where(cc_vels > 0)[0]
    use_left_hr = np.where(cc_vels_hr < 0)[0]
    use_right_hr = np.where(cc_vels_hr > 0)[0]
    if use_left.size == 0 or use_right.size == 0:
        return np.nan, np.nan
    
    vel_max_ind_left, vel_max_ind_right = use_left[np.nanargmax(ccfn[use_left])], use_right[np.nanargmax(ccfn[use_right])]
    vel_max_ind_left_hr, vel_max_ind_right_hr = use_left_hr[np.nanargmax(ccf_hr[use_left_hr])], use_right_hr[np.nanargmax(ccf_hr[use_right_hr])]
    
    use_left = np.where((cc_vels > cc_vels[vel_max_ind_left]) & (cc_vels < 0))[0]
    use_right = np.where((cc_vels > 0) & (cc_vels < cc_vels[vel_max_ind_right]))[0]
    
    use_left_hr = np.where((cc_vels_hr > cc_vels_hr[vel_max_ind_left_hr]) & (cc_vels_hr < 0))[0]
    use_right_hr = np.where((cc_vels_hr > 0) & (cc_vels_hr < cc_vels_hr[vel_max_ind_right_hr]))[0]
    
    # Compute the line bisector
    for idepth in range(depths.size):
        d = depths[idepth]
        vl = cc_vels_hr[use_left_hr[pcmath.find_closest(ccf_hr[use_left_hr], d)[0]]]
        vr = cc_vels_hr[use_right_hr[pcmath.find_closest(ccf_hr[use_right_hr], d)[0]]]
        line_bisectors[idepth] = (vl + vr) / 2

    # Compute the bisector span
    top = np.where((depths > drtop[0]) & (depths < drtop[1]))[0]
    bottom = np.where((depths > drbottom[0]) & (depths < drbottom[1]))[0]
    avg_top = np.nanmean(line_bisectors[top])
    avg_bottom = np.nanmean(line_bisectors[bottom])
    
    # Store the bisector span
    bisector_span = avg_top - avg_bottom
    
    return line_bisectors, bisector_span


def compute_nightly_rvs_single_order(rvs, weights, n_obs_nights, flag_outliers=False, thresh=5):
    """Computes nightly RVs for a single order.

    Args:
        rvs (np.ndarray): The individual rvs array of length n_spec.
        weights (np.ndarray): The weights, also of length n_spec.
    """
    
    # The number of spectra and nights
    n_spec = len(rvs)
    n_nights = len(n_obs_nights)
    
    # Will hold the start and end index for each night.
    f, l = 0, n_obs_nights[0]
    
    # Initialize the nightly rvs and uncertainties
    rvs_nightly = np.zeros(n_nights)
    unc_nightly = np.zeros(n_nights)
    
    for inight in range(n_nights):
        r = rvs[f:l]
        w = weights[f:l]
        good = np.where(w > 0)[0]
        ng = good.size
        if ng == 0:
            rvs_nightly[inight] = np.nan
            unc_nightly[inight] = np.nan
        elif ng == 1:
            rvs_nightly[inight] = r[good[0]]
            unc_nightly[inight] = np.nan
        else:
            if flag_outliers:
                wavg = pcmath.weighted_mean(r, w)
                wstddev = pcmath.weighted_stddev(r, w)
                bad = np.where(np.abs(r - wavg) > thresh*wstddev)[0]
                n_bad = bad.size
                if n_bad > 0:
                    w[bad] = 0
            else:
                n_bad = 0
            rvs_nightly[inight] = pcmath.weighted_mean(r, w)
            unc_nightly[inight] = pcmath.weighted_stddev(r, w) / np.sqrt(ng - n_bad)
            
        if inight < n_nights - 1:
            f += n_obs_nights[inight]
            l += n_obs_nights[inight + 1]
            
    return rvs_nightly, unc_nightly
             
    
def compute_nightly_rvs_from_all(rvs, weights, n_obs_nights, flag_outliers=False, thresh=5):
    """Computes nightly RVs for a single order.

    Args:
        rvs (np.ndarray): The individual rvs array of shape (n_orders, n_spec).
        weights (np.ndarray): The weights, also of length (n_orders, n_spec).
    """
    
    # The number of spectra and nights
    n_orders, n_spec = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Will hold the start and end index for each night.
    f, l = 0, n_obs_nights[0]
    
    # Initialize the nightly rvs and uncertainties
    rvs_nightly = np.zeros(n_nights)
    unc_nightly = np.zeros(n_nights)
    
    for inight in range(n_nights):
        r = rvs[:, f:l].flatten()
        w = weights[:, f:l].flatten()
        good = np.where(w > 0)[0]
        ng = good.size
        if ng == 0:
            rvs_nightly[inight] = np.nan
            unc_nightly[inight] = np.nan
        elif ng == 1:
            rvs_nightly[inight] = r[good[0]]
            unc_nightly[inight] = np.nan
        else:
            if flag_outliers:
                wavg = pcmath.weighted_mean(r, w)
                wstddev = pcmath.weighted_stddev(r, w)
                bad = np.where(np.abs(r - wavg) > thresh*wstddev)[0]
                n_bad = bad.size
                if n_bad > 0:
                    w[bad] = 0
            else:
                n_bad = 0
            rvs_nightly[inight] = pcmath.weighted_mean(r, w)
            unc_nightly[inight] = pcmath.weighted_stddev(r, w) / np.sqrt(ng - n_bad)
            
        if inight < n_nights - 1:
            f += n_obs_nights[inight]
            l += n_obs_nights[inight + 1]
            
    return rvs_nightly, unc_nightly