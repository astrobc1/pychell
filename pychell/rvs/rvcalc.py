import os
from pdb import set_trace as stop

# Multiprocessing
from joblib import Parallel, delayed

# Maths
import numpy as np
import scipy.interpolate
import scipy.stats
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

# Optimization
from robustneldermead.neldermead import NelderMead


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


def weighted_brute_force(forward_model, templates_dict, iter_index, sregion, xcorr_options):
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
    
    pars = copy.deepcopy(forward_model.opt_results[-1][sregion.label]['xbest'])
    v0 = pars[forward_model.models_dict['star'].par_names[0]].value
    vels = np.linspace(v0 - xcorr_options['range'], v0 + xcorr_options['range'], num=xcorr_options['n_vels'])

    # Stores the rms as a function of velocity
    rmss = np.full(vels.size, dtype=np.float64, fill_value=np.nan)
    
    # Starting weights are flux uncertainties and bad pixels. If flux unc are uniform, they have no effect.
    weights_init = np.copy(forward_model.data.mask_chunk * forward_model.data.flux_unc_chunk)
    
    # Flag regions of heavy tellurics
    if 'tellurics' in forward_model.models_dict and forward_model.models_dict['tellurics'].enabled:
        tell_flux_hr = forward_model.models_dict['tellurics'].build(pars, templates_dict['tellurics'], templates_dict['star'][:, 0])
        tell_flux_hrc = forward_model.models_dict['lsf'].convolve_flux(tell_flux_hr, pars=pars)
        tell_flux_lr = np.interp(forward_model.models_dict['wavelength_solution'].build(pars), templates_dict['star'][:, 0], tell_flux_hrc, left=0, right=0)
        tell_weights = tell_flux_lr**2
        # Combine weights
        weights_init *= tell_weights
    
    # Star weights depend on the information content.
    rvc, _ = compute_rv_content(templates_dict['star'][:, 0], templates_dict['star'][:, 1], snr=100, blaze=False, ron=0, width=pars[forward_model.models_dict['lsf'].par_names[0]].value)
    star_weights = 1 / rvc**2
    
    for i in range(vels.size):
        
        # Copy the weights
        weights = np.copy(weights_init)
        
        # Set the RV parameter to the current step
        pars[forward_model.models_dict['star'].par_names[0]].setv(value=vels[i])
        
        # Build the model
        wave_lr, model_lr = forward_model.build_full(pars, templates_dict)
        
        # Shift the stellar weights instead of recomputing the rv content.
        star_weights_shifted = pcmath.doppler_shift(templates_dict['star'][:, 0], vels[i], flux=star_weights, interp='linear', wave_out=wave_lr)
        weights *= star_weights_shifted
        
        # Construct the RMS
        rmss[i] = pcmath.rmsloss(forward_model.data.flux_chunk, model_lr, weights=weights)

    # Extract the best rv
    M = np.nanargmin(rmss)
    vels_for_rv = vels + forward_model.data.bc_vel
    xcorr_rv_init = vels[M] + forward_model.data.bc_vel

    # Fit with a polynomial
    # Include 3 points on each side of min vel
    use = np.arange(M-3, M+3).astype(int)
    try:
        pfit = np.polyfit(vels_for_rv[use], rmss[use], 2)
        xcorr_rv = pfit[1] / (-2 * pfit[0])
    
        # Estimate uncertainty
        xcorr_rv_unc = ccf_uncertainty(vels_for_rv, rmss, xcorr_rv_init, forward_model.data.mask_chunk.sum())
    except:
        xcorr_rv = np.nan
        xcorr_rv_unc = np.nan
        bspan_result = (np.nan, np.nan)
    
    # Bisector span
    try:
        bspan_result = compute_bisector_span(vels_for_rv, rmss, xcorr_rv, n_bs=forward_model.xcorr_options['n_bs'])
    except:
        bspan_result = (np.nan, np.nan)
        
    ccf_result = {'rv': xcorr_rv, 'rv_unc': xcorr_rv_unc, 'bis': bspan_result[1], 'vels': vels_for_rv, 'ccf': rmss}
    return ccf_result

# Super silly and crude but moderately sensible.
def ccf_uncertainty(cc_vels, ccf, v0, n):
    
    # First normalize the RMS function
    ccff = -1 * ccf
    baseline = pcmath.weighted_median(ccff, percentile=0.05)
    ccff -= baseline
    ccff /= np.nanmax(ccff)
    use = np.where(ccff > 0.05)[0]
    init_pars = np.array([1, v0, 1000])
    solver = NelderMead(pcmath.rms_loss_creator(pcmath.lorentz), init_pars, args_to_pass=(cc_vels[use], ccff[use]))
    result = solver.solve()
    best_pars = result['xmin']
    unc = (best_pars[2] / 2.355) / np.sqrt(n)
    return unc
    

def crude_brute_force(forward_model, templates_dict, sregion):
    """Performs a pseudo cross-correlation via brute force RMS minimization and estimating the minumum with a quadratic.

    Args:
        forward_model (ForwardModel): A single forward model object.
        iter_index (int): The iteration to use to construct the forward model. Not used, is None.
        
    Returns:
        forward_model (ForwardModel): The forward model object with cross-correlation results stored in place.
    """

    # Init the whole order
    templates_dict_chunked = forward_model.init_chunk(templates_dict, forward_model.sregion_order)
    
    # Copy the parameters
    pars = copy.deepcopy(forward_model.initial_parameters)
    
    # Velocity grid
    vels = np.arange(-250000, 250000, 500)

    # Stores the rms as a function of velocity
    rmss = np.full(vels.size, dtype=np.float64, fill_value=np.nan)
    
    # Weights are bad pixels
    weights_init = np.copy(forward_model.data.mask)
        
    for i in range(vels.size):
        
        weights = np.copy(weights_init)
        
        # Set the RV parameter to the current step
        pars[forward_model.models_dict['star'].par_names[0]].setv(value=vels[i])
        
        # Build the model
        _, model_lr = forward_model.build_full(pars, templates_dict_chunked)
        
        # Compute the RMS
        rmss[i] = pcmath.rmsloss(forward_model.data.flux_chunk, model_lr, weights=weights[sregion.data_inds])

    # Extract the best rv
    xcorr_star_vel = vels[np.nanargmin(rmss)]
    
    return xcorr_star_vel



def modifed_stddev(rvs, weights, mus, n_obs_nights):
    n_nights = mus.size
    unc = np.full(n_nights, fill_value=np.nan)
    n_orders, n_spec = rvs.shape
    f, l = 0, n_obs_nights[0]
    for i in range(n_nights):
        rr, ww = rvs[:, f:l].flatten(), weights[:, f:l].flatten()
        ng = np.where(ww > 0)[0].size
        if ng in (0, 1):
            unc[i] = np.nan
        else:
            unc[i] = pcmath.weighted_stddev_mumod(rr, ww, mus[i]) / np.sqrt(ng)
            
        if i < n_nights - 1:
            f += n_obs_nights[i]
            l += n_obs_nights[i+1]
    
 
# Wobble Method of combining RVs (Starting from single RVs)
# pars[0:n] = rvs
# pars[n:] = order offsets.
@jit
def rv_solver(pars, rvs, weights, n_obs_nights):
    """Internal function to optimize the rv offsets between orders.
    """
    
    n_ord = rvs.shape[0]
    n_spec = rvs.shape[1]
    n_nights = n_obs_nights.size
    rvs_individual = pars[:n_spec]
    order_offsets = pars[n_spec:]
    term = np.empty(shape=(n_ord, n_spec), dtype=np.float64)
    
    for o in range(n_ord):
        for i in range(n_spec):
            term[o, i] = weights[o, i]**2 * (rvs_individual[i] - rvs[o, i] + order_offsets[o])**2
            
    rms = np.sqrt(np.nansum(term)) # Technically not an rms
    
    bad = np.where(term > 5*rms)
    if bad[0].size > 0:
        term[bad] = 0
    rms = np.sqrt(np.nansum(term))
    
    return rms

# Wobble Method of combining RVs (Starting from single RVs)
# pars[0:n] = rvs
# pars[n:] = order offsets.
@jit
def rv_solver_fast(pars, rvs, weights, n_obs_nights):
    
    n_ord, n_spec = rvs.shape
    n_tot = n_ord * n_spec
    n_nights = len(n_obs_nights)
    rvs_nightly = pars[:n_nights]
    order_offsets = pars[n_nights:]
    diffs = np.zeros(shape=(n_ord, n_spec), dtype=np.float64)
    
    for o in range(n_ord):
        f, l = 0, n_obs_nights[0]
        for i in range(n_nights):
            diffs[o, f:l] = weights[o, f:l]**2 * (rvs_nightly[i] - rvs[o, f:l] + order_offsets[o])**2
            if i < n_nights - 1:
                f += n_obs_nights[i]
                l += n_obs_nights[i+1]
    
    good = np.where(weights > 0)
    diffs  = diffs[good].flatten()
    ss = np.argsort(diffs)
    diffs[ss[-int(0.05 * n_tot):]] = 0
    rms = np.sqrt(np.nansum(diffs)) # Technically not an rms
    
    return rms
 
# Computes the RV content per pixel.
def compute_rv_content(wave, flux, snr=100, blaze=False, ron=0, R=None, width=None, sampling=None, wave_to_sample=None, lsf=None):
    """Computes the radial-velocity information content per pixel and for a whole swath.

    Args:
        wave (np.ndarray): The wavelength grid in units of Angstroms.
        flux (np.ndarray): The flux, normalized to ~ 1.
        snr (int, optional): The peak snr per 1d-pixel. Defaults to 100.
        blaze (bool, optional): Whether or not to modulate by a pseudo blaze function. The pseudo blaze is a polynomial where the end points 
        ron (int, optional): The read out noise of the detector. Defaults to 0.
        R (int, optional): The resolution to convolve the templates. Defaults to 80000.
        width (int, optional): The LSF width to convolve the templates if R is not set. R=80000
        sampling (float, optional): The desired sampling to compute the rv content on. Ideally, the input grid is sampled much higher than the detector grid for proper convolution, and sampling corresponds approximately to the detector grid.
    Returns:
        np.ndarray: The "rv information content" at each pixel.
        np.ndarray: The rv content for the whole swath.
    """
    
    nx_in = wave.size
    fluxmod = np.copy(flux)
    wavemod = np.copy(wave)
    
    # Convert to PE
    fluxmod *= snr**2
    
    # Convolve
    if R is not None or width is not None:
        fluxmod = pcmath.convolve_flux(wavemod, fluxmod, R=R, width=width)
    elif lsf is not None:
        fluxmod = pcmath.convolve_flux(wavemod, fluxmod, lsf=lsf)
        
    # Blaze modulation
    if blaze:
        good = np.where(np.isfinite(wavemod) & np.isfinite(fluxmod))[0]
        ng = good.size
        x, y = np.array([wavemod[good[0]], wavemod[good[int(ng/2)]], wavemod[good[-1]]]), np.array([0.2, 1, 0.2])
        pfit = pcmath.poly_coeffs(x, y)
        blz = np.polyval(pfit, wavemod)
        fluxmod *= blz
        
    # Downsample to detector grid
    if sampling is not None:
        good = np.where(np.isfinite(wavemod) & np.isfinite(fluxmod))[0]
        wave_new = np.arange(wavemod[0], wavemod[-1] + sampling, sampling)
        fluxmod = scipy.interpolate.CubicSpline(wavemod[good], fluxmod[good], extrapolate=False)(wave_new)
        wavemod = wave_new
    elif wave_to_sample is not None:
        good1 = np.where(np.isfinite(wavemod) & np.isfinite(fluxmod))[0]
        good2 = np.where(np.isfinite(wave_to_sample))
        bad2 = np.where(~np.isfinite(wave_to_sample))
        fluxnew = np.full(wave_to_sample.size, fill_value=np.nan)
        fluxnew[good2] = scipy.interpolate.CubicSpline(wavemod[good1], fluxmod[good1], extrapolate=False)(wave_to_sample[good2])
        fluxmod = fluxnew
        wavemod = wave_to_sample
        
    # Compute rv content
    nx = wavemod.size
    rvc_per_pix = np.full(nx, dtype=np.float64, fill_value=np.nan)
    good = np.where(np.isfinite(wavemod) & np.isfinite(fluxmod))[0]
    ng = good.size
    flux_spline = scipy.interpolate.CubicSpline(wavemod[good], fluxmod[good], extrapolate=False)
    for i in range(nx):
        
        if i not in good:
            continue
        
        # Derivative
        slope = flux_spline(wavemod[i], 1)
        
        # Compute rvc per pixel
        if not np.isfinite(slope) or slope == 0:
            continue
        
        rvc_per_pix[i] = cs.c * np.sqrt(fluxmod[i] + ron**2) / (wavemod[i] * np.abs(slope))
    
    good = np.where(np.isfinite(rvc_per_pix))[0]
    if good.size == 0:
        return np.nan, np.nan
    else:
        rvc_tot = np.nansum(1 / rvc_per_pix[good]**2)**-0.5
        return rvc_per_pix, rvc_tot
        

def compute_bisector_span(cc_vels, ccf, v0, n_bs=1000):
    """Computes the Bisector inverse slope of a given cross-correlation (RMS brute force) function.

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
    depth_range_bottom = (0.1, 0.4)
    
    # The top "half"
    depth_range_top = (0.6, 0.8)
    
    # The depths are from 0 to 1 for the normalized CCF
    depths = np.linspace(0, 1, num=n_bs)

    # Initialize the line bisector array (a function of CCF depth)
    line_bisectors = np.empty(depths.size, dtype=np.float64)

    # First normalize the RMS function
    ccf = ccf - np.nanmin(ccf)
    continuum = pcmath.weighted_median(ccf, percentile=0.95)
    ccfn = ccf / continuum
    good = np.where(ccfn < continuum)[0]
    best_loc = np.nanargmin(ccfn)
    good = np.where(good)
    
    # Remove v0
    cc_vels = cc_vels - v0
    
    cc_vels_hr = np.linspace(cc_vels[0], cc_vels[-1], num=cc_vels.size*1000)
    ccf_hr = scipy.interpolate.CubicSpline(cc_vels, ccfn, extrapolate=False)(cc_vels_hr)
    
    # Left and right side of CCF
    use_left_hr = np.where(cc_vels_hr < 0)[0]
    use_right_hr = np.where(cc_vels_hr > 0)[0]
    
    # Compute the line bisector
    for idepth in range(depths.size):
        vl, _ = pcmath.intersection(cc_vels_hr[use_left_hr], ccf_hr[use_left_hr], depths[idepth], precision=None)
        vr, _ = pcmath.intersection(cc_vels_hr[use_right_hr], ccf_hr[use_right_hr], depths[idepth], precision=None)
        line_bisectors[idepth] = (vl + vr) / 2

    # Compute the BIS
    top_inds = np.where((depths > depth_range_top[0]) & (depths < depth_range_top[1]))[0]
    bottom_inds = np.where((depths > depth_range_bottom[0]) & (depths < depth_range_bottom[1]))[0]
    avg_top = np.nanmean(line_bisectors[top_inds])
    avg_bottom = np.nanmean(line_bisectors[bottom_inds])
    bis = avg_top - avg_bottom
    
    return line_bisectors, bis

def detrend_rvs(rvs, vec, thresh=None):
    
    if thresh is None:
        thresh = 0.5
        
    pcc, _ = scipy.stats.pearsonr(vec, rvs)
    if np.abs(pcc) < thresh:
        return rvs
    else:
        pfit = np.polyfit(vec, rvs, 1)
        rvs_out = rvs - np.polyval(pfit, vec)
        return rvs_out

def compute_nightly_rvs_single_order(rvs, weights, n_obs_nights, flag_outliers=False, thresh=4):
    """Computes nightly RVs for a single order.

    Args:
        rvs (np.ndarray): The individual rvs array of shape (n_spec, n_chunks).
        weights (np.ndarray): The weights, also of shape (n_spec, n_chunks).
    """
    
    # The number of spectra and nights
    n_spec = len(rvs)
    n_nights = len(n_obs_nights)
    
    # Will hold the start and end index for each night.
    f, l = 0, n_obs_nights[0]
    
    # Initialize the nightly rvs and uncertainties
    rvs_nightly = np.full(n_nights, np.nan)
    unc_nightly = np.full(n_nights, np.nan)
    
    for inight in range(n_nights):
        r = rvs[f:l, :].flatten()
        w = weights[f:l, :].flatten()
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



def compute_relative_rvs_from_nights(rvs, rvs_nightly, unc_nightly, weights, n_obs_nights):
    """Combines RVs considering the differences between all the data points

    Args:
        rvs (np.ndarray): RVs
        weights (np.ndarray): Corresponding uncertainties
    """
    
    # Numbers
    n_orders, n_spec = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Define the differences
    rvlij = np.zeros((n_orders, n_nights, n_nights))
    unclij = np.zeros((n_orders, n_nights, n_nights))
    for l in range(n_orders):
        for i in range(n_nights):
            for j in range(n_nights):
                rvlij[l, i, j] = rvs_nightly[l, i] - rvs_nightly[l, j]
                unclij[l, i, j] = unc_nightly[l, i] * unc_nightly[l, j]
                
                
    wlij = np.zeros((n_orders, n_nights, n_nights))
    for l in range(n_orders):
        for i in range(n_nights):
            for j in range(n_nights):
                wlij[l, i, j] = (1 / unclij[l, i, j]**2) / np.nansum(1 / unclij[l, i, :]**2)

    # Average over differences
    rvli = np.nansum(wlij * rvlij, axis=2)
    
    # Average over orders
    uncli = np.copy(unc_nightly)
    wli = (1 / uncli**2) / np.nansum(1 / uncli**2, axis=0)
            
    rvi = np.nansum(wli * rvli, axis=0) / np.nansum(wli, axis=0)
    unci = np.sqrt((1 / np.nansum(1 / uncli**2, axis=0)) / n_orders)
        
    return np.copy(rvs[0, :]), np.zeros(n_spec) + 10, rvi, unci


def combine_relative_rvs(rvs, weights, n_obs_nights):
    """Combines RVs considering the differences between all the data points

    Args:
        rvs (np.ndarray): RVs
        weights (np.ndarray): Corresponding uncertainties
    """
    
    # Numbers
    n_orders, n_spec = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Determine differences and weights tensors
    rvlij = np.zeros((n_orders, n_spec, n_spec))
    wlij = np.zeros((n_orders, n_spec, n_spec))
    for l in range(n_orders):
        for i in range(n_spec):
            for j in range(n_spec):
                rvlij[l, i, j] = rvs[l, i] - rvs[l, j]
                wlij[l, i, j] = weights[l, i] * weights[l, j]

    # Average over differences
    rvli = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    uncli = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    for l in range(n_orders):
        for i in range(n_spec):
            good = np.where(wlij[l, i, :] > 0)[0]
            ng = good.size
            if ng == 0:
                continue
            elif ng == 1:
                rvli[l, i] = rvlij[l, i, good[0]]
                uncli[l, i] = np.nan
            else:
                rvli[l, i] = pcmath.weighted_mean(rvlij[l, i, :], wlij[l, i, :])
                uncli[l, i] = pcmath.weighted_stddev(rvlij[l, i, :], wlij[l, i, :]) / np.sqrt(ng)
    
    wli = (1 / uncli**2) / np.nansum(1 / uncli**2, axis=0)
    
    rvs_single_out = np.full(n_spec, fill_value=np.nan)
    unc_single_out = np.full(n_spec, fill_value=np.nan)
    rvs_nightly_out = np.full(n_nights, fill_value=np.nan)
    unc_nightly_out = np.full(n_nights, fill_value=np.nan)
    bad = np.where(~np.isfinite(wli))
    if bad[0].size > 0:
        wli[bad] = 0
        
    for i in range(n_spec):
        good = np.where(wli[:, i] > 0)[0]
        ng = good.size
        if ng == 0:
            continue
        elif ng == 1:
            rvs_single_out[i] = rvli[good[0], i]
            unc_single_out[i] = rvli[good[0], i]
        else:
            rvs_single_out[i] = pcmath.weighted_mean(rvli[:, i], wli[:, i])
            unc_single_out[i] = pcmath.weighted_stddev(rvli[:, i], wli[:, i]) / np.sqrt(ng)
        
    f, l = 0, n_obs_nights[0]

    for i in range(n_nights):
        if n_obs_nights[i] == 1 and np.isfinite(rvs_single_out[f]):
            rvs_nightly_out[i] = rvs_single_out[f]
            unc_nightly_out[i] = unc_single_out[f]
        else:
            ww = 1 / unc_single_out[f:l]**2
            rr = rvs_single_out[f:l]
            unc_all = unc_single_out[f:l]
            good = np.where(ww > 0)[0]
            ng = good.size
            if ng == 0:
                rvs_nightly_out[i] = np.nan
                unc_nightly_out[i] = np.nan
            elif ng == 1:
                rvs_nightly_out[i] = rr[good[0]]
                unc_nightly_out[i] = unc_all[good[0]]
            else:
                rvs_nightly_out[i] = pcmath.weighted_mean(rr[good], ww[good])
                unc_nightly_out[i] = pcmath.weighted_stddev(rr[good], ww[good]) / np.sqrt(ng)
            
        if i < n_nights - 1:
            f += n_obs_nights[i]
            l += n_obs_nights[i+1]
        
    return rvs_single_out, unc_single_out, rvs_nightly_out, unc_nightly_out