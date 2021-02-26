# Default python modules
import os

# Multiprocessing
from joblib import Parallel, delayed

# Maths
import numpy as np
import scipy.interpolate
import scipy.stats
import scipy.constants as cs

# LLVM
from numba import jit

# Graphics
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")
from robustneldermead.neldermead import NelderMead
import pychell.maths as pcmath
import copy

# Optimization
from robustneldermead.neldermead import NelderMead

def get_nightly_jds(jds, sep=0.5):
    """Computes nightly (average) JDs (or BJDs) for a time-series observation over many nights. Average times are computed from the mean of the considered times.

    Args:
        jds (np.ndarray): An array of sorted JDs (or BJDs).
        sep (float): The minimum separation in days between two different nights of data, defaults to 0.5 (half a day).
    Returns:
        np.ndarray: The average nightly jds.
        np.ndarray: The number of observations each night with data, of length n_nights.
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
        
def nightly_iteration(n_obs_nights):
    """A generator for iterating over observations within a given night.

    Args:
        n_obs_nights (np.ndarray): The number of observations on each night.

    Yields:
        int: The night index.
        int: The index of the first observation for this night.
        int: The index of the last observation for this night + 1. The additional + 1 is so one can index the array via array[f:l].
    """
    n_nights = len(n_obs_nights)
    f, l = 0, n_obs_nights[0]
    for i in range(n_nights):
        yield i, f, l
        if i < n_nights - 1:
            f += n_obs_nights[i]
            l += n_obs_nights[i+1]
        
        
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
        bad = np.where(tell_flux_lr < 0.99)
        tell_weights = np.ones_like(weights_init)
        tell_weights[bad] = 0
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
        star_weights_shifted = pcmath.doppler_shift(templates_dict['star'][:, 0], vels[i], flux=star_weights, interp='spline', wave_out=wave_lr)
        weights *= star_weights_shifted
        
        # Construct the RMS
        rmss[i] = pcmath.rmsloss(forward_model.data.flux_chunk, model_lr, weights=weights)

    # Extract the best rv
    M = np.nanargmin(rmss)
    vels_for_rv = vels + forward_model.data.bc_vel
    xcorr_rv_init = vels[M] + forward_model.data.bc_vel

    # Fit with a polynomial
    # Include 5 points on each side of min vel (11 total points)
    use = np.arange(M-5, M+6).astype(int)

    try:
        pfit = np.polyfit(vels_for_rv[use], rmss[use], 2)
        xcorr_rv = pfit[1] / (-2 * pfit[0])
    
        # Estimate uncertainty
        #xcorr_rv_unc = ccf_uncertainty(vels_for_rv, rmss, xcorr_rv_init, forward_model.data.mask_chunk.sum())
        xcorr_rv_unc = np.nan
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

# Silly and crude but moderately sensible.
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

def detrend_rvs(rvs, vec, thresh=None, poly_order=1):
    
    if thresh is None:
        thresh = 0.5
        
    pcc, _ = scipy.stats.pearsonr(vec, rvs)
    if np.abs(pcc) < thresh:
        return rvs
    else:
        pfit = np.polyfit(vec, rvs, poly_order)
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
    
    # Initialize the nightly rvs and uncertainties
    rvs_nightly = np.full(n_nights, np.nan)
    unc_nightly = np.full(n_nights, np.nan)
    
    for i, f, l in nightly_iteration(n_obs_nights):
        rr = rvs[f:l, :].flatten()
        ww = weights[f:l, :].flatten()
        rvs_nightly[i], unc_nightly[i] = pcmath.weighted_combine(rr, ww, yerr=None, err_type="empirical")
            
    return rvs_nightly, unc_nightly

def compute_nightly_rvs_single_chunk(rvs, weights, n_obs_nights):
    """Computes nightly RVs for a single order.

    Args:
        rvs (np.ndarray): The individual rvs array of shape (n_spec,).
        weights (np.ndarray): The weights, also of shape (n_spec,).
        n_obs_nights (np.ndarray): The array of length n_nights containing the number of observations on each night.
    """
    
    # The number of spectra and nights
    n_spec = len(rvs)
    n_nights = len(n_obs_nights)
    
    # Initialize the nightly rvs and uncertainties
    rvs_nightly = np.full(n_nights, np.nan)
    unc_nightly = np.full(n_nights, np.nan)
    
    for i, f, l in nightly_iterable(n_obs_nights):
        rr = rvs[f:l].flatten()
        ww = weights[f:l].flatten()
        rvs_nightly[i], unc_nightly[i] = pcmath.weighted_combine(rr, ww, yerr=None, err_type="empirical")
            
    return rvs_nightly, unc_nightly
   
def bin_rvs_to_nights(jds, rvs, unc, err_type="empirical"):
    """A separate function to bin RVs to 1 per night, primarily for use by external users for now.

    Args:
        jds (np.ndarray): The individual BJDs of length n_obs.
        rvs (np.ndarray): The individual RVs of length n_obs.
        unc (np.ndarray): The corresponding RV errors of length n_obs.
        
    Returns:
        np.ndarray: The nightly BJDs.
        np.ndarray: The nightly RVs.
        np.ndarray: The nightly RV errors.
    """
    
    # Get nightly JDs
    jds_nightly, n_obs_nights = get_nightly_jds(jds, sep=0.5)
    
    # Number of nights
    n_nights = len(n_obs_nights)
    
    # Initialize nightly rvs and errors
    rvs_nightly = np.zeros(n_nights, dtype=float)
    unc_nightly = np.zeros(n_nights, dtype=float)
    
    # For each night, coadd RVs
    for i, f, l in nightly_iteration(n_obs_nights):
        rr, uncc = rvs[f:l].flatten(), unc[f:l].flatten()
        ww = 1 / uncc**2
        rvs_nightly[i], unc_nightly[i] = pcmath.weighted_combine(rr, ww, yerr=uncc, err_type="empirical")
    return jds_nightly, rvs_nightly, unc_nightly
             
def compute_nightly_rvs_from_all(rvs, weights, n_obs_nights, flag_outliers=False, thresh=5):
    """Computes nightly RVs for a single order.

    Args:
        rvs (np.ndarray): The individual rvs array of shape (n_orders, n_spec).
        weights (np.ndarray): The weights, also of length (n_orders, n_spec).
    """
    
    # The number of spectra and nights
    n_orders, n_spec = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Initialize the nightly rvs and uncertainties
    rvs_nightly = np.full(n_nights, np.nan)
    unc_nightly = np.full(n_nights, np.nan)
    
    for i, f, l in nightly_iterable(n_obs_nights):
        rr = rvs[:, f:l].flatten()
        ww = weights[:, f:l].flatten()
        rvs_nightly[i], unc_nightly[i] = pcmath.weighted_combine(rr, ww, yerr=None, err_type="empirical")
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
    """Combines RVs considering the differences between all the data points.
    
    Args:
        rvs (np.ndarray): RVs of shape n_orders, n_spec, n_chunks
        weights (np.ndarray): Corresponding uncertainties of the same shape.
    """
    
    # Numbers
    n_orders, n_spec, n_chunks = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Rephrase problem as n_quasi_orders = n_orders * n_chunks
    n_tot_chunks = n_orders * n_chunks
    
    # Determine differences and weights tensors
    rvlij = np.full((n_tot_chunks, n_spec, n_spec), np.nan)
    wlij = np.full((n_tot_chunks, n_spec, n_spec), np.nan)
    for l in range(n_orders):
        for ichunk in range(n_chunks):
            i_quasi_chunk = l * n_chunks + ichunk
            for i in range(n_spec):
                for j in range(n_spec):
                    rvlij[i_quasi_chunk, i, j] = rvs[l, i, ichunk] - rvs[l, j, ichunk]
                    wlij[i_quasi_chunk, i, j] = weights[l, i, ichunk] * weights[l, j, ichunk]

    # Average over differences
    rvli = np.full(shape=(n_tot_chunks, n_spec), fill_value=np.nan)
    uncli = np.full(shape=(n_tot_chunks, n_spec), fill_value=np.nan)
    for l in range(n_tot_chunks):
        for i in range(n_spec):
            rr = rvlij[l, i, :]
            ww = wlij[l, i, :]
            rvli[l, i], uncli[l, i] = pcmath.weighted_combine(rr, ww, yerr=None, err_type="empirical")
    
    # Weights
    wli = (1 / uncli**2) / np.nansum(1 / uncli**2, axis=0)
    
    # Output arrays
    rvs_single_out = np.full(n_spec, fill_value=np.nan)
    unc_single_out = np.full(n_spec, fill_value=np.nan)
    rvs_nightly_out = np.full(n_nights, fill_value=np.nan)
    unc_nightly_out = np.full(n_nights, fill_value=np.nan)
    bad = np.where(~np.isfinite(wli))
    if bad[0].size > 0:
        wli[bad] = 0
        
    for i in range(n_spec):
        rvs_single_out[i], unc_single_out[i] = pcmath.weighted_combine(rvli[:, i].flatten(), wli[:, i].flatten())
        
    for i, f, l in nightly_iteration(n_obs_nights):
        rr = rvs_single_out[f:l]
        uncc = unc_single_out[f:l]
        ww = 1 / uncc**2
        rvs_nightly_out[i], unc_nightly_out[i] = pcmath.weighted_combine(rr, ww, yerr=uncc)
            
    rvs_out = {"rvs": rvs_single_out, "unc": unc_single_out, "rvs_nightly": rvs_nightly_out, "unc_nightly" : unc_nightly_out}
        
    return rvs_out

def combine_rvs_tfa(rvs, weights, n_obs_nights):
    """Combines RVs considering the differences between all the data points

    Args:
        rvs (np.ndarray): RVs, of shape=(n_orders, n_spec, n_chunks).
        weights (np.ndarray): Corresponding uncertainties with the same shape.
    """
    
    # Numbers
    n_orders, n_spec, n_chunks = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Rephrase problem as n_quasi_orders = n_tot_chunks = n_orders * n_chunks
    n_tot_chunks = n_orders * n_chunks
    
    # Reshape variables
    rvs_nonans = np.copy(rvs)
    rvscp = np.copy(rvs)
    rvs_nonans -= np.nanmedian(rvs_nonans)
    bad = np.where(~np.isfinite(rvs_nonans))
    rvs_nonans[bad] = 0
    S = rvs_nonans.reshape((n_tot_chunks, n_spec))
    rvscp = rvscp.reshape((n_tot_chunks, n_spec))
    w = weights.reshape((n_tot_chunks, n_spec))
    
    # Compute helper variables
    Aj = np.einsum("oj,oj->j", w**2, S)
    Bj = np.einsum("oj->j", w**2)
    Cm = np.einsum("mi,mi->m", w**2, S)
    Dm = np.einsum("mi->m", w**2)
    Hj = np.einsum("oj,o->j", w**2, Cm / Dm)
    bad = np.where(~np.isfinite(Hj))[0]
    Hj[bad] = 0
    Koij = np.zeros((n_tot_chunks, n_spec, n_spec))
    for o in range(n_tot_chunks):
        for i in range(n_spec):
            for j in range(n_spec):
                Koij[o, i, j] = w[o, j]**2 * w[o, i]**2 / Dm[o]
    
    bad = np.where(~np.isfinite(Koij))[0]
    Koij[bad] = 0
    
    Pij = np.einsum("oij->ij", Koij)
    Qj = Aj / Bj
    Rj = Hj / Bj
    Tj = 1 / Bj
    bad = np.where(~np.isfinite(Qj) | ~np.isfinite(Rj) | ~np.isfinite(Tj))[0]
    Qj[bad] = 0
    Rj[bad] = 0
    Tj[bad] = 0
    Pij_twiddle = np.zeros((n_spec, n_spec))
    for i in range(n_spec):
        Pij_twiddle[i, :] = Tj * Pij[i, :]
    
    # The individual RVs
    V = np.dot((Qj - Rj), np.linalg.inv(np.eye(n_spec) - Pij_twiddle.T))
    
    # Compute helper variables for the offsets
    Lm = np.einsum("mi,i->m", w**2, Aj / Bj)
    bad = np.where(~np.isfinite(Lm))[0]
    Lm[bad] = 0
    Zmoi = np.zeros((n_tot_chunks, n_tot_chunks, n_spec))
    for m in range(n_tot_chunks):
        for o in range(n_tot_chunks):
            for i in range(n_spec):
                Zmoi[m, o, i] = w[m, i]**2 * w[o, i]**2 / Bj[i]
                
    bad = np.where(~np.isfinite(Zmoi))[0]
    Zmoi[bad] = 0
    
    Gmo = np.einsum("moi->mo", Zmoi)
    deltam = Cm / Dm
    pim = Lm / Dm
    thetam = 1 / Dm
    bad = np.where(~np.isfinite(deltam) | ~np.isfinite(pim) | ~np.isfinite(thetam))[0]
    deltam[bad] = 0
    pim[bad] = 0
    thetam[bad] = 0
    Gmo_twiddle = np.zeros((n_tot_chunks, n_tot_chunks))
    for m in range(n_tot_chunks):
        Gmo_twiddle[m, :] = thetam * Gmo[m, :]
        
    gamma = np.dot((deltam - pim), np.linalg.inv(np.eye(n_tot_chunks) - Gmo_twiddle))
    
    # With gamma, construct the offset RVs
    rvs_offset = np.zeros((n_tot_chunks, n_spec))
    for o in range(n_tot_chunks):
        rvs_offset[o, :] = rvscp[o, :] - gamma[o]
        
    # With offset RVs, actually perform the offsets and compute weighted means & corresponding errors
    rvs_single_out = np.zeros(n_spec)
    unc_single_out = np.zeros(n_spec)
    rvs_nightly_out = np.zeros(n_nights)
    unc_nightly_out = np.zeros(n_nights)
    
    # For each observation, compute the coadded RVs
    for i in range(n_spec):
        rvs_single_out[i], unc_single_out[i] = pcmath.weighted_combine(rvs_offset[:, i], weights[:, i])
        
    f, l = 0, n_obs_nights[0]
    for i in range(n_nights):
        rvs_nightly_out[i], unc_nightly_out[i] = pcmath.weighted_combine(rvs_offset[:, f:l].flatten(), weights[:, f:l].flatten())
        if i < n_nights - 1:
            f += n_obs_nights[i]
            l += n_obs_nights[i+1]
            
    rvs_out = {"rvs": rvs_single_out, "unc": unc_single_out, "rvs_nightly": rvs_nightly_out, "unc_nightly" : unc_nightly_out}
        
    return rvs_out