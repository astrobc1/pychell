# Base Python
import copy

# Maths
import numpy as np
import scipy.interpolate
import scipy.stats
from scipy.constants import c as SPEED_OF_LIGHT

# Numba
from numba import jit

# Plotting
import matplotlib.pyplot as plt

# Pychell deps
import pychell.maths as pcmath
import pychell.utils as pcutils


#########################################
#### COMPUTE PER-NIGHT JDS (OR BJDS) ####
#########################################

def gen_nightly_jds(jds, utc_offset=0, sep=0.2):
    """Computes nightly (average) JDs (or BJDs) for a time-series observation over many nights. Average times are computed from the mean of the considered times.

    Args:
        jds (np.ndarray): An array of sorted JDs (or BJDs).
        utc_offset (int): The number of hours offset from UTC.
    Returns:
        np.ndarray: The average nightly jds.
        np.ndarray: The number of observations each night with data, of length n_nights.
    """
    
    # Number of spectra
    n_obs_tot = len(jds)

    # Keep track of previous night's last index
    prev_i = 0

    # Calculate mean JD date and number of observations per night for nightly
    # Assume that observations on separate nights if noon passes or if Delta_t > sep.
    jds_binned = []
    n_obs_binned = []
    if n_obs_tot == 1:
        jds_binned.append(jds[0])
        n_obs_binned.append(1)
    else:
        for i in range(n_obs_tot - 1):
            t_noon = np.ceil(jds[i] + utc_offset / 24) - utc_offset / 24
            if jds[i+1] > t_noon or jds[i+1] - jds[i] > sep:
                jd_avg = np.average(jds[prev_i:i+1])
                n_obs_night = i - prev_i + 1
                jds_binned.append(jd_avg)
                n_obs_binned.append(n_obs_night)
                prev_i = i + 1
        jds_binned.append(np.average(jds[prev_i:]))
        n_obs_binned.append(n_obs_tot - prev_i)

    jds_binned = np.array(jds_binned, dtype=float) # convert to np arrays
    n_obs_binned = np.array(n_obs_binned).astype(int)

    return jds_binned, n_obs_binned

####################################
#### CROSS-CORRELATION ROUTINES ####
####################################

def brute_force_ccf(p0, spectral_model, iter_index, vel_window=400_000):
    
    # Copy init params
    pars = copy.deepcopy(p0)
    
    # Get current star vel
    v0 = p0[spectral_model.star.par_names[0]].value
    
    # Make coarse and fine vel grids
    vel_step_coarse = 200
    vels_coarse = np.arange(v0 - vel_window / 2, v0 + vel_window / 2, vel_step_coarse)

    # Stores the rms as a function of velocity
    rmss_coarse = np.full(vels_coarse.size, dtype=np.float64, fill_value=np.nan)
    
    # Starting weights are bad pixels
    weights_init = np.copy(spectral_model.data.mask)

    # Wavelength grid for the data
    wave_data = spectral_model.wls.build(pars)

    # Compute RV info content
    rvc_per_pix, _ = compute_rv_content(p0, spectral_model, snr=100) # S/N here doesn't matter

    # Weights are 1 / rv info^2
    star_weights_init = 1 / rvc_per_pix**2

    # Data flux
    data_flux = np.copy(spectral_model.data.flux)
    
    # Compute RMS for coarse vels
    for i in range(vels_coarse.size):
        
        # Set the RV parameter to the current step
        pars[spectral_model.star.par_names[0]].value = vels_coarse[i]
        
        # Build the model
        _, model_lr = spectral_model.build(pars)
        
        # Shift the stellar weights instead of recomputing the rv content.
        _, star_weights_shifted = pcmath.doppler_shift_flux(wave_data, star_weights_init, vels_coarse[i], wave_out=wave_data)
        
        # Final weights
        weights = weights_init * star_weights_shifted
        bad = np.where(weights < 0)[0]
        weights[bad] = 0
        good = np.where(weights > 0)[0]
        if good.size == 0:
            continue
        
        # Compute the RMS
        rmss_coarse[i] = pcmath.rmsloss(data_flux, model_lr, weights=weights, flag_worst=20, remove_edges=20)

    # Extract the best coarse rv
    M = np.nanargmin(rmss_coarse)
    xcorr_rv_init = vels_coarse[M]

    # Determine the uncertainty from the coarse ccf
    try:
        n = np.nansum(spectral_model.data.mask)
        xcorr_rv_stddev, skew = compute_ccf_moments(vels_coarse, rmss_coarse)
        n_used = np.nansum(spectral_model.data.mask)
        xcorr_rv_unc = xcorr_rv_stddev / np.sqrt(n_used)
    except:
        return np.nan, np.nan, np.nan

    # Define the fine vels
    vel_step_fine = 2
    vel_window_fine = 1000  # For now
    vels_fine = np.arange(xcorr_rv_init - vel_window_fine / 2, xcorr_rv_init + vel_window_fine / 2, vel_step_fine)
    rmss_fine = np.full(vels_fine.size, fill_value=np.nan)
    
    # Now do a finer CCF
    for i in range(vels_fine.size):
        
        # Set the RV parameter to the current step
        pars[spectral_model.star.par_names[0]].value = vels_fine[i]
        
        # Build the model
        _, model_lr = spectral_model.build(pars)
        
        # Shift the stellar weights instead of recomputing the rv content.
        _, star_weights_shifted = pcmath.doppler_shift_flux(wave_data, star_weights_init, vels_fine[i], wave_out=wave_data)
        
        # Final weights
        weights = weights_init * star_weights_shifted
        bad = np.where(weights < 0)[0]
        weights[bad] = 0
        good = np.where(weights > 0)[0]
        if good.size == 0:
            continue
        
        # Compute the RMS
        rmss_fine[i] = pcmath.rmsloss(data_flux, model_lr, weights=weights, flag_worst=20, remove_edges=20)

    # Fit (M-2, M-1, ..., M+1, M+2) with parabola to determine true minimum
    # Extract the best coarse rv
    M = np.nanargmin(rmss_fine)
    use = np.arange(M - 2, M + 3, 1).astype(int)
    try:
        pfit = np.polyfit(vels_fine[use], rmss_fine[use], 2)
        xcorr_rv = -0.5 * pfit[1] / pfit[0] + spectral_model.data.bc_vel
    except:
        xcorr_rv = np.nan

    return xcorr_rv, xcorr_rv_unc, skew

def compute_ccf_moments(vels, rmss):
    p0 = [1.0, vels[np.nanargmin(rmss)], 5000, 10, 0.1] # amp, mean, sigma, alpha (~skewness), offset
    bounds = [(0.8, 1.2), (p0[1] - 2000, p0[1] + 2000), (100, 1E5), (-100, 100), (-0.5, 0.5)]
    opt_result = scipy.optimize.minimize(fit_ccf_skewnorm, x0=p0, bounds=bounds, args=(vels, rmss), method="Nelder-Mead")
    ccf_stddev = opt_result.x[2]
    alpha = opt_result.x[3]
    delta = alpha / np.sqrt(1 + alpha**2)
    skewness = (4 - np.pi) / 2 * (delta * np.sqrt(2 / np.pi))**3 / (1 - 2 * delta**2 / np.pi)**1.5
    return ccf_stddev, skewness

def fit_ccf_skewnorm(pars, vels, rmss):
    y = -1 * rmss - np.nanmin(-1 * rmss)
    y /= np.nanmax(y)
    model = pcmath.skew_normal(vels, pars[1], pars[2], pars[3])
    model /= np.nanmax(model)
    model = pars[0] * model + pars[4]
    residuals = y - model
    n_good = np.where(np.isfinite(residuals))[0].size
    rms = np.sqrt(np.nansum(residuals**2) / n_good)
    return rms


def brute_force_ccf_crude(p0, data, spectral_model):
    
    # Copy the parameters
    pars = copy.deepcopy(p0)
    
    # Velocity grid
    vels = np.arange(spectral_model.p0[spectral_model.star.par_names[0]].lower_bound, spectral_model.p0[spectral_model.star.par_names[0]].upper_bound, 500)

    # Stores the rms as a function of velocity
    rmss = np.full(vels.size, dtype=float, fill_value=np.nan)
    
    # Weights are only bad pixels
    weights = np.copy(data.mask)
        
    for i in range(vels.size):
        
        # Set the RV parameter to the current step
        pars[spectral_model.star.par_names[0]].value = vels[i]
        
        # Build the model
        _, model_lr = spectral_model.build(pars)
        
        # Compute the RMS
        rmss[i] = pcmath.rmsloss(data.flux, model_lr, weights=weights)

    # Extract the best rv
    xcorr_star_vel = vels[np.nanargmin(rmss)]
    
    return xcorr_star_vel

def compute_bis(cc_vels, ccf, v0, n_bs=1000, depth_range_bottom=None, depth_range_top=None):
    """Computes the Bisector inverse slope of a given cross-correlation (RMS brute force) function.

    Args:
        cc_vels (np.ndarray): The velocities used for cross-correlation.
        ccf (int): The corresponding 1-dimensional RMS curve.
        n_bs (int): The number of depths to use in calculating the BIS, defaults to 1000.
    Returns:
        line_bisectors (np.ndarray): The line bisectors of the ccf.
        bis (float): The bisector inverse slope (commonly referred to as the BIS).
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
    if depth_range_bottom is None:
        depth_range_bottom = (0.1, 0.4)
    
    # The top "half"
    if depth_range_top is None:
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


########################
#### CRUDE DETREND #####
########################

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


#########################
#### RV INFO CONTENT ####
#########################

def compute_rv_content(pars, spectral_model, snr=100):

    # Data wave grid
    data_wave = spectral_model.wls.build(pars)

    # Model wave grid
    model_wave = spectral_model.model_wave

    # Star flux on model data wave grid
    star_flux = spectral_model.star.build(pars, spectral_model.templates_dict["star"], model_wave)

    # Convolve stellar flux
    if spectral_model.lsf is not None:
        lsf = spectral_model.lsf.build(pars)
        star_flux = pcmath.convolve_flux(model_wave, star_flux, lsf=lsf)
    
    # Interpolate star flux onto data grid
    star_flux = pcmath.cspline_interp(model_wave, star_flux, data_wave)

    # Gas cell flux on model wave grid for kth observation
    if spectral_model.gas_cell is not None:
        gas_flux = spectral_model.gas_cell.build(pars, spectral_model.templates_dict["gas_cell"], model_wave)
        if spectral_model.lsf is not None:
            gas_flux = pcmath.convolve_flux(model_wave, gas_flux, lsf=lsf)
        
        # Interpolate gas cell flux onto data grid
        gas_flux = pcmath.cspline_interp(model_wave, gas_flux, data_wave)
    
    else:
        gas_flux = None

    # Telluric flux on model wave grid for kth observation
    if spectral_model.tellurics is not None:
        tell_flux = spectral_model.tellurics.build(pars, spectral_model.templates_dict["tellurics"], model_wave)
        if spectral_model.lsf is not None:
            tell_flux = pcmath.convolve_flux(model_wave, tell_flux, lsf=lsf)

        # Interpolate telluric flux onto data grid
        tell_flux = pcmath.cspline_interp(model_wave, tell_flux, data_wave)
    
    else:
        tell_flux = None

    # Find good pixels
    good = np.where(np.isfinite(data_wave) & np.isfinite(star_flux))[0]

    # Create a spline for the stellar flux to compute derivatives
    cspline_star = scipy.interpolate.CubicSpline(data_wave[good], star_flux[good], extrapolate=False)

    # Stores rv content for star
    rvc_per_pix_star = np.full(len(data_wave), np.nan)

    # Create a spline for the gas cell flux to compute derivatives
    if gas_flux is not None:

        # Find good pixels
        good = np.where(np.isfinite(data_wave) & np.isfinite(gas_flux))[0]

        cspline_gas = scipy.interpolate.CubicSpline(data_wave[good], gas_flux[good], extrapolate=False)

        # Stores rv content for gas cell
        rvc_per_pix_gas = np.full(len(data_wave), np.nan)

    # Loop over pixels
    for i in range(len(data_wave)):

        # Skip if this pixel is not used
        if not np.isfinite(data_wave[i]):
            continue

        # Compute stellar flux at this wavelength
        Ai = star_flux[i]

        # Include gas and tell flux
        if gas_flux is not None:
           Ai *= gas_flux[i]
        if tell_flux is not None:
           Ai *= tell_flux[i]

        # Scale to S/N (assumes gain = 1)
        Ai = Ai * snr**2

        # Compute derivative of stellar flux and gas flux
        dAi_dw_star = cspline_star(data_wave[i], 1)
        if gas_flux is not None:
            dAi_dw_star *= gas_flux[i]
        if tell_flux is not None:
            dAi_dw_star *= tell_flux[i]

        # Make sure slope is finite
        if not np.isfinite(dAi_dw_star):
            continue

        # Scale to S/N
        dAi_dw_star *= snr**2

        # Compute stellar rv content
        rvc_per_pix_star[i] = SPEED_OF_LIGHT * np.sqrt(Ai) / (data_wave[i] * np.abs(dAi_dw_star))

        # Compute derivative of gas cell flux
        if gas_flux is not None:
            dAi_dw_gas = cspline_gas(data_wave[i], 1)

            dAi_dw_gas *= star_flux[i]
            
            if tell_flux is not None:
                dAi_dw_gas *= tell_flux[i]

            # Scale to S/N
            dAi_dw_gas *= snr**2

            # Compute gas cell rv content
            rvc_per_pix_gas[i] = SPEED_OF_LIGHT * np.sqrt(Ai) / (data_wave[i] * np.abs(dAi_dw_gas))

    
    # Full RV Content per pixel
    if gas_flux is not None:
        rvc_per_pix = np.sqrt(rvc_per_pix_star**2 + rvc_per_pix_gas**2)
    else:
        rvc_per_pix = rvc_per_pix_star

    # Full RV Content
    rvc_tot = np.nansum(1 / rvc_per_pix**2)**-0.5

    # Return
    return rvc_per_pix, rvc_tot


#######################
#### CO-ADDING RVS ####
#######################

def bin_rvs_to_nights(jds, rvs, unc, err_type="empirical"):
    """A simple wrapper function to bin RVs to 1 per night.

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
    jds_nightly, n_obs_nights = gen_nightly_jds(jds, sep=0.5)
    
    # Number of nights
    n_nights = len(n_obs_nights)
    
    # Initialize nightly rvs and errors
    rvs_nightly = np.full(n_nights, np.nan, dtype=float)
    unc_nightly = np.full(n_nights, np.nan, dtype=float)
    
    # For each night, coadd RVs
    for i, f, l in pcutils.nightly_iteration(n_obs_nights):
        rr, uncc = rvs[f:l].flatten(), unc[f:l].flatten()
        ww = 1 / uncc**2
        rvs_nightly[i], unc_nightly[i] = pcmath.weighted_combine(rr, ww, yerr=uncc, err_type="empirical")
    
    return jds_nightly, rvs_nightly, unc_nightly

def compute_nightly_rvs_single_order(rvs, weights, n_obs_nights):
    """Computes nightly RVs for a single order.

    Args:
        rvs (np.ndarray): The individual rvs array of shape (n_obs, n_chunks).
        weights (np.ndarray): The weights, also of shape (n_obs, n_chunks).
        n_obs_nights (np.ndarray): The number of observations per night.
    """
    
    # The number of spectra and nights
    n_spec = len(rvs)
    n_nights = len(n_obs_nights)
    
    # Initialize the nightly rvs and uncertainties
    rvs_nightly = np.full(n_nights, np.nan)
    unc_nightly = np.full(n_nights, np.nan)
    
    for i, f, l in pcutils.nightly_iteration(n_obs_nights):
        rr = rvs[f:l].flatten()
        ww = weights[f:l].flatten()
        rvs_nightly[i], unc_nightly[i] = pcmath.weighted_combine(rr, ww, yerr=None, err_type="empirical")
            
    return rvs_nightly, unc_nightly

def combine_relative_rvs(rvs, weights, n_obs_nights):
    """Combines RVs considering the differences between all the data points.
    
    Args:
        rvs (np.ndarray): RVs of shape n_orders, n_spec
        weights (np.ndarray): Corresponding uncertainties of the same shape.
        n_obs_nights (np.ndarray): The number of oobservations on each night.
    """
    
    # Numbers
    n_orders, n_spec = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Determine differences and weights tensors
    rvlij = np.full((n_orders, n_spec, n_spec), np.nan)
    wlij = np.full((n_orders, n_spec, n_spec), np.nan)
    wli = np.full((n_orders, n_spec), np.nan)
    for l in range(n_orders):
        for i in range(n_spec):
            wli[l, i] = np.copy(weights[l, i])
            for j in range(n_spec):
                rvlij[l, i, j] = rvs[l, i] - rvs[l, j]
                wlij[l, i, j] = np.sqrt(weights[l, i] * weights[l, j])

    # Average over differences
    rvli = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    for l in range(n_orders):
        for i in range(n_spec):
            rr = rvlij[l, i, :]
            ww = wlij[l, i, :]
            rvli[l, i], _ = pcmath.weighted_combine(rr, ww)
    
    # Output arrays
    rvs_single_out = np.full(n_spec, fill_value=np.nan)
    unc_single_out = np.full(n_spec, fill_value=np.nan)
    rvs_nightly_out = np.full(n_nights, fill_value=np.nan)
    unc_nightly_out = np.full(n_nights, fill_value=np.nan)
    bad = np.where(~np.isfinite(wli))
    if bad[0].size > 0:
        wli[bad] = 0
        
    # Per-observation RVs
    for i in range(n_spec):
        rvs_single_out[i], unc_single_out[i] = pcmath.weighted_combine(rvli[:, i].flatten(), wli[:, i].flatten())
        
    # Per-night RVs
    for i, f, l in pcutils.nightly_iteration(n_obs_nights):
        rr = rvli[:, f:l].flatten()
        ww = wli[:, f:l].flatten()
        rvs_nightly_out[i], unc_nightly_out[i] = pcmath.weighted_combine(rr, ww, err_type="empirical")
        
    return rvs_single_out, unc_single_out, rvs_nightly_out, unc_nightly_out

def combine_rvs_weighted_mean(rvs, weights, n_obs_nights):
    """Combines RVs considering the differences between all the data points.
    
    Args:
        rvs (np.ndarray): RVs of shape n_orders, n_spec
        weights (np.ndarray): Corresponding uncertainties of the same shape.
    """
    
    # Numbers
    n_orders, n_spec = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Output arrays
    rvs_single_out = np.full(n_spec, fill_value=np.nan)
    unc_single_out = np.full(n_spec, fill_value=np.nan)
    rvs_nightly_out = np.full(n_nights, fill_value=np.nan)
    unc_nightly_out = np.full(n_nights, fill_value=np.nan)
    
    # Offset each order and chunk
    rvs_offset = np.copy(rvs)
    for o in range(n_orders):
        rvs_offset[o, :] = rvs[o, :] - pcmath.weighted_mean(rvs[o, :].flatten(), weights[o, :].flatten())
            
    for i in range(n_spec):
        rr = rvs_offset[:, i]
        ww = weights[:, i]
        rvs_single_out[i], unc_single_out[i] = pcmath.weighted_combine(rr.flatten(), ww.flatten())
        
    for i, f, l in pcutils.nightly_iteration(n_obs_nights):
        rr = rvs_offset[:, f:l]
        ww = weights[:, f:l]
        rvs_nightly_out[i], unc_nightly_out[i] = pcmath.weighted_combine(rr.flatten(), ww.flatten())
        
    return rvs_single_out, unc_single_out, rvs_nightly_out, unc_nightly_out
