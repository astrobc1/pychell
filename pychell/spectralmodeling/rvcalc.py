# Base Python
import copy

# Maths
import numpy as np
import scipy.interpolate
import scipy.stats
from scipy.constants import c as SPEED_OF_LIGHT

# Numba
from numba import njit

# Pychell deps
import pychell.maths as pcmath
import pychell.utils as pcutils


#################
#### BIN JDS ####
#################

def bin_jds_for_site(jds, site=None, utc_offset=None, sep=0.5):
    """Computes binned times for a multiple observations.

    Args:
        jds (np.ndarray): An array of sorted JDs (or BJDs).
        site (EarthLocation): The astropy.coordinates.EarthLocation object.
    Returns:
        np.ndarray: The average binned jds (unweighted).
        np.ndarray: An array of length n_nights where each entry contains the indices of the observations for that night as a tuple (first, last).
    """

    # UTC offset
    if utc_offset is None and site is not None:
        utc_offset = pcutils.get_utc_offset(site)

    
    # Number of spectra
    n_obs_tot = len(jds)

    # Keep track of previous night's last index
    prev_i = 0

    # Calculate mean JD date and number of observations per night for binned
    # Assume that observations on separate nights if noon passes or if Delta_t > sep.
    jds_binned = []
    n_obs_binned = []
    indices_binned = []
    if n_obs_tot == 1:
        jds_binned.append(jds[0])
        n_obs_binned.append(1)
    else:
        for i in range(n_obs_tot - 1):
            t_noon = np.ceil(jds[i] + utc_offset / 24) - utc_offset / 24
            if jds[i+1] > t_noon or jds[i+1] - jds[i] > sep:
                jd_avg = np.mean(jds[prev_i:i+1])
                n_obs_night = i - prev_i + 1
                jds_binned.append(jd_avg)
                n_obs_binned.append(n_obs_night)
                indices_binned.append([prev_i, i])
                prev_i = i + 1
        indices_binned.append([prev_i, n_obs_tot - 1])
        jds_binned.append(np.mean(jds[prev_i:]))
        n_obs_binned.append(n_obs_tot - prev_i)

    jds_binned = np.array(jds_binned, dtype=float) # convert to np arrays
    n_obs_binned = np.array(n_obs_binned)

    return jds_binned, indices_binned


######################
#### CCF ROUTINES ####
######################

def brute_force_ccf(p0, data, model, iter_index, measure_unc=False, vel_window_coarse=400_000, vel_step_coarse=200, vel_window_fine=2000, vel_step_fine=5):
    
    # Copy init params
    pars = copy.deepcopy(p0)
    
    # Get current star vel
    v0 = p0[model.star.par_names[0]].value
    
    # Make coarse and fine vel grids
    vels_coarse = np.arange(v0 - vel_window_coarse / 2, v0 + vel_window_coarse / 2, vel_step_coarse)

    # Stores the rms as a function of velocity
    rmss_coarse = np.full(len(vels_coarse), fill_value=np.nan)
    
    # Starting weights are bad pixels
    weights_init = np.copy(data.mask)

    # Wavelength grid for the data
    wave_data = model.wls.build(pars, data)

    # Compute RV info content
    rvc_per_pix, _ = compute_rv_content(model, p0, data, snr=100) # S/N here doesn't matter

    # Weights are 1 / rv info^2
    star_weights_init = 1 / rvc_per_pix**2

    # Data flux
    data_flux = np.copy(data.flux)
    
    # Compute RMS for coarse vels
    for i in range(vels_coarse.size):
        
        # Set the RV parameter to the current step
        pars[model.star.par_names[0]].value = vels_coarse[i]
        
        # Build the model
        _, model_lr = model.build(pars, data)
        
        # Shift the stellar weights instead of recomputing the rv content.
        #star_weights_shifted = pcmath.doppler_shift_flux(wave_data, star_weights_init, vels_coarse[i])
        
        # Final weights
        weights = np.copy(weights_init) # * star_weights_shifted
        bad = np.where((weights < 0) | ~np.isfinite(weights))[0]
        weights[bad] = 0
        good = np.where(weights > 0)[0]
        if good.size == 0:
            continue
        
        # Compute the RMS
        rmss_coarse[i] = pcmath.rmsloss(data_flux, model_lr, weights=weights, flag_worst=20, remove_edges=20)

    # Extract the best coarse rv
    M = np.nanargmin(rmss_coarse)
    xcorr_rv_init = vels_coarse[M]

    #breakpoint() #import matplotlib; import matplotlib.pyplot as plt; matplotlib.use("MacOSX");
    #plt.plot(vels_coarse, rmss_coarse); plt.show()

    # Determine the uncertainty from the coarse ccf
    xcorr_rv_mean, xcorr_rv_stddev, skew = compute_ccf_moments(vels_coarse, rmss_coarse)
    return xcorr_rv_mean + data.header['bc_vel'], xcorr_rv_stddev, skew
    n_used = np.nansum(data.mask)
    xcorr_rv_unc = xcorr_rv_stddev / np.sqrt(n_used)

    # Define the fine vels
    vels_fine = np.arange(xcorr_rv_init - vel_window_fine / 2, xcorr_rv_init + vel_window_fine / 2, vel_step_fine)
    rmss_fine = np.full(vels_fine.size, fill_value=np.nan)
    
    # Now do a finer CCF
    for i in range(vels_fine.size):
        
        # Set the RV parameter to the current step
        pars[model.star.par_names[0]].value = vels_fine[i]
        
        # Build the model
        _, model_lr = model.build(pars, data)
        
        # Shift the stellar weights instead of recomputing the rv content.
        #star_weights_shifted = pcmath.doppler_shift_flux(wave_data, star_weights_init, vels_coarse[i])
        
        # Final weights
        weights = np.copy(weights_init) # * star_weights_shifted
        bad = np.where((weights < 0) | ~np.isfinite(weights))[0]
        weights[bad] = 0
        good = np.where(weights > 0)[0]
        if good.size == 0:
            continue
        
        # Compute the RMS
        rmss_fine[i] = pcmath.rmsloss(data_flux, model_lr, weights=weights, flag_worst=20, remove_edges=20)

    # Fit (M-2, M-1, ..., M+1, M+2) with parabola to determine true minimum
    # Extract the best coarse rv
    #breakpoint() # import matplotlib; import matplotlib.pyplot as plt; matplotlib.use("MacOSX");
    M = np.nanargmin(rmss_fine)
    use = np.arange(M - 2, M + 3, 1).astype(int)
    try:
        pfit = np.polyfit(vels_fine[use], rmss_fine[use], 2)
        xcorr_rv = -0.5 * pfit[1] / pfit[0] + data.header["bc_vel"]
    except:
        xcorr_rv = np.nan

    return xcorr_rv, xcorr_rv_unc, skew

def compute_ccf_moments(vels, rmss):
    p0 = [1.0, vels[np.nanargmin(rmss)], 5000, 10, 0.1] # amp, mean, sigma, alpha (~skewness), offset
    bounds = [(0.8, 1.2), (p0[1] - 2000, p0[1] + 2000), (100, 1E5), (-100, 100), (-0.5, 0.5)]
    opt_result = scipy.optimize.minimize(fit_ccf_skewnorm, x0=p0, bounds=bounds, args=(vels, rmss), method="Nelder-Mead")
    ccf_mean = opt_result.x[1]
    ccf_stddev = opt_result.x[2]
    alpha = opt_result.x[3]
    delta = alpha / np.sqrt(1 + alpha**2)
    skewness = (4 - np.pi) / 2 * (delta * np.sqrt(2 / np.pi))**3 / (1 - 2 * delta**2 / np.pi)**1.5
    return ccf_mean, ccf_stddev, skewness

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


#########################
#### RV INFO CONTENT ####
#########################

def compute_rv_content(model, pars, data, snr=100, gain=1):

    # Ai = Star * Gascell * tell
    # dAi_dl = dStar/dl * Gas * tell + dGas/dl * Star * tell
    # dAi_dl = tell * (dStar/dl * Gas + dGas/dl * Star)
    # Scale each by S/N and gain

    # Data wave grid
    data_wave = model.wls.build(pars, data)

    # Model wave grid
    model_wave = model.templates["wave"]

    # Star
    star_flux = model.star.build(pars, model.templates)
    if model.lsf is not None:
        star_flux = model.lsf.convolve(star_flux, pars=pars)
    star_flux = pcmath.cspline_interp(model_wave, star_flux, data_wave)
    good = np.where(np.isfinite(data_wave) & np.isfinite(star_flux))
    cspline_star = scipy.interpolate.CubicSpline(data_wave[good], star_flux[good], extrapolate=False)

    # Gas cell
    if model.gascell is not None:
        gas_flux = model.gascell.build(pars, model.templates)
        if model.lsf is not None:
            gas_flux = model.lsf.convolve(gas_flux, pars=pars)
        gas_flux = pcmath.cspline_interp(model_wave, gas_flux, data_wave)
        good = np.where(np.isfinite(data_wave) & np.isfinite(gas_flux))
        cspline_gas = scipy.interpolate.CubicSpline(data_wave[good], gas_flux[good], extrapolate=False)

    # Tellurics
    if model.tellurics is not None:
        tell_flux = model.tellurics.build(pars, model.templates)
        if model.lsf is not None:
            tell_flux = model.lsf.convolve(tell_flux, pars=pars)
        tell_flux = pcmath.cspline_interp(model_wave, tell_flux, data_wave)
        good = np.where(np.isfinite(data_wave) & np.isfinite(tell_flux))
        cspline_tell = scipy.interpolate.CubicSpline(data_wave[good], tell_flux[good], extrapolate=False)
            
    # Stores rv content for star
    rvc_per_pix = np.full(len(data_wave), np.nan)

    # Loop over pixels
    for i in range(len(data_wave)):

        # Skip if this pixel is not used
        if not np.isfinite(data_wave[i]):
            continue

        # Compute stellar flux at this wavelength
        Ai = cspline_star(data_wave[i])
        dAidw = cspline_star(data_wave[i], 1)

        if model.gascell is not None:
            Ai *= cspline_gas(data_wave[i])
            dAidw = dAidw * cspline_gas(data_wave[i]) + cspline_gas(data_wave[i], 1) * cspline_star(data_wave[i])

        if model.tellurics is not None:
           Ai *= cspline_tell(data_wave[i])
           dAidw *= cspline_tell(data_wave[i])

        # Scale to S/N
        Ai *= snr**2 * gain
        dAidw *= snr**2 * gain

        # Make sure slope is finite
        if not np.isfinite(Ai) or not np.isfinite(dAidw):
            continue

        # Compute final rv content per detector pixel
        rvc_per_pix[i] = SPEED_OF_LIGHT * np.sqrt(Ai) / (data_wave[i] * np.abs(dAidw))

    # Full RV Content
    rvc_tot = np.nansum(1 / rvc_per_pix**2)**-0.5

    # Return
    return rvc_per_pix, rvc_tot


#######################
#### CO-ADDING RVS ####
#######################

def combine_relative_rvs(bjds, rvs, weights, indices, n_iterations=20, n_sigma=4):
    rvs, weights = np.copy(rvs), np.copy(weights) # Copy so as to not overwrite
    n_chunks, n_spec = rvs.shape
    n_bins = len(indices)
    for i in range(n_iterations):
        print(f"Combining RVs, iteration {i+1}")
        rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out = _combine_relative_rvs(bjds, rvs, weights, indices)
        n_bad = 0
        for j in range(n_bins):
            f, l = indices[j][0], indices[j][1]
            res = rvs_binned_out[j] - (rvs[:, f:l+1] - pcmath.weighted_mean(rvs, weights, axis=1)[:, np.newaxis])
            bad = np.where(np.abs(res) > n_sigma * pcmath.weighted_stddev(res, weights[:, f:l+1]))
            if bad[0].size > 0:
                n_bad += bad[0].size
                rvs[:, f:l+1][bad] = np.nan
                weights[:, f:l+1][bad] = 0
        if n_bad == 0:
            break

    return rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out


def align_chunks(rvs, weights):

    n_chunks, n_spec = rvs.shape
    
    # Determine differences and weights tensors
    rvlij = np.full((n_chunks, n_spec, n_spec), np.nan)
    wlij = np.full((n_chunks, n_spec, n_spec), np.nan)
    wli = np.full((n_chunks, n_spec), np.nan)
    for l in range(n_chunks):
        for i in range(n_spec):
            wli[l, i] = np.copy(weights[l, i])
            for j in range(n_spec):
                rvlij[l, i, j] = rvs[l, i] - rvs[l, j]
                wlij[l, i, j] = np.sqrt(weights[l, i] * weights[l, j])

    # Average over differences
    rvli = np.full(shape=(n_chunks, n_spec), fill_value=np.nan)
    for l in range(n_chunks):
        for i in range(n_spec):
            rr = np.copy(rvlij[l, i, :])
            ww = np.copy(wlij[l, i, :])
            rvli[l, i], _ = pcmath.weighted_combine(rr, ww)

    return rvli, wli

def _combine_relative_rvs(bjds, rvs, weights, indices):
    """Combines RVs considering the differences between all the data points.
    
    Args:
        rvs (np.ndarray): RVs of shape n_chunks, n_spec
        weights (np.ndarray): Corresponding uncertainties of the same shape.
        indices (np.ndarray): The indices for each bin
    """
    
    # Numbers
    n_chunks, n_spec = rvs.shape
    n_bins = len(indices)

    # Align chunks
    rvli, wli = align_chunks(rvs, weights)
    
    # Output arrays
    rvs_single_out = np.full(n_spec, fill_value=np.nan)
    unc_single_out = np.full(n_spec, fill_value=np.nan)
    t_binned_out = np.full(n_bins, fill_value=np.nan)
    rvs_binned_out = np.full(n_bins, fill_value=np.nan)
    unc_binned_out = np.full(n_bins, fill_value=np.nan)
    bad = np.where(~np.isfinite(wli))
    if bad[0].size > 0:
        wli[bad] = 0
        
    # Per-observation RVs
    for i in range(n_spec):
        rvs_single_out[i], unc_single_out[i] = pcmath.weighted_combine(rvli[:, i].flatten(), wli[:, i].flatten())
        
    # Per-night RVs
    for i in range(n_bins):
        f, l = indices[i]
        rr = rvli[:, f:l+1].flatten()
        ww = wli[:, f:l+1].flatten()
        bad = np.where(~np.isfinite(rr))[0]
        if bad.size > 0:
            ww[bad] = 0
        rvs_binned_out[i], unc_binned_out[i] = pcmath.weighted_combine(rr, ww, err_type="empirical")
        t_binned_out[i] = np.nanmean(bjds[f:l+1])
        
    return rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out

def combine_rvs_simple(bjds, rvs, weights, indices):
    """Combines RVs considering the differences between all the data points.
    
    Args:
        rvs (np.ndarray): RVs of shape n_orders, n_spec
        weights (np.ndarray): Corresponding uncertainties of the same shape.
    """
    
    # Numbers
    n_chunks, n_spec = rvs.shape
    n_bins = len(indices)
    
    # Output arrays
    rvs_single_out = np.full(n_spec, fill_value=np.nan)
    unc_single_out = np.full(n_spec, fill_value=np.nan)
    rvs_binned_out = np.full(n_bins, fill_value=np.nan)
    unc_binned_out = np.full(n_bins, fill_value=np.nan)
    t_binned_out = np.full(n_bins, fill_value=np.nan)
    
    # Offset each order and chunk
    rvs_offset = np.copy(rvs)
    for o in range(n_chunks):
        rvs_offset[o, :] = rvs[o, :] - pcmath.weighted_mean(rvs[o, :].flatten(), weights[o, :].flatten())
            
    for i in range(n_spec):
        rr = rvs_offset[:, i]
        ww = weights[:, i]
        rvs_single_out[i], unc_single_out[i] = pcmath.weighted_combine(rr.flatten(), ww.flatten())
        
    for i in range(n_bins):
        f, l = indices[i]
        rr = rvs_offset[:, f:l+1].flatten()
        ww = weights[:, f:l+1].flatten()
        rvs_binned_out[i], unc_binned_out[i] = pcmath.weighted_combine(rr, ww)
        t_binned_out[i] = np.nanmean(bjds[f:l+1])
        
    return rvs_single_out, unc_single_out, t_binned_out, rvs_binned_out, unc_binned_out
