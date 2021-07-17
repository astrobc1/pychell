# Base Python
import copy

# Maths
import numpy as np
import scipy.interpolate
import scipy.stats
from scipy.constants import c as SPEED_OF_LIGHT

# Pychell deps
import pychell.maths as pcmath
import pychell.utils as pcutils


#########################################
#### COMPUTE PER-NIGHT JDS (OR BJDS) ####
#########################################

def gen_nightly_jds(jds, sep=0.5):
    """Computes nightly (average) JDs (or BJDs) for a time-series observation over many nights. Average times are computed from the mean of the considered times.

    Args:
        jds (np.ndarray): An array of sorted JDs (or BJDs).
        sep (float): The minimum separation in days between two different nights of data, defaults to 0.5 (half a day).
    Returns:
        np.ndarray: The average nightly jds.
        np.ndarray: The number of observations each night with data, of length n_nights.
    """
    
    # Number of spectra
    n_obs_tot = len(jds)

    prev_i = 0
    # Calculate mean JD date and number of observations per night for nightly
    # coadded RV points; assume that observations on separate nights are
    # separated by at least "sep" days.
    jds_nightly = []
    n_obs_nights = []
    if n_obs_tot == 1:
        jds_nightly.append(jds[0])
        n_obs_nights.append(1)
    else:
        for i in range(n_obs_tot - 1):
            if jds[i+1] - jds[i] > sep:
                jd_avg = np.average(jds[prev_i:i+1])
                n_obs_night = i - prev_i + 1
                jds_nightly.append(jd_avg)
                n_obs_nights.append(n_obs_night)
                prev_i = i + 1
        jds_nightly.append(np.average(jds[prev_i:]))
        n_obs_nights.append(n_obs_tot - prev_i)

    jds_nightly = np.array(jds_nightly, dtype=float) # convert to np arrays
    n_obs_nights = np.array(n_obs_nights).astype(int)

    return jds_nightly, n_obs_nights

####################################
#### CROSS-CORRELATION ROUTINES ####
####################################

def brute_force_ccf(p0, spectral_model, iter_index, vel_step=10):
    
    # Copy init params
    pars = copy.deepcopy(p0)
    
    # Get current star vel
    v0 = p0[spectral_model.star.par_names[0]].value
    
    # Make a grid +/- 2 km/s
    vels = np.arange(v0 - 2000, v0 + 2000, vel_step)

    # Stores the rms as a function of velocity
    rmss = np.full(vels.size, dtype=np.float64, fill_value=np.nan)
    
    # Starting weights are flux uncertainties and bad pixels. If flux unc are uniform, they have no effect.
    weights_init = np.copy(spectral_model.data.mask * spectral_model.data.flux_unc)
    
    # Heavily downweight tellurics, fully mask regions of heavy absorption
    if spectral_model.tellurics is not None:
        
        # Build the telluric flux
        tell_flux = spectral_model.tellurics.build(pars, spectral_model.templates_dict['tellurics'], spectral_model.model_wave)
        tell_flux = spectral_model.lsf.convolve_flux(tell_flux, pars=pars)
        data_wave = spectral_model.wavelength_solution.build(pars)
        tell_flux = pcmath.lin_interp(spectral_model.model_wave, tell_flux, data_wave)
        
        # Make telluric weights
        tell_weights = tell_flux**4
        bad = np.where(~np.isfinite(tell_flux) | (tell_flux < 0.25))[0]
        tell_weights[bad] = 0
        
        # Combine weights
        weights_init *= tell_weights
        
    # Star weights depend on the information content
    if spectral_model.lsf is not None:
        width = pars[spectral_model.lsf.par_names[0]].value
    else:
        width = 1E-5
    rvc, _ = compute_rv_content(spectral_model.templates_dict['star'][:, 0], spectral_model.templates_dict['star'][:, 1], snr=100, blaze=True, ron=0, width=width)
    star_weights = 1 / rvc**2
    
    for i in range(vels.size):
        
        # Set the RV parameter to the current step
        pars[spectral_model.star.par_names[0]].value = vels[i]
        
        # Build the model
        wave_data, model_lr = spectral_model.build(pars)
        
        # Shift the stellar weights instead of recomputing the rv content.
        star_weights_shifted = pcmath.doppler_shift(spectral_model.templates_dict['star'][:, 0], vels[i], flux=star_weights, interp='linear', wave_out=wave_data)
        
        # Final weights
        weights = weights_init * star_weights_shifted
        
        # Compute the RMS
        rmss[i] = pcmath.rmsloss(spectral_model.data.flux, model_lr, weights=weights)

    # Extract the best rv
    M = np.nanargmin(rmss)
    vels_for_rv = vels + spectral_model.data.bc_vel
    xcorr_rv_init = vels[M] + spectral_model.data.bc_vel

    # Fit with a Polynomial
    # Include 10 points on each side of min vel (5 total points)
    try:
        use = np.arange(M - 2, M + 3).astype(int)
        pfit = np.polyfit(vels_for_rv[use], rmss[use], 2)
        xcorr_rv = -1 * pfit[1] / (2 * pfit[0])
    except:
        xcorr_rv = np.nan

    # Uncertainty in xc rv
    try:
        xcorr_rv_unc = ccf_uncertainty(vels_for_rv, rmss, xcorr_rv_init, np.sum(spectral_model.data.mask))
    except:
        xcorr_rv_unc = np.nan
        
    # BIS from rms surface
    try:
        _, bis = compute_bis(vels_for_rv, rmss, xcorr_rv, n_bs=1000)
    except:
        bis = np.nan

    return xcorr_rv, xcorr_rv_unc, bis, vels_for_rv, rmss

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
    
def brute_force_ccf_crude(p0, data, spectral_model):
    
    # Copy the parameters
    pars = copy.deepcopy(p0)
    
    # Velocity grid
    vels = np.arange(-250000, 250000, 500)

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
        fluxmod = pcmath.cspline_interp(wavemod[good], fluxmod[good], wave_new)
        wavemod = wave_new
    elif wave_to_sample is not None:
        good1 = np.where(np.isfinite(wavemod) & np.isfinite(fluxmod))[0]
        good2 = np.where(np.isfinite(wave_to_sample))
        bad2 = np.where(~np.isfinite(wave_to_sample))
        fluxnew = np.full(wave_to_sample.size, fill_value=np.nan)
        fluxnew[good2] = pcmath.cspline_interp(wavemod[good1], fluxmod[good1], wave_to_sample[good2])
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
        
        rvc_per_pix[i] = SPEED_OF_LIGHT * np.sqrt(fluxmod[i] + ron**2) / (wavemod[i] * np.abs(slope))
    
    good = np.where(np.isfinite(rvc_per_pix))[0]
    if good.size == 0:
        return np.nan, np.nan
    else:
        rvc_tot = np.nansum(1 / rvc_per_pix[good]**2)**-0.5
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
    jds_nightly, n_obs_nights = get_nightly_jds(jds, sep=0.5)
    
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
          
def compute_nightly_rvs_from_all(rvs, weights, n_obs_nights, flag_outliers=False, thresh=5):
    """Computes nightly RVs for a single order.

    Args:
        rvs (np.ndarray): The individual rvs array of shape (n_orders, n_obs).
        weights (np.ndarray): The weights, also of length (n_orders, n_obs).
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
    n_orders, n_obs = rvs.shape
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
        
    return np.copy(rvs[0, :]), np.zeros(n_obs) + 10, rvi, unci

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
        
    # Per-observation RVs
    for i in range(n_spec):
        rvs_single_out[i], unc_single_out[i] = pcmath.weighted_combine(rvli[:, i].flatten(), wli[:, i].flatten())
        
    # Per-night RVs
    for i, f, l in pcutils.nightly_iteration(n_obs_nights):
        rr = rvs_single_out[f:l]
        uncc = unc_single_out[f:l]
        ww = 1 / uncc**2
        rvs_nightly_out[i], unc_nightly_out[i] = pcmath.weighted_combine(rr, ww, yerr=uncc)
        
    return rvs_single_out, unc_single_out, rvs_nightly_out, unc_nightly_out

def combine_rvs_weighted_mean(rvs, weights, n_obs_nights):
    """Combines RVs considering the differences between all the data points.
    
    Args:
        rvs (np.ndarray): RVs of shape n_orders, n_obs, n_chunks
        weights (np.ndarray): Corresponding uncertainties of the same shape.
    """
    
    # Numbers
    n_orders, n_obs, n_chunks = rvs.shape
    n_nights = len(n_obs_nights)
    
    # Rephrase problem as n_quasi_orders = n_orders * n_chunks
    n_tot_chunks = n_orders * n_chunks
    
    # Output arrays
    rvs_single_out = np.full(n_obs, fill_value=np.nan)
    unc_single_out = np.full(n_obs, fill_value=np.nan)
    rvs_nightly_out = np.full(n_nights, fill_value=np.nan)
    unc_nightly_out = np.full(n_nights, fill_value=np.nan)
    
    # Offset each order and chunk
    rvs_offset = np.copy(rvs)
    for o in range(n_orders):
        for ichunk in range(n_chunks):
            rvs_offset[o, :, ichunk] = rvs[o, :, ichunk] - pcmath.weighted_mean(rvs[o, :, ichunk], weights[o, :, ichunk])
            
    for i in range(n_obs):
        rr = rvs_offset[:, i, :].flatten()
        ww = weights[:, i, :].flatten()
        rvs_single_out[i], rvs_single_out[i] = pcmath.weighted_combine(rr, ww)
        
    for i, f, l in pcutils.nightly_iteration(n_obs_nights):
        rr = rvs_offset[:, f:l, :].flatten()
        ww = weights[:, f:l, :].flatten()
        rvs_nightly_out[i], unc_nightly_out[i] = pcmath.weighted_combine(rr, ww)
            
    rvs_out = {"rvs": rvs_single_out, "unc": unc_single_out, "rvs_nightly": rvs_nightly_out, "unc_nightly" : unc_nightly_out}
        
    return rvs_out
