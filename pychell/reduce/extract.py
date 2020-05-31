
# Python default modules
import os
import glob
from pdb import set_trace as stop
import sys
import warnings

# Science / Math
import numpy as np
import scipy.interpolate
from astropy.io import fits
import torch

# LLVM
from numba import njit, jit, prange

# Graphics
import matplotlib.pyplot as plt

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
import pychell.reduce.calib as pccalib
import pychell.reduce.order_map as pcomap
import pychell.reduce.data2d as pcdata

import optimparameters.parameters as OptimParams
from robustneldermead.neldermead import NelderMead

# This function extracts all N orders and will be ran in parallel
def extract_full_image(data, general_settings, calib_settings, extraction_settings):
    """Performs calibration, traces orders if not already done so from flats, then optimally extracts the echelle orders. Results are written to files.

    Args:
        general_settings (dict): The general settings dictionary.
        calib_settings (dict): The calibration settings dictionary.
        extraction_settings (dict): The extraction settings dictionary.
    """
    
    print(' Extracting Spectrum ' + str(data.img_num+1) + ' of ' + str(data.n_tot_imgs) + ' ...', flush=True)
    print(' ' + str(data), flush=True)
    
    # Timer
    stopwatch = pcutils.StopWatch()

    # Load the full frame raw image
    data_image = data.parse_image()
    
    # Detector properties
    gain, read_noise, dark_current = general_settings['gain'], general_settings['read_noise'], general_settings['dark_current']
    
    # Perform the order map from the data itself if set
    if extraction_settings['order_map'] == 'empirical':
        print('    Tracing Orders ...', flush=True)
        data.trace_orders(general_settings['output_dir_root'], extraction_settings, src='empirical')
        
    # Define the order map information from the trace algorithm
    order_dicts, order_map_image = data.order_map.order_dicts, data.order_map.parse_image()
    
    # Flag regions in between orders
    bad = np.where(~np.isfinite(order_map_image))
    data_image[bad] = np.nan
    
    # Extract the image dimensions
    ny, nx = data_image.shape
    
    # The number of orders
    n_orders = len(order_dicts)
    
    # Oversample factor
    M = extraction_settings['oversample']
    
    # Load the calibration images
    if calib_settings['bias_subtraction']:
        master_bias_image = data.master_bias.parse_image()
    else:
        master_bias_image = None
        
    if calib_settings['dark_subtraction']:
        master_dark_image = data.master_dark.parse_image()
    else:
        master_dark_image = None
        
    if calib_settings['flat_division']:
        master_flat_image = data.master_flat.parse_image()
    else:
        master_flat_image = None
    
    # Standard reduction steps
    data_image = pccalib.standard_calibration(data_image, master_flat_image=master_flat_image, master_dark_image=master_dark_image, master_bias_image=master_bias_image)
    
    # Mask edge pixels as nan
    data_image[0:extraction_settings['mask_bottom_edge'], :] = np.nan
    data_image[ny-extraction_settings['mask_top_edge']:, :] = np.nan
    data_image[:, 0:extraction_settings['mask_left_edge']] = np.nan
    data_image[:, nx-extraction_settings['mask_right_edge']:] = np.nan
    
    # Create helpful 1d position arrays
    xarr = np.arange(nx)
    yarr = np.arange(ny)

    # Initiate storage arrays that will be useful for diagnostic plots
    # flux (photoelectrons), flux unc (photoelectrons), badpix
    reduced_orders = np.empty(shape=(n_orders, nx, 3), dtype=np.float64) # Optimal
    boxcar_spectra = np.empty(shape=(n_orders, nx), dtype=np.float64) # Boxcar, no uncertainty or bad pix stored

    # Each entry has shape=(height, 2), the first col is the y pixels, the second col is the profile
    trace_profiles = np.empty(shape=(ny*M, n_orders), dtype=np.float64)
    trace_profile_pars = np.empty(n_orders, dtype=OptimParams.Parameters)
    trace_profile_fits = np.empty(shape=(ny*M, n_orders), dtype=np.float64)
    
    # Estimate the SNR of the exposure
    snrs = np.empty(n_orders, dtype=np.float64)

    # Loop over orders
    for o in range(n_orders):
        
        # Stopwatch
        stopwatch.lap(str(o))
        
        #################################
        ##### Trace Profile & Y Pos #####
        #################################

        # Estimate y trace positions from the given order mapping
        y_positions_estimate = np.polyval(order_dicts[o]['pcoeffs'], xarr)

        # Extract the height of the order
        height = int(np.ceil(order_dicts[o]['height']))

        # Create order_image where only the relevant order is seen, still ny x nx
        order_image = np.copy(data_image)
        good_order = np.where(order_map_image == order_dicts[o]['label'])
        bad_order = np.where(order_map_image != order_dicts[o]['label'])
        n_good_order = good_order[0].size
        n_bad_order = bad_order[0].size
        if n_bad_order != 0:
            order_image[bad_order] = np.nan
            
        # Rectify the data to get an estimate of the trace profile
        order_image_hr_straight = rectify_trace(order_image, y_positions_estimate, M=M)
        
        # Estimate the trace profile from the rectified trace
        trace_profile_estimate = np.nanmedian(order_image_hr_straight, axis=1)
        
        good = np.where(np.isfinite(trace_profile_estimate))[0]
        trace_profile_estimate[good[0:extraction_settings['mask_trace_edges']*M]] = np.nan
        trace_profile_estimate[good[-M*extraction_settings['mask_trace_edges']:]] = np.nan
        
        # Refine trace position with cross correlation
        y_positions_refined = refine_trace_position(order_image, y_positions_estimate, trace_profile_estimate, trace_pos_polyorder=extraction_settings['trace_pos_polyorder'], M=M)
        
        # Now with a better y positions array, re-rectify the data and estimate the trace profile.
        order_image_hr_straight = rectify_trace(order_image, y_positions_refined, M=M)
        
        # New trace profile from better y position.
        trace_profile = np.nanmedian(order_image_hr_straight, axis=1)
        trace_profile_orig = np.copy(trace_profile)
        
        good = np.where(np.isfinite(trace_profile))[0]
        trace_profile[good[0:extraction_settings['mask_trace_edges']*M]] = np.nan
        trace_profile[good[-M*extraction_settings['mask_trace_edges']:]] = np.nan
        
        # Estimate sky and remove from profile
        if extraction_settings['sky_subtraction']:
            sky = estimate_sky(order_image_hr_straight, trace_profile, n_sky_rows=extraction_settings['n_sky_rows'])
            trace_profile -= np.nanmedian(sky) # subtrace off baseline estimate from sky
        else:
            trace_profile = np.copy(trace_profile)
            
        # Fix negative locations from sky subtraction
        bad = np.where(trace_profile < 0)[0]
        if bad.size > 0:
            trace_profile[bad] = np.nan
            
        # Model the trace profile with a modified gaussian
        best_trace_pars, trace_fit = model_trace_profile(order_image, trace_profile, M=M)
        
        # Pass trace to storage array
        trace_profiles[:, o] = trace_profile
        trace_profile_pars[o] = best_trace_pars
        trace_profile_fits[:, o] =  trace_fit

        # Estimate the S/N of the star in units of photoelectrons.
        # PE = ADU * Gain
        snrs[o] = np.sqrt(np.nansum(trace_profile) * gain / M)
        
        # Create a 2d curved profile with the refined trace position
        trace_profile_2d = create_2d_trace_profile(order_image, trace_profile, y_positions_refined, M1=M, M2=1)
        
        # Create a bad pixel mask and flag pixels with less than 5 percent flux
        badpix_mask = np.ones(shape=(ny, nx), dtype=np.float64) # ones and zeros
        bad = np.where((trace_profile_2d / np.nanmax(trace_profile_2d, axis=0) <= 0.05) | (~np.isfinite(order_image)) | (~np.isfinite(trace_profile_2d)))
        if bad[0].size > 0:
            badpix_mask[bad] = 0
            
        ##########################
        ### Optimal Extraction ###
        ##########################
        
        # Do the optimal extraction, Flag
        opt_spectrum, err_opt_spectrum = optimal_extraction(order_image, trace_profile_2d, badpix_mask, data.exp_time, sky=sky, n_sky_rows=extraction_settings['n_sky_rows'], gain=gain, read_noise=read_noise, dark_current=dark_current)
        
        badpix_mask = flag_bad_pixels(order_image, opt_spectrum, trace_profile_2d, badpix_mask, sky, gain, bad_thresh=0.5)

        # Do the optimal extraction, Flag
        opt_spectrum, err_opt_spectrum = optimal_extraction(order_image, trace_profile_2d, badpix_mask, data.exp_time, sky=sky, n_sky_rows=extraction_settings['n_sky_rows'], gain=gain, read_noise=read_noise, dark_current=dark_current)
        
        badpix_mask = flag_bad_pixels(order_image, opt_spectrum, trace_profile_2d, badpix_mask, sky, gain, bad_thresh=0.2)

        # Do a final optimal extraction
        opt_spectrum, err_opt_spectrum = optimal_extraction(order_image, trace_profile_2d, badpix_mask, data.exp_time, sky=sky, n_sky_rows=extraction_settings['n_sky_rows'], gain=gain, read_noise=read_noise, dark_current=dark_current)

        # Do a final crude extraction with the new bad pix mask for comparison
        boxcar_spectrum = boxcar_extraction(order_image, trace_profile_2d, sky=sky, badpix_mask=badpix_mask, gain=gain, units='PE')
        
        ###################
        ##### Outputs #####
        ###################
        
        # Final badpix array
        badpix_1d = np.ones(nx, dtype=int)
        bad = np.where(~np.isfinite(opt_spectrum) | ~np.isfinite(err_opt_spectrum))[0]
        if bad.size > 0:
            
            badpix_1d[bad] = 0
            
            # Also changes infs to nans
            opt_spectrum[bad] = np.nan
            err_opt_spectrum[bad] = np.nan
            boxcar_spectrum[bad] = np.nan
            
        # Flag according to outliers in 1d spectrum
        thresh = 5 / snrs[o] # ~ 5 sigma
        good = np.where(np.isfinite(opt_spectrum))[0]
        opt_spectrum_smooth = pcmath.median_filter1d(opt_spectrum[good], 5)
        med_val = pcmath.weighted_median(opt_spectrum_smooth, med_val=0.99)
        bad = np.where(np.abs(opt_spectrum_smooth - opt_spectrum[good]) > thresh*med_val)[0]
            
        # Pass to storage arrays
        reduced_orders[o, :, 0] = opt_spectrum
        reduced_orders[o, :, 1] = err_opt_spectrum
        reduced_orders[o, :, 2] = badpix_1d
        
        boxcar_spectra[o, :] = boxcar_spectrum

        print('  Extracted Order ' + str(o+1) + ' of ' + str(n_orders) + ' in ' + str(round(stopwatch.time_since(str(o))/60, 3)) + ' min', flush=True)
        
    # Store the SNR in the header
    data.header['SNR'] = str(np.average(snrs))

    # Plot and write to fits file
    plot_trace_profiles(data, trace_profiles, trace_profile_fits, M, general_settings['output_dir_root'])
    plot_full_spectrum(data, reduced_orders, boxcar_spectra, general_settings['output_dir_root'])
    data.save_reduced_orders(reduced_orders)
    
    # Save Trace profiles and models
    np.savez(data.out_file_trace_profiles, pars=trace_profile_pars, models=trace_profile_fits)
    trace_profile_pars = np.empty(n_orders, dtype=OptimParams.Parameters)
    trace_profile_fits = np.empty(shape=(ny*M, n_orders), dtype=np.float64)
    
    print(' Extracted Spectrum ' + str(data.img_num+1) + ' of ' + str(data.n_tot_imgs) + ' in ' + str(round(stopwatch.time_since()/60, 3)) + ' min', flush=True)
    
    
def boxcar_extraction(order_image, profile_2d, sky=None, badpix_mask=None, gain=1.0, units='PE'):
    """Performs a boxcar extraction on the nonrectified data.

    Args:
        order_image (np.ndarray): The data image.
        profile_2d (np.ndarray): The curved 2-dimensional profile, used to correct for bad pixels in the weighted sum.
        sky (np.ndarray): The sky background as a function of detector x-pixels (1-dimensional), defaults to None (no sky subtraction).
        badpix_mask: The bad pixel image mask, defaults to None (all finite pixels are considered).
        gain (float): The gain of the detector.
        units (str): The units to perform the extraction in. 'PE' for photoelectron, 'ADU' for analog to digital. Defaults to 'PE'
    Returns:
        spec (np.ndarray): The optimally extracted 1-dimensional spectrum.
        spec_unc (np.ndarray): The corresponding uncertainty.
    """
    ny, nx = order_image.shape
    
    # Create a bad pix mask is not already set
    if badpix_mask is None:
        badpix_mask = np.ones_like(order_image)
        bad = np.where(~np.isfinite(order_image))
        if bad[0].size > 0:
            badpix_mask[bad] = 0
            
    # Create a dummy sky array if not set
    if sky is None:
       sky = np.zeros(nx) 
    
    # If units in pe, multiply by gain
    if units == 'PE':
        order_image = order_image * gain
        sky = sky * gain
    
    spec = np.full(nx, fill_value=np.nan, dtype=np.float64)
    
    for x in prange(nx):
        S_x = order_image[:, x] - sky[x] # Subtract sky
        weights = badpix_mask[:, x] / np.nansum(badpix_mask[:, x]) # Normalize the weights
        profile_x = profile_2d[:, x] / np.nansum(profile_2d[:, x]) # Normalize this profile such that sum(P) = 1
        spec[x] = np.nansum(S_x * weights) / np.nansum(profile_x * weights) # Sum, account for unused pix.
        
    return spec


def flag_bad_pixels(order_image, current_spectrum, profile_2d, badpix_mask=None, sky=None, gain=1.0, bad_thresh=1):
    """Flags bad pixels in the data by convolving the flux back into 2d space with the curved trace profile and comparing to the data.

    Args:
        order_image (np.ndarray): The data image.
        current_spectrum (np.ndarray): The current extracted 1-dimensional spectrum.
        profile_2d (np.ndarray): The curved 2-dimensional profile, used to correct for bad pixels in the weighted sum.
        badpix_mask: The current bad pixel image mask, defaults to None (all finite pixels are considered).
        sky (np.ndarray): The sky background as a function of detector x-pixels (1-dimensional), defaults to None (no sky subtraction).
        gain (float): The gain of the detector.
        bad_thresh (float): The threshhold for bad pixels. Deviations larger than this in the data are flagged.
    """
    # Image dimensions
    ny, nx = order_image.shape
    
    # Sky
    if sky is None:
        sky = np.ones(nx)
    
    # Normalize trace profile
    profile_2d_sum_norm = profile_2d / np.nansum(profile_2d, axis=0)

    # Smooth the current spectrum
    current_spec_smooth = pcmath.median_filter1d(current_spectrum, 5)
    
    # Extend (convolve) the 1d flux into the spatial direction according to the trace profile
    current_spec_smooth_2d = np.outer(np.ones(ny, dtype=np.float64), current_spec_smooth) * profile_2d_sum_norm
    
    # Deviations
    deviations = np.empty(shape=(ny, nx), dtype=np.float64) + np.nan
    
    # Convert the data to PE
    data_pe = (order_image - np.outer(np.ones(nx), sky)) * gain
    
    for x in range(nx):
        good = np.where(np.isfinite(current_spec_smooth_2d[:, x]) & np.isfinite(data_pe[:, x]) & (badpix_mask[:, x] == 1))[0]
        if good.size < 5:
            continue
        med_val = pcmath.weighted_median(current_spec_smooth_2d[:, x] * badpix_mask[:, x], med_val=0.99)
        deviations[:, x] = np.abs(current_spec_smooth_2d[:, x] - data_pe[:, x]) / med_val
    
    bad = np.where(np.isfinite(deviations) & (deviations > bad_thresh))

    if bad[0].size > 0:
        badpix_mask[bad] = 0
    
    return badpix_mask

def create_2d_trace_profile(order_image, trace_profile, ypositions, M1=1, M2=1):
    """Creates a curved 2-dimensional trace profile given an input order image.

    Args:
        order_image (np.ndarray): A data image.
        trace_profile (np.ndarray): The 1-dimensional trace profile.
        ypositions (np.ndarray): The locations of the trace on the detector, y(x).
        M1 (int): The initial oversample factor, defaults to 1.
        M2 (int): The desired oversample factor, defaults to 1.
    """
    nx, ny = order_image.shape
    
    profile_2d = np.empty_like(order_image) + np.nan
    
    yarr1 = np.arange(0, ny, 1 / M1)
    yarr2 = np.arange(0, ny, 1 / M2)
    
    trace_max_pos_y = yarr1[np.nanargmax(trace_profile)]
    
    good_trace = np.where(np.isfinite(trace_profile))[0]
    
    for x in range(nx):
        
        good_data = np.where(np.isfinite(order_image[:, x]))[0]
        if good_data.size < 5:
            continue
            
        profile_2d[:, x] = scipy.interpolate.CubicSpline(yarr1[good_trace] - trace_max_pos_y + ypositions[x], trace_profile[good_trace], extrapolate=False)(yarr2)
        
    return profile_2d

def upsample_data(order_image, M):
    """Upsamples the data

    Args:
        order_image (np.ndarray): The data image.
        M (int): The desired oversample factor.
    """
    nx, ny = order_image.shape
    
    good = np.where(np.isfinite(order_image))
    
    order_image_hr = scipy.interpolate.interp2d(np.arange(nx), np.arange(ny), order_image, kind='cubic', copy=True, bounds_error=False, fill_value=np.nan)(np.arange(0, ny, 1 / M), np.arange(nx))
    
    return order_image_hr


def rectify_trace(order_image, ypositions, M=1):
    """Rectifies (straightens) the trace.

    Args:
        order_image (np.ndarray): The data image.
        ypositions (np.ndarray): The locations of the trace on the detector, y(x).
        M (int): The desired oversample factor, defaults to 1.
    """
    ny, nx = order_image.shape
    
    yarr1 = np.arange(ny)
    yarr2 = np.arange(0, ny, 1/M)
    fiducial_center_y = int(ny / 2)
    yarr2 = yarr2 - fiducial_center_y
    
    order_image2 = np.empty(shape=(ny*M, nx)) + np.nan
    
    for x in range(nx):
        good = np.where(np.isfinite(order_image[:, x]))[0]
        if good.size < 5:
            continue
        order_image2[:, x] = scipy.interpolate.CubicSpline(yarr1[good] - ypositions[x], order_image[good, x], extrapolate=False)(yarr2)
        
    return order_image2
    
# Sky = Sky(lambda) or Sky(x_pixel). i.e., no y dependence.
def estimate_sky(straight_order_image, trace_profile, n_sky_rows=8):
    """Estimates the sky background, sky(x), from a rectifed image.

    Args:
        straight_order_image (np.ndarray): The rectified data image.
        trace_profile (np.ndarray): The 1-dimensional trace profile.
        n_sky_rows (int): The number of rows to consider in estimating the sky background, sky(x).
    """
    
    # Normalize the trace profile
    trace_profile_maxnorm = trace_profile / np.nanmax(trace_profile)

    # Estimate the sky by considering where the flux is less than 10 percent the max value
    sky_locs = np.argsort(trace_profile_maxnorm)[0:n_sky_rows]
    
    # Create a smoothed image
    smoothed_image = pcmath.median_filter2d(straight_order_image, width=3, preserve_nans=True)
    
    # Estimate the sky background from this smoothed image
    sky_init = np.nanmedian(smoothed_image[sky_locs, :], axis=0)
    
    # Smooth the sky again
    sky = pcmath.median_filter1d(sky_init, width=3)
    
    return sky


# Model Trace profile with a Gaussian with a varying width
def model_trace_profile(order_image, trace_profile, M=1):
    """Models the trace profile with a modified Gaussian (exponent is free to vary).

    Args:
        order_image (np.ndarray): The data image.
        trace_profile (np.ndarray): The 1-dimensional trace profile.
        M (int): The oversample factor, defaults to 1.
    """
    
    yarr = np.arange(0, trace_profile.size / M, 1 / M)

    # Normalize the trace to max = 1
    trace_profile_maxnorm = trace_profile / np.nanmax(trace_profile)
    
    # Estimate parameters
    
    # Center
    mu = yarr[np.nanargmax(trace_profile)]
    
    # Sigma
    left_cut = yarr[np.where((trace_profile_maxnorm < 0.5) & (yarr < mu))[0][-1]]
    right_cut = yarr[np.where((trace_profile_maxnorm < 0.5) & (yarr > mu))[0][0]]
    sig = (right_cut - left_cut) / 3
    
    # Amplitude
    amp = 1.0
    
    # Exponent
    d = 1.0
    
    # Construct Parameter objects
    init_pars = OptimParams.Parameters()
    init_pars.add_parameter(OptimParams.Parameter(name='amp', value=amp, minv=0.9*amp, maxv=1.1*amp))
    init_pars.add_parameter(OptimParams.Parameter(name='mu', value=mu, minv=mu-2, maxv=mu+2))
    init_pars.add_parameter(OptimParams.Parameter(name='sigma', value=sig, minv=sig*0.5, maxv=sig*1.5))
    init_pars.add_parameter(OptimParams.Parameter(name='d', value=d, minv=0.6, maxv=1.4))
    
    # Run the Nelder-Mead solver
    solver = NelderMead(pcmath.gauss_modified_solver, init_pars, args_to_pass=(yarr, trace_profile_maxnorm))
    fit_result = solver.solve()
    best_pars = fit_result[0]

    # Re-construct the best fit trace profile
    trace_fit = pcmath.gauss_modified(yarr, best_pars['amp'].value, best_pars['mu'].value, best_pars['sigma'].value, best_pars['d'].value)
    
    return best_pars, trace_fit


def optimal_extraction(data_image, profile_2d, badpix_mask, exp_time, sky=None, n_sky_rows=None, gain=1, read_noise=0, dark_current=0):
    """Performs optimal extraction on the nonrectified data.

    Args:
        data_image (np.ndarray): The data image.
        profile_2d (np.ndarray): The curved 2-dimensional profile, used to correct for bad pixels in the weighted sum.
        badpix_mask (np.ndarray): The bad pixel image mask, defaults to None (all finite pixels are considered).
        exp_time (float): The exposure time.
        sky (np.ndarray): The sky background as a function of detector x-pixels (1-dimensional), defaults to None (no sky subtraction).
        n_sky_rows (int): The number of rows the sky was estimated from.
        gain (float): The gain of the detector, defaults to 1.
        read_noise (float): The read noise of the detector, defaults to 0.
        dark_current (float): The dark current of the detector, defaults to 0.
    Returns:
        spec (np.ndarray): The optimally extracted 1-dimensional spectrum.
        spec_unc (np.ndarray): The corresponding uncertainty.
    """
    data_image = data_image * gain # Convert the actual data to PEs
    sky = sky * gain # same for background sky
    sky_err = np.sqrt(sky / (n_sky_rows - 1))
    
    eff_read_noise = read_noise + dark_current * exp_time

    ny, nx = data_image.shape

    spec = np.full(nx, fill_value=np.nan, dtype=np.float64)
    spec_unc = np.full(nx, fill_value=np.nan, dtype=np.float64)

    for x in range(nx):
        
        profile_x = profile_2d[:, x] # The trace profile
        badpix_x = badpix_mask[:, x]
        data_x = data_image[:, x] # The data (includes sky) in units of PEs

        S_x = data_x - sky[x] # Star is trace - sky
        
        if np.all(~np.isfinite(S_x)) or np.nansum(badpix_x) == 0:
            spec[x] = np.nan
            spec_unc[x] = np.nan
            continue
        
        negs = np.where(S_x < 0)[0]
        if negs.size > 0:
            S_x[negs] = np.nan
            badpix_x[negs] = 0
        
        # Normalize the trace profile
        profile_x_sum_norm = profile_x / np.nansum(profile_x)
            
        # Variance
        var_x = read_noise**2 + S_x + sky[x] + sky_err[x]**2
        
        # Weights = P^2 / variance.
        # Using a sum or normalized trace profile here does affect things, but hardly.
        weights_x = profile_x_sum_norm**2 / var_x * badpix_x
        
        # Normalize the weights such that sum=1
        weights_x = weights_x / np.nansum(weights_x)
        
        # 1d final flux at column x
        spec[x] = np.nansum(S_x * weights_x) / np.nansum(profile_x_sum_norm * weights_x)
        spec_unc[x] = np.sqrt(np.nansum(var_x)) / np.nansum(profile_x_sum_norm * weights_x)
    
    return spec, spec_unc

def plot_trace_profiles(data, trace_profiles, trace_profile_fits, M, general_settings):
    
    n_orders = trace_profiles.shape[1]
    n_cols = 3
    n_rows = int(np.ceil(n_orders / n_cols))
    
    yarr = np.arange(0, trace_profiles[:, 0].size, 1 / M)
    
    plot_width = 15
    plot_height = 20
    dpi = 300
    
    fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(plot_width, plot_height), dpi=dpi)
    for row in range(n_rows):
        for col in range(n_cols):
            o = n_cols * row + col
            if o + 1 > n_orders:
                continue
            max_val = np.nanmax(trace_profiles[:, o])
            good = np.where(trace_profiles[:, o] > 0.1 * max_val)[0]
            left_cut_y, right_cut_y = yarr[good[0]] - 5, yarr[good[-1]] + 5
            good = np.where((yarr > left_cut_y) & (yarr < right_cut_y))[0]
            x = yarr[good] - yarr[good][0]
            axarr[row, col].plot(yarr[good], trace_profiles[good, o] / max_val, color='red', label='Median Trace Profile', lw=1)
            axarr[row, col].plot(yarr[good], trace_profile_fits[good, o], color='black', label='Modified Gaussian Fit', lw=1)
            axarr[row, col].set_ylim(-0.01, 1.2)
            axarr[row, col].legend(loc='upper right', prop={'size': 4})
    
    axarr[-1, 1].set_xlabel('Y Pixels', fontweight='bold', fontsize=14)
    axarr[int(n_rows / 2), 0].set_ylabel('Norm. Flux', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(data.out_file_trace_profile_plots)
    plt.close()


def plot_full_spectrum(data, reduced_orders, boxcar_spectra, general_settings):
    
    n_orders = reduced_orders.shape[0]
    n_cols = 3
    n_rows = int(np.ceil(n_orders / n_cols))
    
    pixels = np.arange(reduced_orders[0, :, 0].size)
    
    plot_width = 15
    plot_height = 20
    dpi = 300
    
    fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(plot_width, plot_height), dpi=dpi)
    for row in range(n_rows):
        for col in range(n_cols):
            o = n_cols * row + col
            if o + 1 > n_orders:
                continue
            badpix = reduced_orders[o, :, 2]
            good = np.where(badpix == 1)[0]
            axarr[row, col].plot(pixels[good], boxcar_spectra[o, good] / pcmath.weighted_median(boxcar_spectra[o, good], med_val=0.99), color='red', label='Boxcar', lw=0.5)
            axarr[row, col].plot(pixels[good], reduced_orders[o, good, 0] / pcmath.weighted_median(reduced_orders[o, good, 0], med_val=0.99), color='black', label='Optimal', lw=0.5)
            axarr[row, col].set_title('Order ' + str(o+1))
            axarr[row, col].legend(loc='upper right', prop={'size': 4})
    
    axarr[-1, 1].set_xlabel('X Pixels', fontweight='bold', fontsize=14)
    axarr[int(n_rows / 2), 0].set_ylabel('Norm. Flux', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(data.out_file_spectrum_plot)
    plt.close()
    
    
def refine_trace_position(order_image, ypositions, trace_profile, trace_pos_polyorder=2, M=1):
    """Refines the location of the trace on the detector.

    Args:
        order_image (np.ndarray): The data image.
        ypositions (np.ndarray): The current locations of the trace on the detector.
        trace_profile (np.ndarray): The 1-dimensional trace profile.
        M (int): The oversample factor, defaults to 1.
    """
    # The image dimensions
    ny, nx = order_image.shape
    
    # Stores the deviation from the true y position
    ypos_deviations = np.full(nx, dtype=np.float64, fill_value=np.nan)
    
    # Helpful arrays
    yarr_lr = np.arange(ny)
    xarr_lr = np.arange(nx)
    
    # CC lags
    lags = np.arange(-11, 10, 1)
    
    # Create a 2d curved profile from this estimate
    trace_profile_2d_estimate = create_2d_trace_profile(order_image, trace_profile, ypositions, M1=M, M2=1)
    
    # Estimate the initial spectrum to know which values correspond to large absorption features.
    spectrum_1d_estimate = boxcar_extraction(order_image, trace_profile_2d_estimate, sky=None, units='ADU')
    
    # Smooth this initial spectrum
    spectrum_1d_estimate_smooth = pcmath.median_filter1d(spectrum_1d_estimate, width=5)
    
    # Normalize
    spectrum_1d_estimate_smooth /= pcmath.weighted_median(spectrum_1d_estimate_smooth, med_val=0.98)
    
    # Cross correlate each data column with the trace profile estimate
    for x in range(nx):

        # Skip pixels where the intial spectrum has a flux of less than 50% of the max
        if not np.isfinite(spectrum_1d_estimate_smooth[x]) or spectrum_1d_estimate_smooth[x] <= 0.2:
            continue
        
        # If not enough data points, continue
        good_data = np.where(np.isfinite(order_image[:, x]))[0]
        if good_data.size < 8:
            continue
        
        # Only consider good regions for cross correlation
        consider = np.where((trace_profile_2d_estimate[:, x] > 0) & np.isfinite(trace_profile_2d_estimate[:, x]) & np.isfinite(order_image[:, x]))[0]
        
        # Cross correlation
        xcorr = pcmath.cross_correlate(order_image[consider, x], trace_profile_2d_estimate[consider, x], lags)
        xcorr /= pcmath.weighted_median(xcorr, med_val=0.99)
        
        # Fit x corr with a gaussian to precisely determine its max position
        
        # The best lag val
        best_lag_val = lags[np.nanargmax(xcorr)]
        
        # Estimate the width
        left_cut = np.where((xcorr < 0.5) & (lags < best_lag_val))[0]
        right_cut = np.where((xcorr < 0.5) & (lags > best_lag_val))[0]
        if left_cut.size < 2 or right_cut.size < 2:
            continue
        sig = (right_cut[0] - left_cut[-1]) / 3
        
        # Initiate the parameters
        init_pars = OptimParams.Parameters()
        init_pars.add_parameter(OptimParams.Parameter(name='amp', value=1.0, minv=0.8, maxv=1.2))
        init_pars.add_parameter(OptimParams.Parameter(name='mu', value=best_lag_val, minv=best_lag_val-2, maxv=best_lag_val+2))
        init_pars.add_parameter(OptimParams.Parameter(name='sigma', value=sig, minv=sig*0.5, maxv=sig*1.5))
        solver = NelderMead(pcmath.gauss_solver, init_pars, args_to_pass=(lags, xcorr))
        fit_result = solver.solve()
        best_pars = fit_result[0]
        ypos_deviations[x] = best_pars['mu'].value
       
        
    # Fit with polynomial
    
    # Smooth the deviations
    ypos_deviations_smooth = pcmath.median_filter1d(ypos_deviations, width=9, preserve_nans=True)
    
    # Add deviations to current estimate
    y_positions_refined = ypositions - ypos_deviations_smooth
    
    # Good regions
    good = np.where(np.isfinite(y_positions_refined))[0]
    
    # Final trace positions
    pfit = np.polyfit(xarr_lr[good], y_positions_refined[good], trace_pos_polyorder)
    y_positions_refined = np.polyval(pfit, xarr_lr)
    
    return y_positions_refined
    