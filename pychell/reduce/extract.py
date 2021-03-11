
# Python default modules
import os
import glob
import sys
import copy
import warnings

# Science / Math
import numpy as np
import scipy.interpolate
try:
    import torch
except:
    warnings.warn("Could not import pytorch!")
import scipy.signal
from astropy.io import fits

# LLVM
from numba import njit, jit, prange

# Graphics
import matplotlib.pyplot as plt

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
import pychell.reduce.calib as pccalib
import pychell.reduce.order_map as pcomap
import pychell.data as pcdata

import optimparameters.parameters as OptimParams
from robustneldermead.neldermead import NelderMead


def extract_full_image_wrapper(data_all, index, config):
    """A wrapper to extract a full frame image for printing purposes.

    Args:
        data_all (list): A list of SpecDataImage objects.
        index (int): The index of the image in data_all to extract.
        config (dict): The reduction settings dictionary.
    """
    
    # Initialize a timer
    stopwatch = pcutils.StopWatch()
    
    print('Extracting Image ' + str(index + 1) + ' of ' + str(len(data_all)) + ' ...')
    
    # Fetch the image from the full list
    data = data_all[index]
    
    # Extract the full frame image
    extract_full_image(data, config)
    
    print(' Extracted Spectrum ' + str(index+1) + ' of ' + str(len(data_all)) + ' in ' + str(round(stopwatch.time_since()/60, 3)) + ' min', flush=True)

def extract_full_image(data, config):
    """ Performs calibration and extracts a full frame image.

    Args:
        data (SpecDataImage): The data to reduce and extract.
        config (dict): The reduction settings dictionary.
    """
    
    # A stopwatch for timing
    stopwatch = pcutils.StopWatch()
    
    # Load the full frame raw image
    data_image = data.parse_image()

    # Load the order map
    trace_map_image, orders_list = data.order_map.load_map_image(), data.order_map.orders_list
    
    # Standard dark, bias, flat calibration.
    data_image = pccalib.standard_calibration(data, data_image, bias_subtraction=config['bias_subtraction'], dark_subtraction=config['dark_subtraction'], flat_division=config['flat_division'])
    
    # The imenage dimensions
    ny, nx = data_image.shape
    
    # The number of echelle orders, possibly composed of multiple traces.
    n_orders = len(orders_list)
    n_traces = len(orders_list[0])
    
    # Mask edge pixels as nan (not an actual crop)
    data_image = crop_image(data_image, config)
    
    # Also flag regions in between orders
    bad = np.where(~np.isfinite(trace_map_image))
    if bad[0].size > 0:
        data_image[bad] = np.nan
    
    # Used for science, last col is flux, flux_unc, badpix
    reduced_orders = np.empty(shape=(n_orders, n_traces, nx, 3), dtype=float)
    
    # Boxcar extracted spectra (no profile weights)
    boxcar_spectra = np.empty(shape=(n_orders, n_traces, nx), dtype=float)
    
    # Trace profiles and positions for each order, possibly multi-trace
    trace_profile_csplines = np.empty(shape=(n_orders, n_traces), dtype=scipy.interpolate.CubicSpline)
    
    # Y Positions
    y_positions = np.empty(shape=(n_orders, n_traces, nx), dtype=float)
    
    # Loop over orders, possibly multi-trace
    for order_index, single_order_list in enumerate(orders_list):
        
        stopwatch.lap(order_index)
        print('  Extracting Order ' + str(order_index + 1) + ' of ' + str(n_orders) + ' ...')
        
        # Orders are composed of multiple traces
        if len(single_order_list) > 1:
            
            for sub_trace_index, single_trace_dict in enumerate(single_order_list):
                
                stopwatch.lap(sub_trace_index)
                print('    Extracting Sub Trace ' + str(sub_trace_index + 1) + ' of ' + str(len(single_order_list)) + ' ...')
                
                reduced_orders[order_index, sub_trace_index, :, :], boxcar_spectra[order_index, sub_trace_index, :], trace_profile_csplines[order_index, sub_trace_index], y_positions[order_index, sub_trace_index, :] = extract_single_trace(data, data_image, trace_map_image, single_trace_dict, config)
                
                print('    Extracted Sub Trace ' + str(sub_trace_index + 1) + ' of ' + str(len(single_order_list)) + ' in ' + str(round(stopwatch.time_since(sub_trace_index), 3)) + ' min ')
                
        # Orders are composed of single trace
        else:
            reduced_orders[order_index, 0, :, :], boxcar_spectra[order_index, 0, :], trace_profile_csplines[order_index, 0], y_positions[order_index, 0, :] = extract_single_trace(data, data_image, trace_map_image, single_order_list[0], config)
            
        print('  Extracted Order ' + str(order_index + 1) + ' of ' + str(n_orders) + ' in ' + str(round(stopwatch.time_since(order_index) / 60, 3)) + ' min ')

    # Plot and write to fits file
    plot_trace_profiles(data, trace_profile_csplines)
    plot_extracted_spectra(data, reduced_orders, boxcar_spectra)
    
    data.parser.save_reduced_orders(data, reduced_orders)
    
    # Save Trace profiles and refined trace positions
    fname = data.parser.run_output_path + 'trace' + os.sep + data.base_input_file_noext + '_traces.npz'
    np.savez(fname, trace_profiles=trace_profile_csplines, y_positions=y_positions)


# Performs standard extraction
def extract_single_trace(data, data_image, trace_map_image, trace_dict, config, refine_trace_pos=True):
    """Extract a single trace.

    Args:
        data (SpecDataImage): The data to extract.
        data_image (np.ndarray): The corresponding image.
        trace_map_image (np.ndarray): The image trace map image containing labels of each individual trace.
        trace_dict (dict): The dictionary containing location information for this trace
        config (dict): The reduction settings dictionary.
        refine_trace_pos (bool, optional): Whether or not to refine the trace position. Defaults to True.

    Returns:
        np.ndarray: The optimally reduced spectra with shape=(nx, 3)
        np.ndarray: The boxcar reduced spectra with shape=(nx,)
        CubicSpline: The trace profile defined by a CubicSpline object.
        y_positions_refined: The refined trace positions.
    """
    # Stopwatch
    stopwatch = pcutils.StopWatch()
    
    # Image dimensions
    ny, nx = data_image.shape
    
    # Helpful arrays
    xarr, yarr = np.arange(nx), np.arange(ny)
    
    # Extract the oversample factor
    M = config['oversample']
    
    #################################
    ##### Trace Profile & Y Pos #####
    #################################

    # Estimate y trace positions from the given order mapping
    y_positions_estimate = np.polyval(trace_dict['pcoeffs'], xarr)

    # Extract the height of the trace
    height = int(np.ceil(trace_dict['height']))

    # Create trace_image where only the relevant trace is seen, still ny x nx
    trace_image = np.copy(data_image)
    good_data = np.where(np.isfinite(trace_image))
    badpix_mask = np.zeros_like(trace_image)
    badpix_mask[good_data] = 1
    good_trace = np.where(trace_map_image == trace_dict['label'])
    bad_trace = np.where(trace_map_image != trace_dict['label'])
    if bad_trace[0].size > 0:
        trace_image[bad_trace] = np.nan
        badpix_mask[bad_trace] = 0
        
    # Flag obvious bad pixels
    trace_image_smooth = pcmath.median_filter2d(trace_image, width=5)
    med_val = pcmath.weighted_median(trace_image_smooth, percentile=0.99)
    bad = np.where((trace_image < 0) | (trace_image > 2 * med_val))
    if bad[0].size > 0:
        trace_image[bad] = np.nan
        badpix_mask[bad] = 0
        
    # The image in units of PE
    trace_image = convert_image_to_pe(trace_image, config['detector_props'])
    
    print('    Estimating Trace Profile ...', flush=True)
    
    # Estimate the trace profile from the current y positions
    trace_profile_cspline_estimate = estimate_trace_profile(trace_image, y_positions_estimate, height, M=M, mask_edges=[config['mask_trace_edges'], config['mask_trace_edges']])
    
    if refine_trace_pos:
        
        print('    Refining Trace Profile ...', flush=True)
        
        n_iters = 3
        y_positions_refined = np.copy(y_positions_estimate)
        trace_profile_cspline = copy.deepcopy(trace_profile_cspline_estimate)
        
        for iteration in range(n_iters):
    
            # Refine trace position with cross correlation
            y_positions_refined = refine_trace_position(data, trace_image, y_positions_refined, trace_profile_cspline, badpix_mask, height, config, trace_pos_polyorder=config['trace_pos_polyorder'], M=M)
        
            # Now with a possibly better y positions array, re-estimate the trace profile.
            trace_profile_cspline = estimate_trace_profile(trace_image, y_positions_refined, height, M=M, mask_edges=[config['mask_trace_edges'], config['mask_trace_edges']])
        
    else:
        
        # Use the pre-constructed positions and trace profile
        y_positions_refined = y_positions_estimate
        trace_profile_cspline = trace_profile_cspline_estimate
    
  
    ###########################
    ##### Sky Subtraction #####
    ###########################
    
    # Estimate sky and remove from profile
    if config['sky_subtraction']:
        print('    Estimating Background Sky ...', flush=True)
        sky = estimate_sky(trace_image, y_positions_refined, trace_profile_cspline, height, n_sky_rows=config['n_sky_rows'], M=M)
        tp = trace_profile_cspline(trace_profile_cspline.x)
        #_, trace_max = estimate_trace_max(trace_profile_cspline)
        tp -= pcmath.weighted_median(tp, percentile=0.05)
        bad = np.where(tp < 0)[0]
        if bad.size > 0:
            tp[bad] = np.nan
        good = np.where(np.isfinite(tp))[0]
        trace_profile_cspline = scipy.interpolate.CubicSpline(trace_profile_cspline.x[good], tp[good])
    else:
        sky = None
        
    # Determine the fractions of the pixels used
    pixel_fractions = generate_pixel_fractions(trace_image, trace_profile_cspline, y_positions_refined, badpix_mask, height, min_profile_flux=config['min_profile_flux'])
        
    ############################
    #### Optimal Extraction ####
    ############################

    #spec1d, unc1d = slit_decomposition_wrapper(data, trace_image, y_positions_refined, trace_profile_cspline, badpix_mask, pixel_fractions, height, config, config['detector_props'], sky=sky, M=M, n_iters=100)
    
    # The extraction algorithm
    spec_extractor = eval(config['optx_alg'])
    
    print('    Optimally Extracting Trace ...', flush=True)

    spec1d, unc1d, badpix_mask = spec_extractor(data, trace_image, y_positions_refined, trace_profile_cspline, pixel_fractions, badpix_mask, height, config, sky=sky)

    # Also generate a final boxcar spectrum
    boxcar_spectrum1d, _ = boxcar_extraction(trace_image, y_positions_refined, trace_profile_cspline, pixel_fractions, badpix_mask, height, config, config['detector_props'], exp_time=data.itime, sky=sky, n_sky_rows=config['n_sky_rows'])
    
    # Generate a final badpix array and flag obvious bad pixels
    badpix1d = np.ones(nx, dtype=int)
    bad = np.where(~np.isfinite(spec1d) | ~np.isfinite(unc1d))[0]
    if bad.size > 0:
        badpix1d[bad] = 0
        spec1d[bad] = np.nan
        unc1d[bad] = np.nan
        boxcar_spectrum1d[bad] = np.nan
        
    # Flag according to outliers in 1d spectrum
    thresh = 0.3
    spec1d_smooth = pcmath.median_filter1d(spec1d, 3)
    continuum = pcmath.weighted_median(spec1d_smooth, percentile=0.99)
    bad = np.where(np.abs(spec1d_smooth - spec1d) / continuum > thresh)[0]
    if bad.size > 0:
        spec1d[bad] = np.nan
        unc1d[bad] = np.nan
        badpix1d[bad] = 0
    
    # Bring together the science spectrum
    reduced_spectrum = np.empty(shape=(nx, 3), dtype=float)
    reduced_spectrum[:, 0], reduced_spectrum[:, 1], reduced_spectrum[:, 2] = spec1d, unc1d, badpix1d
    
    return reduced_spectrum, boxcar_spectrum1d, trace_profile_cspline, y_positions_refined
    
def boxcar_extraction(trace_image, y_positions, trace_profile_cspline, pixel_fractions, badpix_mask, height, config, detector_props, exp_time, sky=None, n_sky_rows=None):
    """Performs a boxcar extraction on the nonrectified data.

    Args:
        trace_image (np.ndarray): The masked data image.
        y_positions (np.ndarray): The trace positions for each column.
        trace_profile_cspline (CubicSpline): The trace profile defined by a CubicSpline object.
        pixel_fractions (np.ndarray): The fractions of each pixel to use.
        height (int): The height of the trace.
        config (dict): The reduction settings dictionary.
        detector_props (list): List of detector properties to properly calculate read noise.
        exp_time (float): The exposure time.
        badpix_mask (np.ndarray): The bad pixel image mask (1=good, 0=bad).
        sky (np.ndarray): The sky background as a function of detector x-pixels (1-dimensional), defaults to None (no sky subtraction).
        n_sky_rows (int): The number of rows used to determine the sky background.
    Returns:
        np.ndarray: The boxcar extracted 1-dimensional spectrum.
        np.ndarray: The corresponding uncertainty.
    """
    
    # image dims
    ny, nx = trace_image.shape
    
    # Trace profile
    trace_profile_fiducial_grid, trace_profile = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
    good_trace_profile = np.where(np.isfinite(trace_profile))[0]
    
    # Sky background
    if sky is not None:
        sky_err = np.sqrt(sky / (n_sky_rows - 1))
    else:
        sky = np.zeros(nx)
        sky_err = sky

    # Storage arrays
    spec = np.full(nx, fill_value=np.nan, dtype=np.float64)
    spec_unc = np.full(nx, fill_value=np.nan, dtype=np.float64)
    corrections = np.full(nx, fill_value=np.nan, dtype=np.float64)
    
    yarr = np.arange(ny)
    
    badpix_maskcp = np.copy(badpix_mask)
    
    #trace_max_pos, _ = estimate_trace_max(trace_profile_cspline)

    for x in range(nx):
        
        # Define views
        badpix_x = badpix_maskcp[:, x] # Bad pixels (1=good, 0=bad)
        data_x = trace_image[:, x] # The data (includes sky) in units of PEs

        if sky is not None:
            S_x = data_x - sky[x] # Star is trace - sky
        else:
            S_x = np.copy(data_x)
        
        # Flag negative values after sky subtraction
        negs = np.where(S_x < 0)[0]
        
        if negs.size > 0:
            S_x[negs] = np.nan
            badpix_x[negs] = 0
            
        if np.all(~np.isfinite(S_x)) or np.nansum(badpix_x) == 0:
            continue
        
        # Effective read noise
        eff_read_noise = compute_read_noise(detector_props, x, y_positions[x], exp_time, dark_subtraction=config['dark_subtraction'])
        
        # Determine the full and fractional pixels to use
        good_full_pixels = np.where(pixel_fractions[:, x] == 1)[0]
        fractional_pixels = np.where((pixel_fractions[:, x] > 0) & (pixel_fractions[:, x] < 1))[0]
        all_pixels = np.where(pixel_fractions[:, x] > 0)[0]
        
        # shift the trace profile
        P_x = scipy.interpolate.CubicSpline(trace_profile_fiducial_grid[good_trace_profile] + y_positions[x], trace_profile[good_trace_profile], extrapolate=False)(yarr)
        
        # Now construct P from the fractions
        #frac_left, frac_right = pixel_fractions[fractional_pixels[0], x], pixel_fractions[fractional_pixels[-1], x]
        #P_use = np.concatenate(([frac_left * trace_profile_shifted[fractional_pixels[0]]], trace_profile_shifted[good_full_pixels], [frac_right * trace_profile_shifted[fractional_pixels[-1]]]))
        #S_use = np.concatenate(([frac_left * S_x[fractional_pixels[0]]], S_x[good_full_pixels], [frac_right * S_x[fractional_pixels[-1]]]))
        #badpix_use = badpix_x[good_full_pixels[0]-1:good_full_pixels[-1]+2]
        
        P_use = P_x[all_pixels]
        badpix_use = badpix_x[all_pixels]
        S_use = S_x[all_pixels]
        
        # Determine which pixels to use from the trace alone
        P_use /= np.nansum(P_use)
        
        # Variance
        var_use = eff_read_noise**2 + S_use + sky[x] + sky_err[x]**2
        
        # Weights = bad pixels only
        weights_use = np.copy(badpix_use)
        
        # Normalize the weights such that sum=1
        weights_use /= np.nansum(weights_use)
        
        good = np.where(weights_use > 0)[0]
        if good.size <= 1:
            continue
        
        # 1d final flux at column x
        corrections[x] = np.nansum(P_use * weights_use)
        spec[x] = np.nansum(S_use * weights_use) / corrections[x]
        spec_unc[x] = np.sqrt(np.nansum(var_use)) / corrections[x]

    return spec, spec_unc


def flag_bad_pixels(trace_image, current_spectrum, y_positions, trace_profile_cspline, pixel_fractions, badpix_mask, height, sky=None, nsig=6):
    """Flags bad pixels in the data by smoothing the 1d flux and convolving it into 2d space and looking for outliers.

    Args:
        trace_image (np.ndarray): The masked data image.
        current_spectrum (np.ndarray): The current 1d spectrum.
        y_positions (np.ndarray): The trace positions for each column.
        trace_profile_cspline (CubicSpline): The trace profile defined by a CubicSpline object.
        pixel_fractions (np.ndarray): The fractions of each pixel to use.
        badpix_mask (np.ndarray): The bad pixel image mask (1=good, 0=bad).
        sky (np.ndarray): The sky background as a function of detector x-pixels (1-dimensional), defaults to None (no sky subtraction).
        nsig (float): Flags pixels more deviant that nsig*rms of the convovled smooth spectrum.
    Returns:
        np.ndarray: The updated bad pixel mask.
    """
    
    # Image dimensions
    ny, nx = trace_image.shape
    
    # Don't modify current mask, return new one
    badpix_maskcp = np.copy(badpix_mask)
    
    # Trace profile on fiducial grid
    trace_profile_fiducial_grid, trace_profile = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
    good_trace = np.where(np.isfinite(trace_profile))[0]
    
    # Sky
    if sky is None:
        sky = np.zeros(nx)
        
    # Smooth the current spectrum
    current_spec_smooth = pcmath.median_filter1d(current_spectrum, 5)
    continuum = pcmath.weighted_median(current_spec_smooth, percentile=0.95)
    
    # Deviations
    deviations = np.empty(shape=(ny, nx), dtype=np.float64) + np.nan
    
    goody, goodx = np.where(np.isfinite(badpix_mask))
    ys, yf = np.min(goody), np.max(goody)
    
    yarr = np.arange(ny)
    
    for x in range(nx):
        
        # Shift the trace profile
        trace_cspline_shifted = scipy.interpolate.CubicSpline(trace_profile_fiducial_grid[good_trace] + y_positions[x], trace_profile[good_trace], extrapolate=False)
        ytrace = trace_profile_fiducial_grid[good_trace] + y_positions[x]
        P_x = trace_cspline_shifted(trace_cspline_shifted.x)
        good = ytrace[np.where(np.isfinite(P_x))[0]]
        if good.size < 3 or np.nanmin(good) < ys or np.nanmax(good) > yf or np.nansum(badpix_maskcp[:, x]) < 3 or not np.isfinite(current_spec_smooth[x]):
            continue
        
        S_x = trace_image[:, x] - sky[x]
        badpix_x = badpix_mask[:, x]
        
        # Determine the full and fractional pixels to use
        good_full_pixels = np.where(pixel_fractions[:, x] == 1)[0]
        fractional_pixels = np.where((pixel_fractions[:, x] > 0) & (pixel_fractions[:, x] < 1))[0]
        all_pixels = np.where(pixel_fractions[:, x] > 0)[0]
        bad = np.where(pixel_fractions[:, x] == 0)[0]
        
        # Shift the trace profile
        P_x = scipy.interpolate.CubicSpline(trace_profile_fiducial_grid[good_trace] + y_positions[x], trace_profile[good_trace], extrapolate=False)(yarr)
        
        # Now construct P from the fractions
        #frac_left, frac_right = pixel_fractions[fractional_pixels[0], x], pixel_fractions[fractional_pixels[-1], x]
        #P_use = np.concatenate(([frac_left * trace_profile_shifted[fractional_pixels[0]]], trace_profile_shifted[good_full_pixels], [frac_right * trace_profile_shifted[fractional_pixels[-1]]]))
        #S_use = np.concatenate(([S_x[fractional_pixels[0]]], S_x[good_full_pixels], [S_x[fractional_pixels[-1]]]))
        
        S_x[bad] = np.nan
        P_x[bad] = np.nan
        badpix_x[bad] = 0
        
        # Convolve into 2d space
        P_x /= np.nansum(P_x)
        spec_conv = P_x * current_spec_smooth[x]
        
        # Deviations between convolved 1d spectrum and true 2d data.
        diffs = ((S_x - spec_conv) / continuum)**2 * badpix_x
        good = np.where(np.isfinite(diffs))[0]
        w = badpix_x / P_x**2
        if good.size < 3:
            continue
        deviations[:, x] = diffs
        
    
    deviations_smooth = pcmath.median_filter2d(deviations, width=3)
    ng = np.where(np.isfinite(deviations_smooth))[0].size
    rms = np.sqrt(np.nansum(deviations_smooth**2) / ng)
    bad = np.where(np.abs(deviations) > nsig*rms)

    if bad[0].size > 0:
        badpix_maskcp[bad] = 0
    
    return badpix_maskcp


def rectify_trace(trace_image, y_positions, height, M=1):
    """Rectifies (straightens) the trace via cubic spline interpolation.

    Args:
        trace_image (np.ndarray): The masked data image.
        y_positions (np.ndarray): The trace positions for each column.
        M (int): The desired oversample factor, defaults to 1.
    Returns:
        np.ndarray: The rectified trace image
    """
    
    # The data shape
    ny, nx = trace_image.shape
    
    # Low and high res grids
    yarr1 = np.arange(ny)
    yarr2 = np.arange(-height / 2 , height / 2 + 1, 1/M)
    n2 = len(yarr2)

    # Shift
    trace_image_rectified = np.empty(shape=(n2, nx)) + np.nan
    
    for x in range(nx):
        good = np.where(np.isfinite(trace_image[:, x]))[0]
        if good.size <= 3:
            continue
        trace_image_rectified[:, x] = scipy.interpolate.CubicSpline(yarr1[good] - y_positions[x], trace_image[good, x], extrapolate=False, bc_type='clamped')(yarr2)
    
    return trace_image_rectified
    

def estimate_sky(trace_image, y_positions, trace_profile_cspline, height, n_sky_rows=8, M=1):
    """Estimates the sky background, sky(x).

    Args:
        trace_image (np.ndarray): The masked data image.    
        y_positions (np.ndarray): The trace positions for each column.
        trace_profile_cspline (CubicSpline): The trace profile defined by a CubicSpline object.
        height (int): The height of the trace.
        n_sky_rows (int, optional): The number of rows used to determine the sky background, defaults to 8.
        M (int): The desired oversampling factor.
    Returns:
        np.ndarray: The computed background sky, sky(x).
    """
    
    # The image dimensions
    ny, nx = trace_image.shape
    
    # Smooth the image
    trace_image_smooth = pcmath.median_filter2d(trace_image, width=5)
    
    # Rectify
    trace_image_smooth_rectified = rectify_trace(trace_image_smooth, y_positions, height, M=M)
    nyr, nxr = trace_image_smooth_rectified.shape
    
    # rectified hr grid
    yarrhr = np.copy(trace_profile_cspline.x)

    # Estimate the sky by considering N rows of a rectifed trace on a high resolution grid to minimize sampling artifacts.
    trace_profile = trace_profile_cspline(yarrhr)
    good = np.where(np.isfinite(trace_profile))[0]
    sky_locs = np.argsort(trace_profile)[0:int(n_sky_rows*M)]
    
    # Estimate the sky background from this smoothed image

    sky_init = np.nanmedian(trace_image_smooth_rectified[sky_locs, :], axis=0)
    
    # Smooth the sky again
    sky_out = np.copy(sky_init)
    good = np.where(np.isfinite(sky_init))[0]
    sky_out[good] = scipy.signal.savgol_filter(sky_init[good], 17, 3)
    
    return sky_out


def pmassey_wrapper(data, trace_image, y_positions, trace_profile_cspline, pixel_fractions, badpix_mask, height, config, sky=None):
    """A wrapper for Philip Massey extraction.

    Args:
        data (SpecDataImage): The SpecData Image
        trace_image (np.ndarray): The corresponding masked trace_image to extract.
        y_positions (np.ndarray): The trace positions for each column.
        trace_profile_cspline ([type]): [description]
        pixel_fractions (np.ndarray): The fractions of each pixel to use.
        badpix_mask ([type]): [description]
        height ([type]): [description]
        config (dict): The reduction settings dictionary.
        sky (np.ndarray): The sky background as a function of detector x-pixels (1-dimensional), defaults to None (no sky subtraction).
    Returns:
        np.ndarray: The 1d flux in units of PE
        np.ndarray: The 1d flux uncertainty in units of PE
    """

    n_iters = len(config['pmassey_settings']['bad_thresh']) + 1

    for iteration in range(n_iters):
        # Do the optimal extraction then flag bad pixels
        spec1d, unc1d = optimal_extraction_pmassey(trace_image, y_positions, trace_profile_cspline, pixel_fractions, badpix_mask, height, config, config['detector_props'], exp_time=data.itime, sky=sky, n_sky_rows=config['n_sky_rows'])
        
        if iteration + 1 < n_iters:
            badpix_mask = flag_bad_pixels(trace_image, spec1d, y_positions, trace_profile_cspline, pixel_fractions, badpix_mask, height, sky=sky, nsig=config['pmassey_settings']['bad_thresh'][iteration])

    return spec1d, unc1d, badpix_mask


def optimal_extraction_pmassey(trace_image, y_positions, trace_profile_cspline, pixel_fractions, badpix_mask, height, config, detector_props, exp_time, sky=None, n_sky_rows=None):
    """Performs optimal extraction on the nonrectified data.

    Args:
        trace_image (np.ndarray): The masked data image.
        y_positions (np.ndarray): The trace positions for each column.
        trace_profile_cspline (CubicSpline): The trace profile defined by a CubicSpline object.
        pixel_fractions (np.ndarray): The fractions of each pixel to use.
        badpix_mask (np.ndarray): The bad pixel image mask (1=good, 0=bad).
        height (int): The height of the order.
        config (dict): The reduction settings dictionary.
        detector_props (list): List of detector properties to properly calculate read noise.
        exp_time (float): The exposure time.
        sky (np.ndarray): The sky background as a function of detector x-pixels (1-dimensional), defaults to None (no sky subtraction).
        n_sky_rows (int): The number of rows used to determine the sky background, defaults to None.
    Returns:
        spec (np.ndarray): The optimally extracted 1-dimensional spectrum.
        spec_unc (np.ndarray): The corresponding uncertainty.
    """
    
    # Trace profile
    trace_profile_fiducial_grid, trace_profile = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
    good_trace_profile = np.where(np.isfinite(trace_profile))[0]
    
    # image dims
    ny, nx = trace_image.shape
    
    # Sky background
    if sky is not None:
        sky_err = np.sqrt(sky / (n_sky_rows - 1))
    else:
        sky = np.zeros(nx)
        sky_err = sky

    # Storage arrays
    spec = np.full(nx, fill_value=np.nan, dtype=np.float64)
    spec_unc = np.full(nx, fill_value=np.nan, dtype=np.float64)
    corrections = np.full(nx, fill_value=np.nan, dtype=np.float64)
    
    yarr = np.arange(ny)
    
    badpix_maskcp = np.copy(badpix_mask)

    for x in range(nx):
        
        # Define views
        badpix_x = badpix_maskcp[:, x] # Bad pixels (1=good, 0=bad)
        data_x = trace_image[:, x] # The data (includes sky) in units of PEs

        if sky is not None:
            S_x = data_x - sky[x] # Star is trace - sky
        else:
            S_x = np.copy(data_x)
        
        # Flag negative values after sky subtraction
        negs = np.where(S_x < 0)[0]
        
        if negs.size > 0:
            S_x[negs] = np.nan
            badpix_x[negs] = 0
            
        if np.all(~np.isfinite(S_x)) or np.nansum(badpix_x) == 0:
            continue
        
        # Effective read noise
        eff_read_noise = compute_read_noise(detector_props, x, y_positions[x], exp_time, dark_subtraction=config['dark_subtraction'])
        
        # Determine the full and fractional pixels to use
        good_full_pixels = np.where(pixel_fractions[:, x] == 1)[0]
        fractional_pixels = np.where((pixel_fractions[:, x] > 0) & (pixel_fractions[:, x] < 1))[0]
        all_pixels = np.where(pixel_fractions[:, x] > 0)[0]
        
        # Shift the trace profile
        P_x = scipy.interpolate.CubicSpline(trace_profile_fiducial_grid[good_trace_profile] + y_positions[x], trace_profile[good_trace_profile], extrapolate=False)(yarr)
        
        P_use = P_x[all_pixels] # pixel_fractions[all_pixels, x]
        badpix_use = badpix_x[all_pixels]
        S_use = S_x[all_pixels] # * pixel_fractions[all_pixels, x]
        
        # Determine which pixels to use from the trace alone
        P_use /= np.nansum(P_use)
        
        # Variance
        var_use = eff_read_noise**2 + S_use + sky[x] + sky_err[x]**2
        
        # Weights = bad pixels only
        weights_use = P_use**2 / var_use * badpix_use
        
        # Normalize the weights such that sum=1
        weights_use /= np.nansum(weights_use)
        
        good = np.where(weights_use > 0)[0]
        if good.size <= 1:
            continue
        
        # 1d final flux at column x
        corrections[x] = np.nansum(P_use * weights_use)
        spec[x] = np.nansum(S_use * weights_use) / corrections[x]
        spec_unc[x] = np.sqrt(np.nansum(var_use)) / corrections[x]

    return spec, spec_unc


def fit_2d_wrapper(trace_image, y_positions, trace_profile_cspline, pixel_fractions, badpix_mask, height, config, detector_props, exp_time, sky=None, M=16, n_iters=100, n_chunks=5):
    
    ny, nx = trace_image.shape
    
    goody, goodx = np.where(badpix_mask == 1)
    x_start, x_end = goodx[0], goodx[-1]
    y_start, y_end = goody[0], goody[-1]
    x_chunks = np.linspace(x_start, x_end, num=n_chunks+1).astype(int)
    
    xarr = np.arange(nx)
    yarr = np.arange(ny)
    
    for ichunk in range(n_chunks):
        
        chunk_x_start, chunk_x_end = x_chunks[ichunk], x_chunks[ichunk + 1]
        goody_chunk, _ = np.where(badpix_mask[:, chunk_x_start:chunk_x_end] == 1)
        chunk_y_start, chunk_y_end = goody_chunk[0], goody_chunk[-1]
        
        trace_image_chunk = trace_image[chunk_y_start:chunk_y_end, chunk_x_start:chunk_x_end]
        badpix_mask_chunk = badpix_mask[chunk_y_start:chunk_y_end, chunk_x_start:chunk_x_end]
        y_positions_chunk = y_positions[chunk_x_start:chunk_x_end] - chunk_y_start
        pixel_fractions_chunk = pixel_fractions[chunk_y_start:chunk_y_end, chunk_x_start:chunk_x_end]
        sky_chunk = sky[chunk_x_start:chunk_x_end]
        fit_result = fit_2d_chunk_modgauss(trace_image_chunk, y_positions_chunk, trace_profile_cspline, badpix_mask_chunk, pixel_fractions_chunk, height, config, detector_props, sky=sky_chunk, fit_profile=True, fit_sky=False, fit_ypos=True)
        
        ypos = optimal_extraction_pmassey()
    
    

def fit_2d_chunk_modgauss(trace_image, y_positions, trace_profile_cspline, badpix_mask, pixel_fractions, height, config, detector_props, sky=None, fit_profile=True, fit_sky=False, fit_ypos=True):
    
    # The chunk shape
    ny, nx = trace_image.shape
    
    # Good pixels
    goody, goodx = np.where(pixel_fractions == 1)
    
    # The trace image position determined by a quadratic
    if fit_ypos:
        trace_pos_poly_order = 2
        n_trace_pos_pars = trace_pos_poly_order + 1
        ypos_xsample = np.array([goodx[10], goodx.size / 2 - 1, goodx[-10]])
        ypos_pars = np.copy(y_positions[ypos_xsample])
        ypos_pars_bounds = [(ypos_pars[0] - 5, ypos_pars[0] + 5), (ypos_pars[1] - 5, ypos_pars[1] + 5), (ypos_pars[2] - 5, ypos_pars[2] + 5)]
        ypos_par_inds = (0, 1, 2)
    else:
        trace_pos_poly_order = None
        n_trace_pos_pars = 0
        ypos_pars = []
        ypos_par_inds = None
    
    # Trace Pars
    if fit_profile:
        n_trace_profile_pars = 2
        trace_profile = trace_profile_cspline(trace_profile_cspline.x)
        trace_profile /= np.nanmax(trace_profile)
        left_cut = np.max(np.where((trace_profile < 0.5) & trace_profile_cspline.x < 0)[0])
        right_cut = np.min(np.where((trace_profile < 0.5) & trace_profile_cspline.x > 0)[0])
        sigma_guess = (right_cut - left_cut) / 2.355
        d_guess = 2
        trace_profile_pars = [sigma_guess, d_guess]
        trace_profile_bounds = [(sigma_guess * 0.5, sigma_guess * 1.5), (1, 3)]
        trace_profile_par_inds = (n_trace_pos_pars, n_trace_pos_pars + 1, n_trace_pos_pars + 2)
    else:
        trace_profile_par_inds = None
        n_trace_profile_pars = 0
        
    # Sky Pars
    if fit_sky:
        sky_poly_order = 2
        n_sky_pars = sky_poly_order + 1
        sky_xsample = np.array([goodx[10], goodx.size / 2 - 1, goodx[-10]])
        sky_pars = np.copy(sky[sky_xsample])
        sky_pars_bounds = [(sky_pars[0] - 5, sky_pars[0] + 5), (sky_pars[1] - 5, sky_pars[1] + 5), (sky_pars[2] - 5, sky_pars[2] + 5)]
        sky_par_inds = (n_trace_pos_pars + n_trace_profile_pars, n_trace_pos_pars + n_trace_profile_pars + 1, n_trace_pos_pars + n_trace_profile_pars + 2)
    else:
        sky_par_inds = None
        n_sky_pars = 0
        
    # Optimize
    xarr, yarr = np.arange(nx), np.arange(ny)
    init_pars = np.array(ypos_pars + trace_profile_pars + sky_pars)
    bounds = ypos_pars_bounds + trace_profile_pars_bounds + sky_pars_bounds
    args = (trace_image, xarr, yarr, spec1d, pixel_fractions, weights, trace_profile_cspline, y_positions, sky, ypos_par_inds, profile_par_inds, sky_par_inds)
    result = scipy.optimize.minimize(fit_2d_chunk_solver, init_pars, bounds=bounds, tol=1E-6, method='Powell', args=args)
    best_pars = result.x
    
    # Extract pars
    ypos_pars = best_pars[ypos_par_inds] if fit_ypos else None
    trace_profile_pars = best_pars[trace_profile_par_inds] if fit_profile else None
    sky_pars = best_pars[sky_par_inds] if fit_sky else None
        
    return ypos_pars, trace_profile_pars, sky_pars
    
    

def fit_2d_chunk_solver(pars, trace_image, xarr, yarr, spec1d, pixel_fractions, weights, trace_profile_cspline=None, y_positions=None, sky=None, ypos_par_inds=None, profile_par_inds=None, sky_par_inds=None):
    
    # The image dimensions
    ny, nx = trace_image.shape
    
    # y positions
    if ypos_par_inds is not None:
        ypos = np.polyval(pars[0:3], xarr)
    else:
        ypos = y_positions
    
    # profile params
    if profile_par_inds is not None:
        sigma, d = pars[3], pars[4]
        
    
    # Background params
    if sky_par_inds is not None:
        B = np.polyval(pars[5:8], xarr)
    else:
        B = sky
        
    # Model
    model = np.full(shape=(ny, nx), fill_value=np.nan)
    
    for x in range(nx):
        
        # Good and bad
        bad = np.where(pixel_fractions[:, x] <= 1)[0]
        good = np.where(pixel_fractions[:, x] == 1)[0]
        
        if good.size == 0:
            continue
        
        # Build Profile
        if trace_profile_cspline is None:
            P = pcmath.gauss_modified(xarr[good], 1, ypos[x], sigma, d)
        else:
            P = scipy.interpolate.CubicSpline(trace_profile_cspline.x + ypos[x], trace_profile_cspline(trace_profile_cspline.x))(xarr[good])
            
        P /= np.nansum(P)
        model[good, x] = P * spec1d[x] + B[x]
        
    good = np.where((weights > 0))[0]
    nflag = 50
    wdiffs2 = (weights[good] * (trace_image[good] - model[good])**2).flatten().sort()
    wdiffs2[-nflag:] = 0
    chi2 = np.nansum(wdiffs2) / (good.size - nflag - 1)
    return chi2


def fit_2d_chunk_(pars, trace_image, xarr, yarr, yarrhr, spec1d, pixel_fractions, weights):
    
    # The image dimensions
    ny, nx = trace_image.shape
    
    # y positions
    y = np.polyval(pars[0:3], xarr)
    
    # profile params
    sigma, d = pars[3], pars[4]
    
    # Background params
    B = np.polyval(pars[5:8], xarr)
    
    for x in prange(nx):
        P = pcmath.gauss_modified(xarrhr, 1, y[x], sigma, d)
        P = scipy.interpolate.CubicSpline(xarrhr, P, extrapolate=False)(yarr)
        good = np.where(pixel_fractions[:, x] == 1)[0]
        if good.size == 0:
            continue
        P = P[good]
        P /= np.nansum(P)
        star = P * spec1d
        model[:, x] = star + B[x]
        
    wdiffs2 = weights * (trace_image - model)**2
    good = np.where((weights > 0) & np.isfinite(wdiffs2))[0]
    wdiffs2 = wdiffs2[good].flatten()
    wdiffs2 = np.sort(wdiffs2)
    wdiffs2[-50:] = 0
    chi2 = np.nansum(wdiffs2) / (ng - 50)
    return chi2


def generate_pixel_fractions(trace_image, trace_profile_cspline, y_positions, badpix_mask, height, min_profile_flux=0.05):
    """Computes the fraction of each pixel to use according to a minumum profile flux.

    Args:
        trace_image (np.ndarray): The masked data image.
        trace_profile_cspline (CubicSpline): The trace profile defined by a CubicSpline object.
        y_positions (np.ndarray): The trace positions for each column.
        badpix_mask (np.ndarray): The bad pixel image mask (1=good, 0=bad).
        height (int): The height of the trace.
        min_profile_flux (float, optional): The minimum flux to consider in the trace profle. Defaults to 0.05 (~ 5 percent).

    Returns:
        (np.ndarray): The fractions of each pixel to use.
    """
    
    # Image dimensions
    ny, nx = trace_image.shape
    
    # Determine the left and right cut
    trace_profile_fiducial_grid, trace_profile = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
    
    # The left and right profile positions
    left_trace_profile_inds = np.where(trace_profile_fiducial_grid < -1)[0]
    right_trace_profile_inds = np.where(trace_profile_fiducial_grid > 1)[0]
    left_trace_profile_ypos= trace_profile_fiducial_grid[left_trace_profile_inds]
    right_trace_profile_ypos = trace_profile_fiducial_grid[right_trace_profile_inds]
    left_trace_profile = trace_profile[left_trace_profile_inds]
    right_trace_profile = trace_profile[right_trace_profile_inds]
    
    # Find where it intersections at some minimum flux value
    left_ycut, _ = pcmath.intersection(left_trace_profile_ypos, left_trace_profile, min_profile_flux, precision=1000)
    right_ycut, _ = pcmath.intersection(right_trace_profile_ypos, right_trace_profile, min_profile_flux, precision=1000)
    
    # Initiate y arr, pix fractions
    yarr = np.arange(ny)
    pixel_fractions = np.zeros_like(trace_image)
    
    # Good locations
    goody, goodx = np.where(np.isfinite(badpix_mask))
    ys, yf = np.min(goody), np.max(goody)
    
    # Good trace profile
    good_trace = np.where(np.isfinite(trace_profile))[0]

    for x in range(nx):
        
        # New y arr for this column
        ytrace = trace_profile_fiducial_grid[good_trace] + y_positions[x]
        
        # Shift the trace profile
        trace_cspline_shifted = scipy.interpolate.CubicSpline(ytrace, trace_profile[good_trace], extrapolate=False)
        profile_x = trace_cspline_shifted(ytrace)
        good = ytrace[np.where(np.isfinite(profile_x))[0]]
        if good.size < 3 or np.nanmin(good) < ys or np.nanmax(good) > yf:
            continue
        
        # Left cuts
        ysl = left_ycut + y_positions[x]
        yl1, yl2 = int(np.floor(ysl)), int(np.ceil(ysl))
        if ysl - 0.5 == yl1:
            frac_left = 0
        elif ysl - 0.5 > yl1:
            frac_left = yl2 - ysl + 0.5
        else:
            frac_left = 0.5 - (ysl - yl1)
            
        # Right cuts
        ysr = right_ycut + y_positions[x]
        yr1, yr2 = int(np.floor(ysr)), int(np.ceil(ysr))
        if ysr + 0.5 == yr2:
            frac_right = 0
        elif ysr + 0.5 > yr2:
            frac_right = yr2 - ysr + 0.5
        else:
            frac_right = (yr2 - ysr) - 0.5
        
        full_pixels = np.arange(yl2, yr1+1, 1).astype(int)
        fractional_pixels = np.array([yl1, yr2])
        pixel_fractions[full_pixels, x] = 1
        pixel_fractions[fractional_pixels[0], x] = frac_left
        pixel_fractions[fractional_pixels[1], x] = frac_right

    return pixel_fractions


def compute_read_noise(detector_props, x, y, exp_time, dark_subtraction=False):
    """Computes the read noise according to:
    
          ron(x, y) + dark_current(x, y) * exp_time

    Args:
        detector_props (list): List of detector properties to properly calculate read noise.
        x (float): The x point to consider
        y (float): The y point to consider.
        exp_time (float): The exposure time.
        dark_subtraction (bool, optional): Whether or not dark subtraction was performed. If True, the dark current will not be included in the read noise calculation. Defaults to False.

    Returns:
        float: The read noise
    """
    
    # Get the detector
    detector = get_detector(detector_props, x, y)
    
    # Detector read noise
    if dark_subtraction:
        eff_read_noise = detector['read_noise']
    else:
        
        return detector['read_noise'] + detector['dark_current'] * exp_time

def get_detector(detector_props, x, y):
    """ Determines which detector a given point is on.

    Args:
        detector_props ([type]): [description]
        x (float): The x point to test.
        y (float): The y point to test.

    Returns:
        dict: The correct detector.
    """
    
    if len(detector_props) == 1:
        return detector_props[0]
    for detector in detector_props:
        if (detector['xmin'] < x < detector['xmin']) and (detector['ymin'] < y < detector['ymin']):
            return detector
        
    return ValueError("Point (" + str(x) + str(y) + ") not part of any detector !")

def convert_image_to_pe(trace_image, detector_props):
    """ Converts an image to photo electrons, approximately.

    Args:
        trace_image (np.ndarray): The masked data image.
        detector_props (list): List of detector properties.

    Returns:
        np.ndarray: The image converted to PE.
    """
    
    if len(detector_props) == 1:
        return trace_image * detector_props[0]['gain']
    else:
        trace_image_pe = np.empty_like(trace_image) + np.nan
        for detector in detector_props:
            xmin, xmax, ymin, ymax = detector['xmin'], detector['xmax'], detector['ymin'], detector['ymax']
            trace_image_pe[ymin:ymax+1, xmin:xmax + 1] = trace_image[ymin:ymax+1, xmin:xmax + 1] * detector['gain']

def plot_trace_profiles(data, trace_profile_csplines):
    """Plots the trace profiles.

    Args:
        data (SpecDataImage): The corresponding data object.
        trace_profile_csplines (list): The list of CubicSpline objects.
    """
    
    # The numbr of orders and traces
    n_orders = trace_profile_csplines.shape[0]
    n_traces = trace_profile_csplines.shape[1]
    
    # Plot settings
    plot_width = 20
    plot_height = 20
    dpi = 300
    n_cols = 3
    n_rows = int(np.ceil(n_orders / n_cols))
    
    # Create a figure
    fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(plot_width, plot_height), dpi=dpi)
    
    # For each subplot, plot each single trace
    for row in range(n_rows):
        for col in range(n_cols):
            
            # The order index
            o = n_cols * row + col
            order_num = o + 1
            if order_num > n_orders:
                continue
            
            for itrace in range(n_traces):
                
                # Generate the trace profile
                if trace_profile_csplines[o, itrace] is None:
                    continue
                grid, tp = trace_profile_csplines[o, itrace].x, trace_profile_csplines[o, itrace](trace_profile_csplines[o, itrace].x)
                
                good = np.where(np.isfinite(tp))[0]
                if good.size == 0:
                    continue
                f, l = good[0], good[-1] + 1
                
                # Plot the Trace profile for each trace
                axarr[row, col].plot(grid[f:l], tp[f:l] / np.nanmax(tp[f:l]) + itrace, color='black', lw=1)
                
            # Title
            axarr[row, col].set_title('Order ' + str(order_num))
            
            
    # X and Y labels
    fig.text(0.5, 0.01, 'Y Pixels', fontweight='bold', fontsize=14)
    fig.text(0.01, 0.5, 'Norm. Flux', fontweight='bold', fontsize=14, rotation=90)
    
    # Try a tight layout, may fail
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.4)
    
    # Create a filename
    fname = data.parser.run_output_path + 'trace' + os.sep + data.base_input_file_noext + '_trace_profiles.png'
    
    # Save
    plt.savefig(fname)
    plt.close()

def estimate_snr(trace_profile_cspline, M=1):
    """[summary]

    Args:
        trace_profile_cspline (CubicSpline): The trace profile defined by a CubicSpline object.
        M (int, optional): The desired oversample factor. Defaults to 1.

    Returns:
        float: The approximate S/N of the observation.
    """
    
    tp = trace_profile_cspline(trace_profile_cspline.x) / M
    snr = np.sqrt(np.nansum(tp))
    return snr

def plot_extracted_spectra(data, reduced_orders, boxcar_spectra):
    
    # The numbr of orders and traces
    n_orders = reduced_orders.shape[0]
    n_traces = reduced_orders.shape[1]
    
    # The number of x pixels
    xpixels = np.arange(reduced_orders[0, 0, :, 0].size)
    
    # Plot settings
    plot_width = 20
    plot_height = 20
    dpi = 300
    n_cols = 3
    n_rows = int(np.ceil(n_orders / n_cols))
    
    # Create a figure
    fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(plot_width, plot_height), dpi=dpi)
    
    # For each subplot, plot each single trace
    for row in range(n_rows):
        for col in range(n_cols):
            
            # The order index
            o = n_cols * row + col
            order_num = o + 1
            if order_num > n_orders:
                continue
            
            for itrace in range(n_traces):
                
                # Extract the bad pixel array
                badpix = reduced_orders[o, itrace, :, 2]
                flux_opt = reduced_orders[o, itrace, :, 0]
                flux_box = boxcar_spectra[o, itrace, :]
                
                good = np.where(badpix == 1)[0]
                if good.size == 0:
                    continue
                f, l = good[0], good[-1] + 1
                
                
                # Plot the boxcar extracted spectrum
                axarr[row, col].plot(xpixels[f:l], flux_box[f:l] / pcmath.weighted_median(flux_box[f:l], percentile=0.99) + itrace, color='red', label='Boxcar', lw=0.5)
                
                # Plot the optimally extracted spectrum
                axarr[row, col].plot(xpixels[f:l], flux_opt[f:l] / pcmath.weighted_median(flux_opt[f:l], percentile=0.99) + itrace, color='black', label='Optimal', lw=0.5)
                
            # Title
            axarr[row, col].set_title('Order ' + str(order_num))
            axarr[row, col].legend(loc='upper right', prop={'size': 4})
            
            
    # X and Y labels
    fig.text(0.5, 0.01, 'X Pixels', fontweight='bold', fontsize=14)
    fig.text(0.01, 0.5, 'Norm. Flux', fontweight='bold', fontsize=14, rotation=90)
    
    # Try a tight layout, may fail
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.4)
    
    # Create a filename
    fname = data.parser.run_output_path + 'spectra' + os.sep + data.base_input_file_noext + '_' + data.target.replace(' ', '_') + '_preview.png'
    
    # Save
    plt.savefig(fname)
    plt.close()
    
    
def refine_trace_position(data, trace_image, y_positions, trace_profile_cspline, badpix_mask, height, config, trace_pos_polyorder=2, M=1):
    """Refines the trace positions via cross-correlating the current trace profile with the data.

    Args:
        data (SpecDataImage): The corresponding data object.
        trace_image (np.ndarray): The masked data image.
        y_positions (np.ndarray): The trace positions for each column.
        trace_profile_cspline (CubicSpline): The trace profile defined by a CubicSpline object.
        badpix_mask (np.ndarray): The bad pixel image mask (1=good, 0=bad).
        height (int): The height of the trace.
        config (dict): The reduction settings dictionary.
        trace_pos_polyorder (int, optional): The polynomial to model the trace positions. Defaults to 2.
        M (int, optional): The desired oversample factor. Defaults to 1.

    Returns:
        np.ndarray: The refined trace positions, y(x).
    """

    # The image dimensions
    ny, nx = trace_image.shape
    
    # Smooth the image
    trace_image_smooth = pcmath.median_filter2d(trace_image, width=5)
    
    # Stores the deviation from the true y position
    ypos_deviations = np.full(nx, dtype=np.float64, fill_value=np.nan)
    
    # Helpful arrays
    yarr = np.arange(ny)
    xarr = np.arange(nx)
    
    # CC lags
    lags = np.arange(-height/2, height/2).astype(int)
    
    # Estimate the sky from the current profile
    if config['sky_subtraction']:
        sky = estimate_sky(trace_image_smooth, y_positions, trace_profile_cspline, height, n_sky_rows=config['n_sky_rows'], M=config['oversample'])
        tp = trace_profile_cspline(trace_profile_cspline.x)
        tp -= pcmath.weighted_median(tp, percentile=0.05)
        
        bad = np.where(tp < 0)[0]
        if bad.size > 0:
            tp[bad] = 0
        good = np.where(np.isfinite(tp))[0]
        trace_profile_cspline_nosky = scipy.interpolate.CubicSpline(trace_profile_cspline.x[good], tp[good])
    else:
        sky = np.zeros(nx)
        trace_profile_cspline_nosky = copy.deepcopy(trace_profile_cspline)
        
    # Fractions of each pixel
    pixel_fractions = generate_pixel_fractions(trace_image_smooth, trace_profile_cspline_nosky, y_positions, badpix_mask, height, min_profile_flux=config['min_profile_flux'])
        
    trace_profile_fiducial_grid, trace_profile = trace_profile_cspline_nosky.x, trace_profile_cspline_nosky(trace_profile_cspline_nosky.x)
    
    good_trace = np.where(np.isfinite(trace_profile))[0]
    
    # Estimate the initial spectrum to know which values correspond to large absorption features.
    spectrum_1d_estimate, _ = boxcar_extraction(trace_image_smooth, y_positions, trace_profile_cspline_nosky, pixel_fractions, badpix_mask, height, config, config['detector_props'], data.itime, sky=sky, n_sky_rows=config['n_sky_rows'])
    
    # Smooth this initial spectrum
    spectrum_1d_estimate_smooth = pcmath.median_filter1d(spectrum_1d_estimate, width=5)
    
    # Normalize
    spectrum_1d_estimate_smooth /= pcmath.weighted_median(spectrum_1d_estimate_smooth, percentile=0.98)

    # Cross correlate each data column with the trace profile estimate
    y_positions_xcorr = np.zeros(nx) + np.nan
    
    for x in range(nx):

        # Skip pixels where the intial spectrum has a flux of less than 50% of the max
        if not np.isfinite(spectrum_1d_estimate_smooth[x]) or spectrum_1d_estimate_smooth[x] <= 0.3:
            continue
        
        # If not enough data points, continue
        good_data = np.where(np.isfinite(trace_image_smooth[:, x]) & (badpix_mask[:, x] == 1))[0]
        if good_data.size < 3:
            continue
        
        # Cross correlation
        # Shift the trace profile
        lags = np.arange(y_positions[x]-height / 2, y_positions[x]+height/2, 1).astype(int)
        trace_profile_shifted = scipy.interpolate.CubicSpline(trace_profile_fiducial_grid[good_trace] + y_positions[x], trace_profile[good_trace], extrapolate=False)(yarr)
        data_no_sky = (trace_image_smooth[:, x] - sky[x]) / np.nanmax(trace_image_smooth[good_data, x] - sky[x])
        bad = np.where(data_no_sky < 0)[0]
        if bad.size > 0:
            data_no_sky[bad] = np.nan
        xcorr = pcmath.cross_correlate2(yarr, data_no_sky, trace_profile_fiducial_grid, trace_profile, lags)
        xcorr /= np.nanmax(xcorr)
        xcorr *= np.exp(-1*(np.arange(xcorr.size) - height / 2)**2 / (2 * lags.size**2)*3)
        good = np.where(np.isfinite(xcorr) & (xcorr > 0.5))[0]
        if good.size <= 2:
            continue
            
        # Fit
        xcorr_fit = np.polyfit(lags[good], xcorr[good], 2)
        yfit = -1 * xcorr_fit[1] / (2 * xcorr_fit[0])
        
        if np.abs(yfit - y_positions[x]) > height / 4:
                continue
            
        y_positions_xcorr[x] = yfit
    
    # Smooth the deviations
    good = np.where(np.isfinite(y_positions_xcorr))[0]
    if good.size < 30:
        return y_positions
    pfit = np.polyfit(xarr[good], y_positions_xcorr[good], trace_pos_polyorder)
    y_positions_refined = np.polyval(pfit, xarr)
    
    return y_positions_refined

def estimate_trace_max(trace_profile_cspline):
    """Estimates the location of the max of the trace profile to a precision of 1000. Crude.

    Args:
        trace_profile_cspline (CubicSpline): The trace profile defined by a CubicSpline object.
        height (int): The height of the trace.
    """
    
    prec = 1000
    trace_profile_fiducial_grid, trace_profile = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
    xhr = np.arange(trace_profile_fiducial_grid[0], trace_profile_fiducial_grid[-1], 1 / prec)
    tphr = trace_profile_cspline(xhr)
    mid = np.nanmedian(trace_profile_fiducial_grid)
    consider = np.where((xhr > mid - 10) & (xhr < mid + 10))[0]
    trace_max_pos = xhr[consider[np.nanargmax(tphr[consider])]]
    
    return trace_max_pos, np.nanmax(tphr)

def crop_image(data_image, config, cval=np.nan):
    """ Masks the image according to left right, top, and bottom values.

    Args:
        data_image (np.ndarray): [description]
        config (dict): The reduction settings dictionary.
        cval (float, optional): The value to mask with. Defaults to np.nan.

    Returns:
        np.ndarray: The masked image.
    """
    
    # The data shape
    ny, nx = data_image.shape
    
    data_image[0:config['mask_image_bottom'], :] = cval
    data_image[ny-config['mask_image_top']:, :] = cval
    data_image[:, 0:config['mask_image_left']] = cval
    data_image[:, nx-config['mask_image_right']:] = cval
    return data_image
            

def estimate_trace_profile(trace_image, y_positions, height, M=16, mask_edges=None):
    """ Estimates the trace profile

    Args:
        trace_image (np.ndarray): The masked data image.
        y_positions (np.ndarray): The trace positions for each column.
        height (int): The height of the trace.
        M (int, optional): The desired oversample factor. Defaults to 16.
        mask_edges (list): [mask_left, mask_right]; Masks additional pixels in the trace profile. Defaults to [5, 5].
    Returns:
        CubicSpline: The trace profile defined by a CubicSpline object.
    """
    
    # The image dimensions
    ny, nx = trace_image.shape
    
    if mask_edges is None:
        mask_edges = (5, 5)
        
    # Smooth the image
    trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)
    
    # Rectify and normalize
    trace_image_smooth_rectified = rectify_trace(trace_image_smooth, y_positions, height, M=M)
    nyr, nxr = trace_image_smooth_rectified.shape
    max_vals = np.nanmax(trace_image_smooth_rectified, axis=0)
    trace_image_smooth_rectified /= np.outer(np.ones(nyr), max_vals)
    good = np.where(max_vals > 0.3*np.nanmedian(max_vals))[0]
    
    # Median Crunch and mask
    trace_profile = np.zeros(nyr) + np.nan
    for yr in range(nyr):
        trace_profile[yr] = np.nanmedian(trace_image_smooth_rectified[yr, good])

    good = np.where(np.isfinite(trace_profile))[0]
    trace_profile[good[0:mask_edges[0]*M]] = np.nan
    trace_profile[good[nyr - mask_edges[1]*M:]] = np.nan
    
    # Construct on hr grid
    yhrt = np.arange(-nyr/2, nyr/2 + 1, 1/M)
    good = np.where(np.isfinite(trace_profile))[0]
    cspline = scipy.interpolate.CubicSpline(yhrt[good], trace_profile[good], extrapolate=False, bc_type='not-a-knot')

    # Offset to max=1 and centered at zero.
    max_pos, max_val = estimate_trace_max(cspline)

    cspline = scipy.interpolate.CubicSpline(cspline.x - max_pos, cspline(cspline.x) / max_val, extrapolate=False, bc_type='natural')

    return cspline



