# Default Python modules
import os

# Graphics
import matplotlib.pyplot as plt

# Science/Math
import numpy as np
import astropy.units as units
import scipy.interpolate
from astropy.io import fits

# Clustering algorithms (DBSCAN)
import sklearn.cluster

# LLVM
from numba import jit, njit, prange

# Pychell modules
import pychell.maths as pcmath
import pychell.data as pcdata
from robustneldermead.neldermead import NelderMead

# Generate a master flat from a flat cube
def generate_master_flat(individuals, bias_subtraction=False, dark_subtraction=False, norm=0.75):
    """Computes a median master flat field image from a subset of flat field images. Dark and bias subtraction is also performed if set.

        Args:
            individuals (list): The list of FlatFieldImages.
            bias_subtraction (bool): Whether or not to perform bias subtraction. If so, each flat must have a master_bias attribute.
            dark_subtraction (bool): Whether or not to perform dark subtraction. If so, each flat must have a master_dark attribute.
            norm (float): The percentile to normalize the master flat field image to. Defaults to 0.75 (75 percent).
        Returns:
            master_flat (np.ndarray): The median combined and corrected master flat image
        """
    
    # Generate a data cube
    n_flats = len(individuals)
    flats_cube = pcdata.Image.generate_cube(individuals)
    
    # For each flat, subtract master dark and bias
    # Also normalize each image and remove obvious bad pixels
    for i in range(n_flats):
        if bias_subtraction:
            master_bias = individuals[i].master_bias.parse_image()
            flats_cube[i, :, :] -= master_bias
        if dark_subtraction:
            master_dark = individuals[i].master_dark.parse_image()
            flats_cube[i, :, :] -= master_dark
        flats_cube[i, :, :] /= pcmath.weighted_median(flats_cube[i, :, :], percentile=norm)
        bad = np.where((flats_cube[i, :, :] < 0) | (flats_cube[i, :, :] > norm*100))
        if bad[0].size > 0:
            flats_cube[i, :, :][bad] = np.nan
    
    # Median crunch, flag one more time
    master_flat = np.nanmedian(flats_cube, axis=0)
    bad = np.where((master_flat < 0) | (master_flat > norm*100))
    if bad[0].size > 0:
        master_flat[bad] = np.nan
        
    return master_flat

# Generate a master dark image
def generate_master_dark(individuals, bias_subtraction=False):
    """Computes a median master flat field image from a subset of flat field images. Dark and bias subtraction is also performed if set.

        Args:
            individuals (list): The list of DarkImages.
            bias_subtraction (bool): Whether or not to perform bias subtraction. If so, each dark must have a master_bias attribute.
        Returns:
            master_dark (np.ndarray): The median combined master dark image
        """
    darks_cube = pcdata.SpecDataImage.generate_cube(individuals)
    if bias_subtraction:
        for i in darks_cube.shape[0]:
            master_bias = individuals[i].master_bias.parse_image()
            darks_cube[i, :, :] -= master_bias
    master_dark = np.nanmedian(darks_cube, axis=0)
    return master_dark

# Generate a master bias image
def generate_master_bias(individuals):
    """Generates a median master bias image.

    Args:
        individuals (list): The list of BiasImages.
    Returns:
        master_bias (np.ndarray): The master bias image.
    """
    bias_cube = pcdata.SpecDataImage.generate_cube(individuals)
    master_bias = np.nanmedian(bias_cube, axis=0)
    return master_bias

# Bias, Dark, Flat calibration
def standard_calibration(data, data_image, bias_subtraction=False, dark_subtraction=False, flat_division=False):
    """Performs standard bias, flat, and dark corrections to science frames.

    Args:
        data (SpecImage): The SpecImage to calibrate
        data_image (np.ndarray): The corresponding parsed image to calibrate.
        bias_subtraction (bool):  Whether or not to perform bias subtraction, defaults to False.
        dark_subtraction (bool): Whether or not to perform dark subtraction, defaults to False.
        flat_division (bool): Whether or not to perform flat division, defaults to False.
    Returns:
        data_image (np.ndarray): The corrected data image.
    """
    
    # Bias correction
    if bias_subtraction:
        master_bias_image = data.master_bias.parse_image()
        data_image -= master_bias_image
        
    # Dark correction
    if dark_subtraction:
        master_dark_image = data.master_dark.parse_image()
        data_image -= master_dark_image
        
    # Flat division
    if flat_division:
        master_flat_image = data.master_flat.parse_image()
        data_image /= master_flat_image

    return data_image

# Corrects fringing and / or the blaze transmission.
def correct_flat_artifacts(flat, redux_settings):
    """ Corrects artifacts present in flat fields, under dev.

    Args:
        flat (FlatFieldImage): The FlatFieldImage to correct
        redux_settings (dict): redux_settings dictionary.
        
    Returns:
        np.ndarray: The corrected flat field
    """
    flat_image = flat.parse_image()

    order_dicts, order_map_image = order_map.order_dicts, order_map.parse_image()
    n_orders = len(order_dicts)
    
    ny, nx = flat_image.shape

    # Create a straightened version of the flat in this order
    xarr = np.arange(nx).astype(int)
    yarr = np.arange(ny).astype(int)
    xrep = np.outer(np.ones(ny), xarr)
    yrep = np.outer(np.arange(ny), yarr)

    # Create output arrays
    # Ideally,
    # ORIGINAL FLAT = pixel_response_flat * fringing_flat * detector_patterns_no_fringing_flat * spectral_profile_flat
    final_flat = np.full(shape=(ny, nx), fill_value=np.nan)
    pixel_response_flat = np.full(shape=(ny, nx), fill_value=np.nan)
    fringing_flat = np.full(shape=(ny, nx), fill_value=np.nan)
    detector_patterns_no_fringing_flat = np.full(shape=(ny, nx), fill_value=np.nan)
    spectral_profile_flat = np.full(shape=(ny, nx), fill_value=np.nan)

    # Find out the number of orders for that particular object
    detector_patterns_1d = np.full(shape=(nx, n_orders), fill_value=np.nan)
    spectral_profiles = np.full(shape=(nx, n_orders), fill_value=np.nan)

    for o in range(n_orders):

        # Height of order
        height = int(np.ceil(order_dicts[o]['height']))
        
        # Create an image of this order only by masking everything outside of it
        order_image = np.copy(flat_image)

        # Mask everything outside the order
        ypositions = np.polyval(order_dicts[o]['pcoeffs'], xarr)
        bad = np.where(order_map_image != order_dicts[o]['label'])
        if bad[0].size > 0:
            order_image[bad] = np.nan
        
        straight_flat_order = rectify_trace(order_image, ypositions)
        straight_flat_order_smooth = pcmath.median_filter2d(straight_flat_order, 3)
        straight_flat_order_smooth_norm = np.copy(straight_flat_order_smooth)
        for x in range(nx):
            good = np.where(np.isfinite(straight_flat_order_smooth_norm[:, x]))[0]
            if good.size == 0:
                continue
            straight_flat_order_smooth_norm[:, x] = straight_flat_order_smooth_norm[:, x] / np.nanmax(straight_flat_order_smooth_norm[:, x])
        
        # Further mask regions with no data.
        flat_border_cutoff = 0.65
        bad = np.where(straight_flat_order_smooth_norm < flat_border_cutoff)
        if bad[0].size != 0:
            straight_flat_order[bad] = np.nan

        # Create a horizontally smoothed version of the flat to bring out the lamp spectrum.
        # This will "remove" features smaller than length_scale, leaving only the blaze (ideally!)
        length_scale = 30
        straight_flat_hsmooth = pcmath.horizontal_median(straight_flat_order, length_scale)

        # Create a spectral and spatial profile
        spatial_profile = np.nanmedian(straight_flat_hsmooth, axis=1)
        spectral_profile = np.nanmedian(straight_flat_hsmooth, axis=0)
        
        # Extend profile across rows
        spectral_profile_2d = np.outer(np.ones(ny, dtype=float), spectral_profile)
        spectral_profiles[:, o] = spectral_profile
        
        # Dividing the original flat by the spectral profile leaves only
        # the intrinsic detector patterns + 
        detector_patterns_with_pix_response_2d = straight_flat_order / spectral_profile_2d
        
        # Remove the pix to pix response by taking a median across detector columns
        # This leaves behind only the true vertical detector patterns which are row independent.
        # Ideally, we should be using the entire image, but for now this will do.
        detector_patterns = np.nanmedian(detector_patterns_with_pix_response_2d, axis=0)
        
        # Further smooth these detector patterns
        detector_patterns_smooth = pcmath.median_filter1d(detector_patterns, width=3, preserve_nans=True)

        # Store in the 1D correction
        detector_patterns_1d[:, o] = detector_patterns_smooth

    # Model the fringing for each order.

    # Initiate arrays that will contain the best-fitting parameters and the models
    fringing_best_pars = np.full(shape=(n_orders, 4), fill_value=np.nan)
    fringing_models = np.full(shape=(nx, n_orders), fill_value=np.nan)

    # Pixel column indices on which to perform the fit
    min_fit_index = 400
    max_fit_index = 1500

    # Defaut parameter estimates
    # Amplitude, Period, Phase, Period slope (pixels per period)
    period_estimates_ishell = np.array([37.50174251, 37.3355705 , 37.1693985 , 37.0032265 , 36.83705449, 36.67088249, 36.50471049, 36.33853848, 36.17236648, 36.00619448, 35.84002248, 35.67385047, 35.50767847, 35.34150647, 35.17533446, 35.00916246, 34.84299046, 34.67681846, 34.51064645, 34.34447445, 34.17830245, 34.01213044, 33.84595844, 33.67978644, 33.51361443,33.34744243, 33.18127043, 33.01509843, 32.84892642])
    
    # For consistency / testing
    np.random.seed(1)
    
    # Fit fringing in each order
    for o in range(n_orders):
    
        # Data to be fitted with the fringing model
        print('  Modeling fringing for order ' + str(o+1) + '... ', flush=True)
    
        fit_x = np.arange(min_fit_index, max_fit_index+1)
        fit_y = np.copy(detector_patterns_1d[min_fit_index:max_fit_index+1, o])

        # Parameters
        # Amplitude, Phase, Period, Period Slope (y-intercept is pixel=0)
        amp_guess = pcmath.weighted_median(np.abs(fit_y - 1), percentile=0.9)

        period_guess = period_estimates_ishell[o]
        init_pars = np.array([amp_guess, 2 * np.pi, period_guess, -0.0015])
    
        lower_bounds = np.array([np.max([amp_guess-0.005, 1E-5]), 1E-8, period_guess-1, -0.0018])
        upper_bounds = np.array([np.min([amp_guess+0.005, 0.07]), 4*np.pi, period_guess+1, -0.0005])
        solver = NelderMead(fringing_1d_solver, init_pars=init_pars, minvs=lower_bounds, maxvs=upper_bounds, ftol=1E-5, n_iterations=10, no_improve_break=5, args_to_pass=(fit_x, fit_y))
        opt_result = solver.solve()
        
        fringing_best_pars[o, :] = np.copy(opt_result[0])
        fringing_models[:, o] = fringing_1d_compute(xarr, fringing_best_pars[o, :])
        
    # Bring out detector patterns by dividing observed fringing by fringing model
    # and then taking a vertical median to average out any fringing model residual
    # We are seeking column-dependent detector patterns so those will survive a vertical median
    # (observed fringing contains not only true fringing but also detector patterns)
    detector_patterns_no_fringing_ord_dependent_1d = detector_patterns_1d / fringing_models
    detector_patterns_no_fringing = np.nanmedian(detector_patterns_no_fringing_ord_dependent_1d, axis=1)
    
    # Plot the 1d detector pattern
    out_file_plot = output_dir + 'calib' + os.sep + flat.base_input_file[0:-5] + '_detector_patterns.png'
    grid = np.arange(1, 2049, 64)
    plt.figure(1, figsize=(8, 5))
    plt.plot(np.arange(nx)+1, detector_patterns_no_fringing)
    for j in range(grid.size):
        plt.axvline(x=grid[j], linestyle=':', color='darkred', alpha=0.7)
    plt.title('Detector Patterns ' + flat.base_input_file)
    plt.xlim(16, 2032)
    plt.ylim(0.9, 1.1)
    plt.xlabel('X Pixels')
    plt.ylabel('Fractional Deviation from Median')
    plt.savefig(out_file_plot, dpi=150)
    plt.close()
    
    # Save the detector patterns to a text file as well
    out_file_text = output_dir + 'calib' + os.sep + flat.base_input_file[0:-5] + '_detector_patterns.txt'
    np.savetxt(out_file_text, detector_patterns_no_fringing)
    
    # Create a 2D version of the detector patterns, order independent
    detector_patterns_no_fringing_1d = np.outer(detector_patterns_no_fringing, np.ones(n_orders))

    # Remove the detector patterns from each 1d order
    # Smooth the fringing solution w/o detector patterns to eliminate any pixel-to-pixel
    # sensitivity variations which should remain in the flat field
    fringing_1d = detector_patterns_1d / detector_patterns_no_fringing_1d
    for o in range(n_orders):
        fringing_1d[:, o] = pcmath.median_filter1d(fringing_1d[:, o], 5)
        
    # For each order, recreate 2D images
    for o in range(n_orders):
        fringing_2d = np.outer(np.ones(ny), fringing_1d[:, o])
        detector_patterns_2d = np.outer(np.ones(ny), detector_patterns_no_fringing_1d[:, o])
        spectral_profile_2d = np.outer(np.ones(ny), spectral_profiles[:, o])
        g_within_order = np.where(order_map_image == order_dicts[o]['label'])
        ng_within_order = g_within_order[0].size
        fringing_flat[g_within_order] = fringing_2d[g_within_order]
        detector_patterns_no_fringing_flat[g_within_order] = detector_patterns_2d[g_within_order]
        spectral_profile_flat[g_within_order] = spectral_profile_2d[g_within_order]
    
    # Recreate final corrected flat and other relevant quantities
    pixel_response_flat = flat_image / (spectral_profile_flat * fringing_flat * detector_patterns_no_fringing_flat)
    
    if calibration_settings['correct_fringing_in_flatfield'] and calibration_settings['correct_blaze_function_in_flatfield']:
        corrected_flat = flat_image / (fringing_flat * spectral_profile_flat)
    elif calibration_settings['correct_fringing_in_flatfield'] and not calibration_settings['correct_blaze_function_in_flatfield']:
        corrected_flat = flat_image / fringing_flat
    elif not calibration_settings['correct_fringing_in_flatfield'] and calibration_settings['correct_blaze_function_in_flatfield']:
        corrected_flat = flat_image / spectral_profile_flat
    else:
        corrected_flat = np.copy(flat_image)

    # Output flats
    out_file_pix_response = output_dir + 'calib' + os.sep + flat.base_input_file[0:-5] + '_pixel_response_flat.fits'
    
    out_file_fringing = output_dir + 'calib' + os.sep + flat.base_input_file[0:-5] + '_fringing_flat.fits'
    
    out_file_detector_patterns_no_fringing = output_dir + 'calib' + os.sep + flat.base_input_file[0:-5] + '_detector_patterns_flat.fits'
    
    out_file_spectral_profile = output_dir + 'calib' + os.sep + flat.base_input_file[0:-5] + '_spectral_profile_flat.fits'
    
    fits.writeto(out_file_pix_response, pixel_response_flat, overwrite=True)
    fits.writeto(out_file_fringing, fringing_flat, overwrite=True)
    fits.writeto(out_file_detector_patterns_no_fringing, detector_patterns_no_fringing_flat, overwrite=True)
    fits.writeto(out_file_spectral_profile, spectral_profile_flat, overwrite=True)
        
    # Full order plot
    out_file_fringing_model_plot = output_dir + 'calib' + os.sep + flat.base_input_file + '_fringing_models.png'
    
    plot_full_fringing_models(fringing_models, fringing_1d, out_file_fringing_model_plot)
    
    # Text file of parameters
    out_file_fringing_model_text = output_dir + 'calib' + os.sep + flat.base_input_file + '_fringing_models.txt'
    
    np.savetxt(out_file_fringing_model_text, fringing_best_pars, delimiter=',')
    
    return corrected_flat
