# Default Python modules
import os
import glob
import sys
from pdb import set_trace as stop


# Graphics
import matplotlib.pyplot as plt

# Science/Math
import numpy as np
from astropy.coordinates import Angle, SkyCoord
import astropy.units as units
from astropy.io import fits

# Clustering algorithms (DBSCAN)
import sklearn.cluster

# LLVM
from numba import jit, njit, prange

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
import pychell.reduce.data2d as pcdata

# Generate a master flat from a flat cube
def generate_master_flat(flats_cube, master_dark=None, master_bias=None):
    """Computes a median master flat field image from a subset of flat field images.

        Args:
            flats_cube (np.ndarray): The flats data cube, with shape=(n_flats, ny, nx)
            master_dark (np.ndarray): The master dark image (no dark subtraction if None)
            master_bias (np.ndarray): The master bias image (no bias subtraction if None).
        Returns:
            master_flat (np.ndarray): The median combined and corrected master flat image
        """
    for i in range(flats_cube.shape[0]):
        if master_bias is not None:
            flats_cube[i, :, :] -= master_bias
        if master_dark is not None:
            flats_cube[i, :, :] -= master_dark
        flats_cube[i, :, :] /= pcmath.weighted_median(flats_cube[i, :, :], med_val=0.99)
        bad = np.where((flats_cube[i, :, :] < 0) | (flats_cube[i, :, :] > 100))
        if bad[0].size > 0:
            flats_cube[i, :, :][bad] = np.nan
    master_flat = np.nanmedian(flats_cube, axis=0)
    bad = np.where((master_flat < 0) | (master_flat > 100))
    if bad[0].size > 0:
        master_flat[bad] = np.nan
    return master_flat

# Generate a master dark image
def generate_master_dark(darks_cube):
    """Generates a median master dark image given a darks image cube.

    Args:
        darks_cube (np.ndarray): The data cube of dark images, with shape=(n_bias, ny, nx)
    Returns:
        master_dark (np.ndarray): The master dark image.
    """
    master_dark = np.nanmedian(darks_cube, axis=0)
    return master_dark

# Generate a master bias image
def generate_master_bias(bias_cube):
    """Generates a median master bias image given a bias image cube.

    Args:
        bias_cube (np.ndarray): The data cube of bias images, with shape=(n_bias, ny, nx)
    Returns:
        master_bias (np.ndarray): The master bias image.
    """
    master_bias = np.nanmedian(bias_cube, axis=0)
    return master_bias

# Bias, Dark, Flat calibration
def standard_calibration(data_image, master_bias_image=None, master_flat_image=None, master_dark_image=None):
    """Performs standard bias, flat, and dark corrections.

    Args:
        data_image (np.ndarray): The data to calibrate.
        master_bias_image (np.ndarray): The master bias image (no bias subtraction if None).
        master_dark_image (np.ndarray): The master dark image (no dark subtraction if None).
        master_flat_image (np.ndarray): The master flat image (no flat correction if None).
    Returns:
        data_image (np.ndarray): The corrected data image.
    """
    
    # Bias correction
    if master_bias_image is not None:
        data_image -= master_bias_image
        
    # Dark correction
    if master_dark_image is not None:
        data_image -= master_dark_image
        
    # Flat division
    if master_flat_image is not None:
        data_image /= master_flat_image

    return data_image


# Corrects fringing and / or the blaze transmission.
def correct_flat_artifacts(master_flat):
    
    master_flat_image = master_flat.load()

    xval = np.outer(np.ones(gpars['ny']), np.arange(gpars['nx']))
    yval = np.outer(np.arange(gpars['ny']), np.ones(gpars['nx']))

    # Create output arrays
    # final: master flat to be used with data
    # pixel_response: a flat with only the pixel to pixel response (no detector patterns, no blaze, no fringing)
    # fringing: a flat with only the isolated fringing signal (no detector patterns (possibly), no blaze, no pixel to pixel response )
    # detector_patterns_no_fringing_flat: a flat with only the isolated detector patterns (no fringing).
    # The order dependent spectral profile flat
    final_flat = np.full(shape=(gpars['ny'], gpars['nx']), fill_value=np.nan)
    pixel_response_flat = np.full(shape=(gpars['ny'], gpars['nx']), fill_value=np.nan)
    fringing_flat = np.full(shape=(gpars['ny'], gpars['nx']), fill_value=np.nan)
    detector_patterns_no_fringing_flat = np.full(shape=(gpars['ny'], gpars['nx']), fill_value=np.nan)
    spectral_profile_flat = np.full(shape=(gpars['ny'], gpars['nx']), fill_value=np.nan)

    # Find out the number of orders for that particular object
    detector_patterns_1d = np.full(shape=(n_orders, gpars['nx']), fill_value=np.nan)
    spectral_profiles = np.full(shape=(n_orders, gpars['nx']), fill_value=np.nan)

    for i in range(n_orders):

        print('    Order ' + str(i+1) + ' of ' + str(n_orders))

        # Height of order
        height = int(np.ceil(orders_structure[i]['height']))
        
        # Create an image of this order only by masking everything outside of it
        order_image = np.copy(flat_image)

        # Mask everything outside the order
        order_left_location = np.polyval(orders_structure[i]['left_coeffs'], xval)
        order_right_location = np.polyval(orders_structure[i]['right_coeffs'], xval)
        bad = np.where((yval < order_left_location) | (yval > order_right_location))
        nbad = bad[0].size
        if nbad > 0:
            order_image[bad] = np.nan

        # Create a straightened version of the flat in this order
        straight_flat_order = np.full(shape=(height, gpars['nx']), fill_value=np.nan)
        xarr = np.arange(gpars['nx']).astype(int)
        yarr = np.arange(gpars['ny']).astype(int)

        # Actually do the straightening of the flat
        order_center_location = np.polyval(orders_structure[i]['mid_coeffs'], xarr)
        subyarr = np.arange(int(height))
        for l in range(gpars['nx']):
            # Skip this column if the order is partially out of frame
            if (order_center_location[l] - height / 2) <= 0:
                continue
            straight_flat_order[:, l] = interpolate_fix_nans(yarr, order_image[:, l], (order_center_location[l] - height / 2) + subyarr)

        # Mask the regions outside of the trace
        hmedian_flat = np.nanmedian(straight_flat_order, axis=1)
        
        flat_border_cutoff = 0.65
        bad = np.where(hmedian_flat <= (np.nanmax(hmedian_flat) * flat_border_cutoff))[0]
        nbad = bad.size
        if nbad != 0:
            straight_flat_order[bad, :] = np.nan
        
        # Count how many pixels were removed on each side
        gleft = np.where(bad < int(height / 2))[0]
        ngleft = gleft.size
        gright = np.where(bad > int(height / 2))[0]
        ngright = gright.size

        # Mask these lines in the original flat image
        bad = np.where((yval < (order_left_location + ngleft)) | (yval > (order_right_location - ngright)))
        nbad = bad[0].size
        if nbad != 0:
            order_image[bad] = np.nan

        # Create a horizontally smoothed version of the flat to bring out the lamp spectrum
        straight_flat_hsmooth = horizontal_median(straight_flat_order, gpars['nh_smooth_fringing'])

        # If required mask more pixels in the vertical direction
        spatial_profile = np.nanmedian(straight_flat_hsmooth, axis=1)
        gfin = np.where(np.isfinite(spatial_profile))[0]
        ngfin = gfin.size
        if gpars['npix_cutoff_top'] > 1:
            straight_flat_hsmooth[gfin[0:gpars['npix_cutoff_top']], :] = np.nan
        if gpars['npix_cutoff_top'] == 1:
            straight_flat_hsmooth[gfin[0], :] = np.nan
        if gpars['npix_cutoff_bottom'] > 1:
            straight_flat_hsmooth[gfin[-1*gpars['npix_cutoff_bottom']:], :] = np.nan
        if gpars['npix_cutoff_bottom'] == 1:
            straight_flat_hsmooth[gfin[-1], :] = np.nan

        # Create a spectral profile
        spectral_profile = np.nanmedian(straight_flat_hsmooth, axis=0)
        
        # Apply additional smoothing to remove any detector pattern residuals
        spectral_profile = smooth_error(median_filter1d(spectral_profile, gpars['nh_smooth_fringing']), gpars['nh_smooth_fringing'], gpars)
        
        # Extend profile across rows
        spectral_profile_2d = np.outer(np.ones(height, dtype=float), spectral_profile)
        spectral_profiles[i, :] = spectral_profile
        
        # Dividing the original flat by the spectral profile leaves only
        # the detector patterns, no blaze but still has pix to pix response
        detector_patterns_with_pix_response_2d = straight_flat_order / spectral_profile_2d
        
        # Remove the pix to pix response by taking a median across detector columns
        # This leaves behind only the true vertical detector patterns which are row independent but possibly order dependent
        detector_patterns = np.nanmedian(detector_patterns_with_pix_response_2d, axis=0)
        
        # Create a horizontally smoothed version of the detector patterns (only 3 pixels)
        patterns_smooth = np.copy(detector_patterns)
        # else:
        if gpars['fringe_nsmooth'] > 1:
            patterns_smooth = median_filter1d(detector_patterns, gpars['fringe_nsmooth'])
        else:
            patterns_smooth = np.copy(detector_patterns)

        # Store in the 1D correction
        detector_patterns_1d[i, :] = patterns_smooth

    # Number of "walkers" that will independently run a Levenberg-Marquardt least-squares fitting
    # A large number requires more CPU but will be more robust against local minima
    nfit = 30

    # Initiate arrays that will contain the best-fitting parameters and the models
    model_pars = np.full(shape=(n_orders, 4), fill_value=np.nan)
    models = np.full(shape=(n_orders, gpars['nx']), fill_value=np.nan)

    # Pixel column indices on which to perform the fit
    min_fit_index = 400
    max_fit_index = 1500

    # Defaut parameter estimates
    # Amplitude, Period, Phase, Period slope (pixels per period)
    period_estimates = np.array([37.50174251, 37.3355705 , 37.1693985 , 37.0032265 , 36.83705449, 36.67088249, 36.50471049, 36.33853848, 36.17236648, 36.00619448, 35.84002248, 35.67385047, 35.50767847, 35.34150647, 35.17533446, 35.00916246, 34.84299046, 34.67681846, 34.51064645, 34.34447445, 34.17830245, 34.01213044, 33.84595844, 33.67978644, 33.51361443,33.34744243, 33.18127043, 33.01509843, 32.84892642])
    
    np.random.seed(1)
    
    # Fit fringing in each order
    for i in range(n_orders):
    
        # Data to be fitted with the fringing model
        print('  Modeling fringing ... ')
    
        fit_y = np.copy(detector_patterns_1d[i, min_fit_index:max_fit_index+1])
        fit_x = np.arange(min_fit_index, max_fit_index+1)

        # Parameters
        # Amplitude, Phase, Period, Period Slope (y-intercept is pixel=0)
        amp_guess = weighted_median(np.abs(fit_y - 1), med_val=.9)

        period_guess = period_estimates[i]
        init_pars = np.array([amp_guess, 2 * np.pi, period_guess, -0.0015])
    
        lower_bounds = np.array([np.max([amp_guess-0.005, 1E-5]), 1E-8, period_guess-1, -0.0018])
        upper_bounds = np.array([np.min([amp_guess+0.005, 0.07]), 4*np.pi, period_guess+1, -0.0005])
        bounds = (lower_bounds, upper_bounds)
        
        # Typical variations to be explored by the N walkers in parameter space
        fit_par_scatter = np.array([amp_guess / 100, np.pi/10, 0.02, 0.0015/100])
    
        # Create random initial positions for the walkers
        fit_par_noise = np.outer(np.ones(nfit), init_pars) + np.outer(np.ones(nfit), fit_par_scatter) * np.random.randn(nfit, init_pars.size)
        fit_par_noise[0, :] = np.copy(init_pars)
    
        # Perform least-squares fitting for each walker
        best_pars_order_i = np.full(shape=(nfit, 4), fill_value=np.nan)
        chi2s_order_i = np.full(nfit, fill_value=np.nan)
        good = np.where(np.isfinite(fit_y))[0]
        fit_x_ = np.copy(fit_x[good])
        fit_y_ = np.copy(fit_y[good])
        #for j in range(nfit):
        #    if np.any(fit_par_noise[j, :] < lower_bounds) or np.any(fit_par_noise[j, :] > upper_bounds):
        #        print(fit_par_noise[j, :])
        #        print(lower_bounds)
        #        print(upper_bounds)
        #    best_pars_order_i[j, :] = scipy.optimize.least_squares(fringing_1d_solver, x0=fit_par_noise[j, :], args=(fit_x_, fit_y_), bounds=bounds).x
        #    
        #    model_ij = fringing_1d_compute(fit_x_, best_pars_order_i[j, :])
        #    diffs = fit_y_ - model_ij
        #    chi2s_order_i[j] = np.nansum(diffs**2) / fit_x.size
        
        
        fit_result = nelder_mead.simps(init_pars, fringing_1d_solver, vlb=lower_bounds, vub=upper_bounds, xtol=1E-4, ftol=1E-5, n_sub_calls=10, no_improv_break=5, args_to_pass=(fit_x_, fit_y_))
        model_pars[i, :] = np.copy(fit_result[0])
        models[i, :] = fringing_1d_compute(np.arange(gpars['nx']), model_pars[i, :])
            
        #bpi = np.nanargmin(chi2s_order_i)
        #print(bpi)
        
        
        # Store best fit parameters and model
        #model_pars[i, :] = np.copy(best_pars_order_i[bpi, :])
        #models[i, :] = fringing_1d_compute(np.arange(gpars['nx']), best_pars_order_i[bpi, :])
        
    # Bring out detector patterns by dividing observed fringing by fringing model
    # and then taking a vertical median to average out any fringing model residual
    # We are seeking column-dependent detector patterns so those will survive a vertical median
    # (observed fringing contains not only true fringing but also detector patterns)
    detector_patterns_no_fringing_ord_dependent_1d = detector_patterns_1d / models
    detector_patterns_no_fringing = np.nanmedian(detector_patterns_no_fringing_ord_dependent_1d, axis=0)
    
    # Plot the 1d detector pattern
    out_file_plot = gpars['output_dir_root'] + 'flats' + os.sep + 'detector_patterns_' + obs_details_unique['slit'] + '_' + obs_details_unique['object'] + '_' + obs_details_unique['wavelength_band'] + '.png'
    grid = np.arange(1, 2049, 64)
    plt.figure(1, figsize=(8, 5))
    plt.plot(np.arange(gpars['nx'])+1, detector_patterns_no_fringing)
    for j in range(grid.size):
        plt.axvline(x=grid[j], linestyle=':', color='darkred', alpha=0.7)
    plt.title('Detector Patterns ' + obs_details_unique['slit'] + ' ' + obs_details_unique['object'] + ' ' + obs_details_unique['wavelength_band'])
    plt.xlim(16, 2032)
    plt.ylim(0.9, 1.1)
    plt.xlabel('X Pixels')
    plt.ylabel('Fractional Deviation from Median')
    plt.savefig(out_file_plot, dpi=150)
    plt.close()
    
    # Save the detector patterns to a text file as well
    out_file_text = gpars['output_dir_root'] + 'flats' + os.sep + 'detector_patterns_' + obs_details_unique['slit'] + '_' + obs_details_unique['object'] + '_' + obs_details_unique['wavelength_band'] + '.txt'
    np.savetxt(out_file_text, detector_patterns_no_fringing)
    
    # Create a 2D version of the detector patterns, order independent
    detector_patterns_no_fringing_1d = np.outer(np.ones(n_orders), detector_patterns_no_fringing)

    # Remove the detector patterns from each 1d order
    # Smooth the fringing solution w/o detector patterns to eliminate any pixel-to-pixel
    # sensitivity variations which should remain in the flat field
    # Detector_patterns_all = fringing * other
    fringing_1d = detector_patterns_1d / detector_patterns_no_fringing_1d
    for sri in range(n_orders):
        fringing_1d[sri, :] = median_filter1d(fringing_1d[sri, :], gpars['fringe_nsmooth'])
        
    # For each order, recreate 2D images
    for sri in range(n_orders):
        fringing_2d = np.outer(np.ones(gpars['ny']), fringing_1d[sri, :])
        detector_patterns_2d = np.outer(np.ones(gpars['ny']), detector_patterns_no_fringing_1d[sri, :])
        spectral_profile_2d = np.outer(np.ones(gpars['ny']), spectral_profiles[sri, :])
        g_within_order = np.where(orders_mask == orders_structure[sri]['order_id'] - 1)
        ng_within_order = g_within_order[0].size
        fringing_flat[g_within_order] = fringing_2d[g_within_order]
        detector_patterns_no_fringing_flat[g_within_order] = detector_patterns_2d[g_within_order]
        spectral_profile_flat[g_within_order] = spectral_profile_2d[g_within_order]

    # Recreate 2D image of illumination
    #spectral_profile_image = np.full(shape=(gpars['ny'], gpars['nx']), fill_value=np.nan)
    #for sri in range(n_orders):
        #spectral_profile_2d = np.outer(np.ones(gpars['ny']), spectral_profiles[sri, :])
        #g_within_order = np.where(orders_mask == orders_structure[sri]['order_id']  - 1)
        #ng_within_order = g_within_order[0].size
        #spectral_profile_image[g_within_order] = spectral_profile_2d[g_within_order]
    
    # Recreate final corrected flat and other relevant quantities
    pixel_response_flat = flat_image / (spectral_profile_flat * fringing_flat * detector_patterns_no_fringing_flat)
    
    if gpars['correct_fringing_in_flatfield'] and gpars['correct_blaze_function_in_flatfield']:
        final_flat = flat_image / (fringing_flat * spectral_profile_flat)
    elif gpars['correct_fringing_in_flatfield'] and not gpars['correct_blaze_function_in_flatfield']:
        final_flat = flat_image / fringing_flat
    elif not gpars['correct_fringing_in_flatfield'] and gpars['correct_blaze_function_in_flatfield']:
        final_flat = flat_image / spectral_profile_flat
    else:
        final_flat = np.copy(flat_image)
        
    # Fix any negatives with nans since this basically ruins flat division if the pixel in the data is good
    bad = np.where(final_flat < 0)
    if bad[0].size > 0:
        final_flat[bad] = np.nan
        
    # Remove edge data from the flat
    # Mask the flat edges
    if gpars['nmask_bottom_rows'] > 0:
        final_flat[0:gpars['nmask_bottom_rows'], :] = np.nan
    if gpars['nmask_top_rows'] > 0:
        final_flat[-1*gpars['nmask_top_rows']:, :] = np.nan
    if gpars['nmask_left_cols'] > 0:
        final_flat[:, 0:gpars['nmask_left_cols']] = np.nan
    if gpars['nmask_right_cols'] > 0:
        final_flat[:, -1*gpars['nmask_right_cols']:] = np.nan

    # Output fringing and Pixel to pixel reponse flats
    out_file_pix_response = gpars['output_dir_root'] + 'flats' + os.sep + 'pixel_response_flat_' + obs_details_unique['slit'] + '_' + obs_details_unique['object'] + '_' + obs_details_unique['wavelength_band'] + '.fits'
    out_file_fringing_corr = gpars['output_dir_root'] + 'flats' + os.sep + 'fringing_flat_' + obs_details_unique['slit'] + '_' + obs_details_unique['object'] + '_' + obs_details_unique['wavelength_band'] + '.fits'
    out_file_detector_patterns_no_fringing_corr = gpars['output_dir_root'] + 'flats' + os.sep + 'detector_patterns_no_fringing_flat_' + obs_details_unique['slit'] + '_' + obs_details_unique['object'] + '_' + obs_details_unique['wavelength_band'] + '.fits'
    out_file_spectral_profile_corr = gpars['output_dir_root'] + 'flats' + os.sep + 'spectral_profile_flat_' + obs_details_unique['slit'] + '_' + obs_details_unique['object'] + '_' + obs_details_unique['wavelength_band'] + '.fits'
    fits.writeto(out_file_pix_response, pixel_response_flat, overwrite=True)
    fits.writeto(out_file_fringing_corr, fringing_flat, overwrite=True)
    fits.writeto(out_file_detector_patterns_no_fringing_corr, detector_patterns_no_fringing_flat, overwrite=True)
    fits.writeto(out_file_spectral_profile_corr, spectral_profile_flat, overwrite=True)
    
    #if gpars['correct_fringing_in_flatfield']:
        
    # Full order plot
    out_file_fringing_model_plot = gpars['output_dir_root'] + 'flats' + os.sep + 'fringing_model1d_' + obs_details_unique['slit'] + '_' + obs_details_unique['object'] + '_' + obs_details_unique['wavelength_band'] + '.png'
    plot_full_fringing_models(models.T, fringing_1d.T, out_file_fringing_model_plot, n_orders, obs_details_unique['object'], gpars)
    
    # Text file of parameters
    out_file_fringing_model_text = gpars['output_dir_root'] + 'flats' + os.sep + 'fringing_model1d_pars_' + obs_details_unique['slit'] + '_' + obs_details_unique['object'] + '_' + obs_details_unique['wavelength_band'] + '.txt'
    np.savetxt(out_file_fringing_model_text, model_pars, delimiter=',')
    
    return final_flat