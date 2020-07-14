import numpy as np
import os
import pychell.rvs # to grab templates dir

##################################
######## GENERAL SETTINGS ########
##################################

# A few global configurations to get things going
general_settings = {
    
    # Number of cores to use (for Nelder-Mead fitting and cross corr analysis)
    'n_cores': 1,
    
    # Both pipelines utilize a verbose_print and plot keyword
    'verbose_plot': True,
    'verbose_print': False,
    
    # Plotting parameters
    'dpi': 200, # the dpi used in plots
    'plot_wave_unit': 'nm', # The units for plots. Options are nm, ang, microns
}

####################################################################
####### Reduction / Extraction #####################################
####################################################################

redux_settings = {
    
    # Calibration
    'dark_subtraction': False,
    'flat_division': False,
    'flatfield_percentile': 0.75,
    'bias_subtraction': False,
    'wavelength_calibration': False, # Via ThAr, not implemented
    'telluric_correction': False, # Via flat star, not implemented
    
    # Pixels to mask on the top, bottom, left, and right edges
    'mask_left_edge': 20,
    'mask_right_edge': 20,
    'mask_top_edge': 20,
    'mask_bottom_edge': 20,
    
    # The height of an order is defined as where the flat is located.
    # This masks additional pixels on each side of the initial trace profile before moving forward.
    # The profile is further flagged after thes sky background is estimated.
    'mask_trace_edges':  3,
    
    # The degree of the polynomial to fit the individual order locations
    'trace_pos_polyorder' : 2,
    
    # Whether or not to perform a sky subtraction
    # The number of rows used to estimate the sky background (lowest n_sky_rows in the trace profile are used).
    'sky_subtraction': True,
    'n_sky_rows': 8,
    
    # The trace profile is constructed using oversampled data.
    # This is the oversample factor. Use to not oversample.
    'oversample': 4,
    
    # The minimum percentile in the profile to consider.
    'min_profile_flux': 0.05,
    
    # The optimal extraction algorithm
    'optx_alg': 'pmassey_wrapper',
    'pmassey_settings': {'n_iters': 3,'bad_thresh': [100, 50, 25]}
    
}



####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# The default forward model settings
forward_model_settings = {
    
    # The number of pixels to crop on each side of the spectrum
    'crop_data_pix': [10, 10],
    
    # If the user only wishes to compute the BJDS and barycorrs for later.
    'compute_bc_only': False,
    
    # Path of the default provided templates (tellurics, gas cell)
    'default_templates_path': pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep,
    
    # Barycenter file
    'bary_corr_file': None,
    
    # Stellar template augmentation
    'target_function': 'simple_rms',
    
    'flag_n_worst_pixels': 20,
    
    # Stellar template augmentation
    'template_augmenter': 'cubic_spline_lsq',
    'nights_for_template': [],
    'templates_to_optimize': [],
    
    # Number of iterations to update the stellar template
    'n_template_fits': 10, # a zeroth iteration (flat template) does not count towards this number.
    
    # Cross correlation / bisector span stuff for each iteration. Will take longer.
    # A cross correlation will still be run to estimate the correct overall RV before fitting
    # if starting from a synthetic template
    'do_xcorr': False,
    'xcorr_range': 10*1000, # m/s
    'xcorr_step': 50, # m/s
    'n_bs' : 1000, # 0, ..., 1 (approx)
    
    'remove_continuum': False,
        
    # Model Resolution (n_model_pixels = model_resolution * n_data_pixels)
    # This is only important because of the instrument line profile (LSF)
    # 8 seems sufficient.
    'model_resolution': 8
}
