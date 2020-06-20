import numpy as np
import os
import pychell.rvs # to grab templates dir

# Supported Instruments
supported_instruments = {
    'reduce': ['iSHELL', 'Generic', 'NIRSPEC (dev)'],
    'rvs': ['iSHELL', 'CHIRON', 'MinervaAustralis (dev)', 'PARVI (dev)']
}

# Return the file number and date
# DEFAULT FILE FORMAT for generic instrument: TAG1.DATE.TAG2.NUM.EXT
# Registered in general_settings
def parse_filename(filename):
    res = filename.split('.')
    filename_info = {'number': res[3], 'date': res[1]}
    return filename_info

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
    
    # Where templates are stored for RVs
    'default_templates_path': pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep,
    
    'filename_parser': parse_filename
}

####################################################################
####### Reduction / Extraction #####################################
####################################################################

# Default header keys for reduction
# NOTE: For now, this is only used reduction.
# The keys are common to all instruments
# The items are lists.
# item[0] = actual key in the header
# item[1] = default values
header_keys = {
    'target': ['TCS_OBJ', 'NA'],
    'RA': ['TCS_RA', '00:00:00'],
    'DEC': ['TCS_DEC', '00:00:00'],
    'slit': ['SLIT', 'NA'],
    'wavelength_range': ['WAVELENGTH', 'NA'],
    'gas_cell': ['GASCELL', 'NA'],
    'exp_time': ['ITIME', 0],
    'time_of_obs': ['TIME', 2457000],
    'NDR': ['NDR', 1],
    'BZERO': ['BZERO', 0],
    'BSCALE': ['BSCALE', 1]
}

# calibration settings
# flat_correlation options are
# 'closest_time' for flats to be applied from the closest in time, 'single' (single set)
# 'closest_space' for flats to be applied from the closest in space angular sepration,
# 'single' for a single set.
calibration_settings = {
    'dark_subtraction': False,
    'flat_division': False,
    'bias_subtraction': False,
    'wavelength_calibration': False
}

# Extraction settings
extraction_settings = {
    
    # Order map algorithm (options: 'from_flats, 'empirical')
    'order_map': 'empirical',
    
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
    
    # The optimal extraction algorithm
    'optxalg': 'optimal_extraction_pmassey'
}

####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# The default forward model settings
forward_model_settings = {
    
    # For dev purposes right now ...
    'wave_logspace': False,
    'flux_logspace': False,
    
    # The number of pixels to crop on each side of the spectrum
    'crop_data_pix': [10, 10],
    
    # If the user only wishes to compute the BJDS and barycorrs for later.
    'compute_bc_only': False,
    
    # Barycenter file
    'bary_corr_file': None,
    
    # Stellar template augmentation
    'target_function': 'simple_rms',
    
    'flag_n_worst_pixels': 20,
    
    # Stellar template augmentation
    'template_augmenter': 'cubic_spline_lsq',
    
    'nights_for_template': [],
    
    # Number of iterations to update the stellar template
    'n_template_fits': 10, # a zeroth iteration (flat template) does not count towards this number.
    
    # Cross correlation / bisector span stuff for each iteration. Will take longer.
    # A cross correlation will still be run to estimate the correct overall RV before fitting
    # if starting from a synthetic template
    'do_xcorr': False,
    'xcorr_range': 10*1000,
    'xcorr_step': 50,
    'n_bs' : 1000,
    
    # Model Resolution (n_model_pixels = model_resolution * n_data_pixels)
    # This is only important because of the instrument line profile (LSF)
    # 8 seems sufficient.
    'model_resolution': 8,
}
