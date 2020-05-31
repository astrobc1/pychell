import os
import numpy as np
import pychell.rvs

# Path to default templates for rvs
default_templates_path = pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep

##################################################################
####### General Stuff ############################################
##################################################################

# Instrument parameters
hard_settings = {
    
    # Image number locations (first, last) in the filenames
    'image_nums': None
    
    # Gain of primary detector
    'gain': None,
    
    # Dark current of primary detector
    'dark_current': None,
    
    # Read noise of the primary detector
    'read_noise': None,
    
    # Whether or not the flat-field lamp illuminates the entire image (True) or not (False).
    'full_flat_illumination': None,
}

# Header keys for reduction and forward modeling
header_keys = {
    "object": None,
    "slit": None,
    "wavelength_range": None,
    "gas_cell": None,
    "exptime": None,
    "time_of_obs": None
}



####################################################################
####### Reduction / Extraction #####################################
####################################################################

# calibration settings
 # flat_correlation options are
 # 'closest_time' for flats to be applied from the closest in time, 'single' (single set)
 # 'closest_space' for flats to be applied from the closest in space angular sepration,
 # 'single' for a single set.
calibration_settings = {
    'dark_subtraction': None,
    'flat_division': None,
    'flat_correlation': None,
    'bias_subtraction': None,
    'wavelength_calibration': None
}

# Extraction settings
extraction_settings = {
    
    # Pixels to mask on the top, bottom, left, and right edges
    'mask_left_edge': None,
    'mask_right_edge': None,
    'mask_top_edge': None,
    'mask_bottom_edge': None,
    
    # The height of an order is defined as where the flat is located.
    # This masks additional pixels on each side of the initial trace profile before moving forward.
    # The profile is further flagged after thes sky background is estimated.
    'mask_trace_edges':  None,
    
    # The degree of the polynomial to fit the individual order locations
    'ndegree_poly_fit_trace' : None,
    
    # Whether or not to perform a sky subtraction
    'sky_subtraction': None,
    'n_sky_rows': None,
    
    # The window to search for the trace profile (+/- this value )
    'trace_pos_window': None
}


####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# Default forward model settings
# Number of orders
forward_model_settings = {
    
    # The spectrograph name. Can be anything.
    'spectrograph': 'PARVI',
    
    # The name of the observatory. Must be a recognized astropy EarthLocation.
    'observatory': 'Palomar',
    
    # The number of data pixels
    'n_data_pix' : 2038,
    
    # The number of echelle orders
    'n_orders': 29,
    
    # The cropped pixels
    'crop_pix': [10, 10],
    
    # The units for plotting
    'plot_wave_unit': 'nm'
}

# Construct the default PARVI forward model
# Each entry must have a name and class.
# A given model can be effectively not used if n_delay is greater than n_template_fits
# Mandatory: Wavelength solution and star. Technically the rest are optional.
# Keywords are special, but their classes they point to can be anything.
# Keywords are rarely used explicitly in the code, but they are.
# Keywords:
# 'star' = star
# 'wavelength_solution' = wavelength solution
# 'lsf' = the line spread function
# 'tellurics' = the telluric model
# Remaining model components can have any keys, since the code won't be doing anything special with them.
default_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'StarModel',
        'input_file': None,
        'vel': [-np.inf, 0, np.inf]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'yjhband_tellurics', # NOTE: full parameter names are name + component + base_name.
        'class_name': 'TelluricModelTAPAS',
        'vel': [-4000, -1300, 1000],
        'components': {
            'water': {
                'input_file': default_templates_path + 'telluric_water_tapas_palomar.npz',
                'depth': [0.01, 1.5, 5.0],
            },
            'methane': {
                'input_file': default_templates_path + 'telluric_methane_tapas_palomar.npz',
                'depth': [0.1, 1.0, 3.0],
            },
            'nitrous_oxide': {
                'input_file': default_templates_path + 'telluric_nitrous_oxide_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0],
            },
            'carbon_dioxide': {
                'input_file': default_templates_path + 'telluric_carbon_dioxide_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0],
            },
            'oxygen': {
                'input_file': default_templates_path + 'telluric_oxygen_tapas_palomar.npz',
                'depth': [0.1, 1.1, 3.0],
            },
            'ozone': {
                'input_file': default_templates_path + 'telluric_ozone_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0],
            }
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class_name': 'ResidualBlazeModel',
        'n_splines': 0,
        'base_quad': [-5.5E-5, -2E-6, 5.5E-5],
        'base_lin': [-0.001, 1E-5, 0.001],
        'base_zero': [0.96, 1.0, 1.15],
        'spline': [-0.025, 0.001, 0.025],
        
        # Blaze is centered on the blaze wavelength.
        'blaze_wavelengths': [14807.35118155, 14959.9938945 , 15115.82353631, 15274.96519038,
       15946.37631354, 16123.53607164, 16304.66829244]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 4,
        'compress': 64,
        'width': [0.15, 0.21, 0.28], # LSF width, in angstroms (slightly larger than this for PARVI)
        'ak': [-0.075, 0.001, 0.075] # See cale et al 2019 or arfken et al some year for definition of ak > 0
    },
    
    # Frequency comb (no splines since no gas cell)
    'wavelength_solution': {
        'name': 'laser_comb_wls',
        'class_name': 'WaveModelKnown',
        'n_splines': 0,
        'spline': [-0.15, 0.01, 0.15]
    }
    
}
    
    