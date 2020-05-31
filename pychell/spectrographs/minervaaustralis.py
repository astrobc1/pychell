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
    
    # Gain of primary detector
    'gain': None,
    
    # Dark current of primary detector
    'dark_current': None,
    
    # Read noise of the primary detector
    'read_noise': None,
    
    'orientation': None,
    
    'wave_direction': None,
}

# Header keys for reduction and forward modeling
# The keys are common to all instruments
# The items are lists.
# item[0] = actual key in the header
# item[1] = default values
header_keys = {
    'target': ['TCS_OBJ', 'STAR'],
    'RA': ['TCS_RA', '00:00:00.0'],
    'DEC': ['TCS_DEC', '00:00:00.0'],
    'slit': ['SLIT', '0.0'],
    'wavelength_range': ['XDTILT', 'NA'],
    'gas_cell': ['GASCELL', 'NA'],
    'exp_time': ['ITIME', 'NA'],
    'time_of_obs': ['TCS_UTC', 'NA'],
    'NDR': ['NDR', 1],
    'BZERO': ['BZERO', 0],
    'BSCALE': ['BSCALE', 1]
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
    'dark_subtraction': False,
    'flat_division': True,
    'flat_correlation': 'closest_space',
    'bias_subtraction': False,
    'wavelength_calibration': False
}

# Extraction settings
extraction_settings = {
    
    # Order map algorithm (options: 'from_flats, 'empirical')
    'order_map': 'from_flats',
    
    # Pixels to mask on the top, bottom, left, and right edges
    'mask_left_edge': 200,
    'mask_right_edge': 200,
    'mask_top_edge': 30,
    'mask_bottom_edge': 10,
    
    # The height of an order is defined as where the flat is located.
    # This masks additional pixels on each side of the initial trace profile before moving forward.
    # The profile is further flagged after thes sky background is estimated.
    'mask_trace_edges':  0,
    
    # The degree of the polynomial to fit the individual order locations
    'ndegree_poly_fit_trace' : 2,
    
    # Whether or not to perform a sky subtraction
    'sky_subtraction': True,
    'n_sky_rows': 8,
    
    # The window to search for the trace profile (+/- this value )
    'trace_pos_window': 5
}


####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# Default forward model settings
# Number of orders
forward_model_settings = {
    
    # The spectrograph name. Can be anything.
    'spectrograph': 'MinervaAustralis',
    
    # The name of the observatory. Must be a recognized astropy EarthLocation.
    'observatory': 'Mt. Kent',
    
    # The cropped pixels
    'crop_pix': [200, 200],
    
    # The units for plotting
    'plot_wave_unit': 'nm'
}

# Forward model blueprints for RVs
forward_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'StarModelOrderDependent',
        'input_dir': None,
        'vel': [-np.inf, 0, np.inf]
    },
    
    # The Iodine gas cell
    #'gas_cell': {
    #    'name': 'methane_gas_cell', # NOTE: full parameter names are name + base_name.
    #    'class_name': 'GasCellModel',
    #    'input_file': default_templates_path + 'methane_gas_cell_ishell_kgas.npz',
    #    'depth': [1, 1, 1],
    #    'shift': [0, 0, 0]
    #},
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'vis_tellurics', # NOTE: full parameter names are name + component + base_name.
        'class_name': 'TelluricModelTAPAS',
        'vel': [-250, -100, 250],
        'components': {
            'water': {
                'input_file': default_templates_path + 'telluric_water_tapas_ctio.npz',
                'depth': [0.01, 1.5, 4.0]
            },
            'methane': {
                'input_file': default_templates_path + 'telluric_methane_tapas_ctio.npz',
                'depth': [0.1, 1.0, 3.0]
            },
            'nitrous_oxide': {
                'input_file': default_templates_path + 'telluric_nitrous_oxide_tapas_ctio.npz',
                'depth': [0.05, 0.65, 3.0]
            },
            'carbon_dioxide': {
                'input_file': default_templates_path + 'telluric_carbon_dioxide_tapas_ctio.npz',
                'depth': [0.05, 0.65, 3.0]
            },
            'oxygen': {
                'input_file': default_templates_path + 'telluric_oxygen_tapas_ctio.npz',
                'depth': [0.1, 1.1, 3.0]
            },
            'ozone': {
                'input_file': default_templates_path + 'telluric_ozone_tapas_ctio.npz',
                'depth': [0.05, 0.65, 3.0]
            }
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class_name': 'ResidualBlazeModel',
        'n_splines': 14,
        'base_quad': [-5.5E-5, -2E-6, 5.5E-5],
        'base_lin': [-0.001, 1E-5, 0.001],
        'base_zero': [0.96, 1.0, 1.08],
        'spline': [-0.135, 0.01, 0.135],
        'n_delay_splines': 0,
        
        # Blaze is centered on the blaze wavelength. Crude estimates
        'blaze_wavelengths': [4858.091694040058, 4896.964858182707, 4936.465079384465, 4976.607650024426, 5017.40836614558, 5058.88354743527, 5101.050061797753, 5143.9253397166585, 5187.527408353689, 5231.87491060088, 5276.98712989741, 5322.884028578407, 5369.586262921349, 5417.11522691744, 5465.493074938935, 5514.742760771861, 5564.888075329751, 5615.953682999512, 5667.96515950171, 5720.949036590132, 5774.932851929652, 5829.94518764045, 5886.015725989253, 5943.1753026380065, 6001.455961651197, 6060.891016560821, 6121.515108109428, 6183.364282120176, 6246.47605505618]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 0,
        'compress': 64,
        'n_delay': 0,
        'width': [0.055, 0.12, 0.2], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        'name': 'wavesol_ThAr_I2',
        'class_name': 'WaveModelKnown',
        'n_splines': 0, # Zero until I2 cell is implemented
        'n_delay_splines': 0,
        'spline': [-0.03, 0.0005, 0.03]
    }
}


# Return the file number and date
# icm.2019B047.191109.data.00071.a.fits
#def parse_filename(filename):
#    res = filename.split('.')
#    filename_info = {'number': res[4], 'date': res[1][0:4] + res[2][2:]}
#    return filename_info