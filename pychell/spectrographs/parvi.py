import os
import numpy as np
import pychell.rvs

# Path to default templates for rvs
default_templates_path = pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep


##################################################################
####### General Stuff ############################################
##################################################################

# Some general parameters
general_settings = {
    
    # The spectrograph name. Can be anything.
    'spectrograph': 'PARVI',
    
    # The name of the observatory.
    # Must be a recognized astropy EarthLocation if not computing own barycenter info.
    'observatory': 'Palomar',
    
    # Gain of primary detector
    'gain': NotImplemented,
    
    # Dark current of primary detector
    'dark_current': NotImplemented,
    
    # Read noise of the primary detector
    'read_noise': NotImplemented,
    
     # The orientation of the spectral axis for 2d images
    'orientation': NotImplemented,
    
    # The number of data pixels for forward modeling (includes cropped pix on the ends)
    'n_data_pix': 2038,
    
    # increasing => left to right, decreasing => right to left
    'wave_direction': 'increasing',
    
    # The time offset used in the headers
    'time_offset': NotImplemented,
    
    # The tags to recognize science, bias, dark, and flat field images
    'sci_tag': NotImplemented,
    'bias_tag': NotImplemented,
    'darks_tag': NotImplemented,
    'flats_tag': NotImplemented,
    
    # The file name parser
    'filename_parser': NotImplemented
}


# Header keys for reduction
header_keys = NotImplemented

####################################################################
####### Reduction / Extraction #####################################
####################################################################

# calibration settings
calibration_settings = NotImplemented

# Extraction settings
extraction_settings = NotImplemented

####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# Default forward model settings
# Default forward model settings
forward_model_settings = {
    
    # The cropped pixels
    'crop_data_pix': [10, 10],
    
    # The units for plotting
    'plot_wave_unit': 'nm'
}

# Forward model blueprints for RVs
forward_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'StarModel',
        'input_file': None,
        'vel': [-np.inf, 0, np.inf]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'vis_tellurics', # NOTE: full parameter names are name + species + base_name.
        'class_name': 'TelluricModelTAPAS',
        'vel': [-2000, -100, 500],
        'species': {
            'water': {
                'input_file': default_templates_path + 'telluric_water_tapas_palomar.npz',
                'depth': [0.01, 1.5, 4.0]
            },
            'methane': {
                'input_file': default_templates_path + 'telluric_methane_tapas_palomar.npz',
                'depth': [0.1, 1.0, 3.0]
            },
            'nitrous_oxide': {
                'input_file': default_templates_path + 'telluric_nitrous_oxide_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0]
            },
            'carbon_dioxide': {
                'input_file': default_templates_path + 'telluric_carbon_dioxide_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0]
            },
            'oxygen': {
                'input_file': default_templates_path + 'telluric_oxygen_tapas_palomar.npz',
                'depth': [0.1, 1.1, 3.0]
            },
            'ozone': {
                'input_file': default_templates_path + 'telluric_ozone_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0]
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
        'n_delay_splines': 0,
        
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
        'n_delay': 0,
        'width': [0.15, 0.21, 0.28], # LSF width, in angstroms (slightly larger than this for PARVI)
        'ak': [-0.075, 0.001, 0.075] # See cale et al 2019 or arfken et al some year for definition of ak > 0
    },
    
    # Frequency comb (no splines since no gas cell)
    'wavelength_solution': {
        'name': 'laser_comb_wls',
        'class_name': 'WaveModelKnown',
        'n_splines': 0,
        'n_delay_splines': 0,
        'spline': [-0.15, 0.01, 0.15]
    }
}