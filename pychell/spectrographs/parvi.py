import os
import numpy as np
import pychell.rvs

spectrograph = "PARVI"
observatory = "Palomar"

# These settings are currently tuned for PARVI simulations.

####################################################################
####### Reduction / Extraction #####################################
####################################################################


redux_settings = NotImplemented

####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# Default forward model settings
# Default forward model settings
forward_model_settings = {
    
    # The cropped pixels
    'crop_data_pix': [5, 5],
    
    # The units for plotting
    'plot_wave_unit': 'nm',
    
    'observatory': observatory
}

# Forward model blueprints for RVs
forward_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'StarModel',
        'input_file': None,
        'vel': [-3E5, 100, 3E5]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'nir_tellurics',
        'class_name': 'TelluricModelTAPAS',
        'vel': [0, 0, 0],
        'species': {
            'water': {
                'input_file': 'telluric_water_tapas_palomar.npz',
                'depth': [1, 1, 1]
            },
            'methane': {
                'input_file': 'telluric_methane_tapas_palomar.npz',
                'depth': [1, 1, 1]
            },
            'nitrous_oxide': {
                'input_file': 'telluric_nitrous_oxide_tapas_palomar.npz',
                'depth': [1, 1, 1]
            },
            'carbon_dioxide': {
                'input_file': 'telluric_carbon_dioxide_tapas_palomar.npz',
                'depth': [1, 1, 1]
            },
            'oxygen': {
                'input_file': 'telluric_oxygen_tapas_palomar.npz',
                'depth': [1, 1, 1]
            },
            'ozone': {
                'input_file': 'telluric_ozone_tapas_palomar.npz',
                'depth': [1, 1, 1]
            }
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class_name': 'ResidualBlazeModel',
        'n_splines': 0,
        'base_quad': [0, 0, 0],
        'base_lin': [0, 0, 0],
        'base_zero': [1, 1, 1],
        'spline': [-0.025, 0.001, 0.025],
        'n_delay_splines': 0
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 0,
        'compress': 64,
        'n_delay': 0,
        'width': [0.08, 0.08, 0.08], # LSF width, in angstroms
        'ak': [-0.075, 0.001, 0.075] # See cale et al 2019 or arfken et al some year for definition of ak > 0
    },
    
    # Frequency comb (no splines since no gas cell)
    'wavelength_solution': {
        'name': 'laser_comb_wls',
        'class_name': 'WaveModelHybrid',
        'n_splines': 0,
        'n_delay_splines': 0,
        'spline': [-0.15, 0.01, 0.15]
    }
}