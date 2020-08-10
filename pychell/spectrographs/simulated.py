import os
import numpy as np
import pychell.rvs

# Other notes for iSHELL:
# blaze_model parameters for a sinc model (full unmodified blaze)
# a: [1.02, 1.05, 1.08], b: [0.008, 0.01, 0.0115], c: [-5, 0.1, 5], d: [0.51, 0.7, 0.9]

#############################
####### Name and Site #######
#############################

spectrograph = 'Simulated'
observatory = 'IRTF'

####################################################################
####### Reduction / Extraction #####################################
####################################################################

redux_settings = NotImplemented

####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# Default forward model settings
forward_model_settings = {
    
    # X corr options
    'xcorr_options': {'method': 'weighted_brute_force', 'weights': [], 'n_bs': 1000, 'step': 50, 'range': 1E4},
    
    # The cropped pixels
    'crop_data_pix': [200, 200],
    
    # The units for plotting
    'plot_wave_unit': 'microns',

    # Crops (masks) data pixels
    'crop_pix': [200, 200],
    
    'observatory': observatory
}

# Forward model blueprints for RVs
# No default blueprints are defined.
forward_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'StarModel',
        'input_file': None,
        'vel': [-1000 * 300, 10, 1000 * 300]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'sim_tellurics',
        'class_name': 'TelluricModelTAPAS',
        'vel': [-500, -100, 500],
        'species': {
            'water': {
                'input_file': 'telluric_water_tapas_palomar.npz',
                'depth':[0.01, 1.5, 4.0]
            },
            'methane': {
                'input_file': 'telluric_methane_tapas_palomar.npz',
                'depth': [0.1, 1.0, 3.0]
            },
            'nitrous_oxide': {
                'input_file': 'telluric_nitrous_oxide_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0]
            },
            'carbon_dioxide': {
                'input_file': 'telluric_carbon_dioxide_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0]
            },
            'oxygen': {
                'input_file': 'telluric_oxygen_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0]
            },
            'ozone': {
                'input_file': 'telluric_ozone_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0]
            }
        }
    },
    
    'blaze': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class_name': 'ResidualBlazeModel',
        'n_splines': 0,
        'base_quad': [0, 0, 0],
        'base_lin': [-0.0001, 1E-5, 0.0001],
        'base_zero': [0.99, 1.0, 1.01],
        'spline': [-0.135, 0.01, 0.135],
        'n_delay': 0,
        'n_delay_splines': 0
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 6,
        'n_delay': 0,
        'compress': 64,
        'width': [0.055, 0.12, 0.2], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Determined by splines
    'wavelength_solution': {
        
        'name': 'wavelength_sol_known',
        'class_name': 'WaveModelHybrid',
        
        'n_splines': 0,
        'n_delay_splines': 0,
        'spline': [-0.5, 0.01, 0.5]
    }
}