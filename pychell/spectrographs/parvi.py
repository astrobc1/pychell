import os
import numpy as np
import pychell.rvs

spectrograph = "PARVI"
observatory = "Palomar"

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
    'crop_data_pix': [10, 10],
    
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
        'vel': [-2000, -100, 500],
        'species': {
            'water': {
                'input_file': 'telluric_water_tapas_palomar.npz',
                'depth': [0.01, 1.5, 4.0]
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
                'depth': [0.1, 1.1, 3.0]
            },
            'ozone': {
                'input_file': 'telluric_ozone_tapas_palomar.npz',
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
        'width': [0.16, 0.18, 0.21], # LSF width, in angstroms (slightly larger than this for PARVI)
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