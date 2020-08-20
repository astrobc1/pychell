import os
import numpy as np
import pychell.rvs

spectrograph = "PARVI"
observatory = {"name" :"Palomar"}

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
        'class_name': 'Star',
        'input_file': None,
        'vel': [-3E5, 100, 3E5]
    },
    
    # Tellurics (from TAPAS) NOTE: Still need proper tellurics, so steal Whipple
    'tellurics': {
        'name': 'nir_tellurics',
        'class_name': 'TelluricsTAPASV2',
        'vel': [-300, 0, 300],
        'water_depth': [0.01, 1.5, 4.0],
        'airmass_depth': [0.8, 1.2, 4.0],
        'min_range': 0.01,
        'input_files': {
            'water': 'telluric_water_tapas_palomar.npz',
            'methane': 'telluric_methane_tapas_palomar.npz',
            'nitrous_oxide': 'telluric_nitrous_oxide_tapas_palomar.npz',
            'carbon_dioxide': 'telluric_carbon_dioxide_tapas_palomar.npz',
            'oxygen' : 'telluric_oxygen_tapas_palomar.npz',
            'ozone': 'telluric_ozone_tapas_palomar.npz'
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'blaze', # The blaze model after a division from a flat field
        'class_name': 'PolyBlaze',
        'n_splines': 0,
        'poly_2': [0, 0, 0],
        'poly_1': [0, 0, 0],
        'poly_0': [1, 1, 1]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'HermiteLSF',
        'hermdeg': 0,
        'compress': 64,
        'n_delay': 0,
        'width': [0.08, 0.08, 0.08], # LSF width, in angstroms
        'ak': [-0.075, 0.001, 0.075] # See cale et al 2019 or arfken et al some year for definition of ak > 0
    },
    
    # Frequency comb (no splines since no gas cell)
    'wavelength_solution': {
        'name': 'laser_comb_wls',
        'class_name': 'HybridWavelengthSolution',
        'n_splines': 0,
        'n_delay_splines': 0,
        'spline': [-0.15, 0.01, 0.15]
    }
}