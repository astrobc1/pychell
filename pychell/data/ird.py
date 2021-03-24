import os
import numpy as np
import pychell.rvs

spectrograph = "IRD"
observatory = {"name" :"Subaru"}

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
        'class': 'AugmentedStar',
        'input_file': None,
        "augmenter": "weighted_median",
        'vel': [-3E5, 100, 3E5]
    },
    
    # Tellurics (from TAPAS) NOTE: Still need proper tellurics, so steal Whipple
    'tellurics': {
        'name': 'nir_tellurics',
        'class': 'TelluricsTAPAS',
        'vel': [-200, 10, 200],
        'water_depth': [0.03, 1.1, 5],
        'airmass_depth': [0.8, 1.1, 3],
        'min_range': 0.01,
        'flag_thresh': [0.05, 0.5], # below this level of norm flux is flagged
        'flag_and_ignore': 0,
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
    'continuum': {
        'name': 'continuum', # The blaze model after a division from a flat field
        'class': 'SplineContinuum',
        'n_splines': 10,
        'spline': [0.3, 0.99, 1.1],
        'poly_2': [-1E-7, -1E-6, 1E-7],
        'poly_1': [-1E-6, 1E-5, 1E-6],
        'poly_0': [0.99, 1, 1.01]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class': 'HermiteLSF',
        'hermdeg': 0,
        'width': [0.08, 0.08, 0.08], # LSF width, in angstroms
        'ak': [-0.075, 0.001, 0.075] # See cale et al 2019 or arfken et al some year for definition of ak > 0
    },
    
    # Frequency comb (no splines since no gas cell)
    'wavelength_solution': {
        'name': 'laser_comb_wls',
        'class': 'HybridWavelengthSolution',
        'n_splines': 0
    }
}