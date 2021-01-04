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
observatory = {"name": 'IRTF'}

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
        'class': 'AugmentedStar',
        'augmenter': 'weighted_median',
        'input_file': None,
        'vel': [-1000 * 400, 10, 1000 * 400]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'kband_tellurics',
        'class': 'TelluricsTAPAS',
        'vel': [-500, -100, 500],
        'water_depth': [0.01, 1.2, 5.0],
        'airmass_depth': [0.8, 1.2, 4.0],
        'min_range': 0.01,
        'flag_thresh': [0.05, 0.98], # below this level of norm flux is flagged
        'flag_and_ignore': 0,
        'input_files': {
            'water': 'telluric_water_tapas_maunakea.npz',
            'methane': 'telluric_methane_tapas_maunakea.npz',
            'nitrous_oxide': 'telluric_nitrous_oxide_tapas_maunakea.npz',
            'carbon_dioxide': 'telluric_carbon_dioxide_tapas_maunakea.npz',
            'oxygen' : 'telluric_oxygen_tapas_maunakea.npz',
            'ozone': 'telluric_ozone_tapas_maunakea.npz'
        }
    },
    
    'continuum': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class': 'SplineContinuum',
        'n_splines': 8,
        'poly_order': 2,
        'poly_2': [0, 0, 0],
        'poly_1': [-0.0001, 1E-5, 0.0001],
        'poly_0': [0.99, 1.0, 1.01],
        'spline': [0.3, 0.95, 1.1],
        'n_delay': 0
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class': 'HermiteLSF',
        'hermdeg': 0,
        'n_delay': 0,
        'width': [0.028, 0.04, 0.072], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Determined by splines
    'wavelength_solution': {
        
        'name': 'wls_known',
        'class': 'HybridWavelengthSolution',
        
        'n_splines': 0,
        'n_delay_splines': 0,
        'spline': [-0.5, 0.01, 0.5]
    }
}