import os
import numpy as np
import pychell.rvs

# Path to default templates for rvs
default_templates_path = pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep

#############################
####### Name and Site #######
#############################

spectrograph = 'MinervaAustralis'
observatory = 'Mt. Kent'

####################################################################
####### Reduction / Extraction #####################################
####################################################################

redux_settings = NotImplemented


####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

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
        'vel': [-400000, 0, 400000]
    },
    
    # Tellurics (from TAPAS), NOTE: stealing CTIO until TAPAS is updated!!!!
    'tellurics': {
        'name': 'vis_tellurics', # NOTE: full parameter names are name + component + base_name.
        'class_name': 'TelluricModelTAPAS',
        'vel': [-250, -100, 250],
        'species': {
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
        'class_name': 'SplineBlazeModel',
        'n_splines': 5,
        'base_quad': [-5.5E-5, -2E-6, 5.5E-5],
        'base_lin': [-0.001, 1E-5, 0.001],
        'base_zero': [0.96, 1.0, 1.08],
        'spline': [-0.1, 0.01, 0.1],
        'n_delay_splines': 0,
        
        # Blaze is centered on the blaze wavelength. Crude estimates unless using a full blaze model
        'blaze_wavelengths': [4858.091694040058, 4896.964858182707, 4936.465079384465, 4976.607650024426, 5017.40836614558, 5058.88354743527, 5101.050061797753, 5143.9253397166585, 5187.527408353689, 5231.87491060088, 5276.98712989741, 5322.884028578407, 5369.586262921349, 5417.11522691744, 5465.493074938935, 5514.742760771861, 5564.888075329751, 5615.953682999512, 5667.96515950171, 5720.949036590132, 5774.932851929652, 5829.94518764045, 5886.015725989253, 5943.1753026380065, 6001.455961651197, 6060.891016560821, 6121.515108109428, 6183.364282120176, 6246.47605505618]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 0,
        'compress': 64,
        'n_delay': 0,
        'width': [0.050, 0.12, 0.2], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        'name': 'wavesol_ThAr_I2',
        'class_name': 'WaveModelHybrid',
        'n_splines': 0, # Zero until I2 cell is implemented
        'n_delay_splines': 0,
        'spline': [-0.03, 0.0005, 0.03]
    }
}