import os
import numpy as np
import pychell.rvs

#############################
####### Name and Site #######
#############################

spectrograph = 'MinervaAustralis'

observatory = {
    'name': 'Mt. Kent',
    'lat': -27.7977,
    'lon': 151.8554,
    'alt': 682
}

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
        'class_name': 'Star',
        'input_file': None,
        'vel': [-400000, 0, 400000]
    },
    
    # Tellurics (from TAPAS) NOTE: Still need proper tellurics, so steal Whipple
    'tellurics': {
        'name': 'vis_tellurics',
        'class_name': 'TelluricsTAPAS',
        'vel': [-300, 0, 300],
        'water_depth': [0.01, 1.5, 4.0],
        'airmass_depth': [0.8, 1.2, 4.0],
        'min_range': 0.01,
        'input_files': {
            'water': 'telluric_water_tapas_whipple.npz',
            'methane': 'telluric_methane_tapas_whipple.npz',
            'nitrous_oxide': 'telluric_nitrous_oxide_tapas_whipple.npz',
            'carbon_dioxide': 'telluric_carbon_dioxide_tapas_whipple.npz',
            'oxygen' : 'telluric_oxygen_tapas_whipple.npz',
            'ozone': 'telluric_ozone_tapas_whipple.npz'
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'blaze', # The blaze model after a division from a flat field
        'class_name': 'SplineBlaze',
        'n_splines': 5,
        'poly_2': [-5.5E-5, -2E-6, 5.5E-5],
        'poly_1': [-0.001, 1E-5, 0.001],
        'poly_0': [0.96, 1.0, 1.08],
        'spline': [0.2, 0.8, 1.1],
        
        # Blaze is centered on the blaze wavelength. Crude estimates unless using a full blaze model
        'blaze_wavelengths': [4858.091694040058, 4896.964858182707, 4936.465079384465, 4976.607650024426, 5017.40836614558, 5058.88354743527, 5101.050061797753, 5143.9253397166585, 5187.527408353689, 5231.87491060088, 5276.98712989741, 5322.884028578407, 5369.586262921349, 5417.11522691744, 5465.493074938935, 5514.742760771861, 5564.888075329751, 5615.953682999512, 5667.96515950171, 5720.949036590132, 5774.932851929652, 5829.94518764045, 5886.015725989253, 5943.1753026380065, 6001.455961651197, 6060.891016560821, 6121.515108109428, 6183.364282120176, 6246.47605505618]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'HermiteLSF',
        'hermdeg': 0,
        'n_delay': 1,
        'width': [0.0234, 0.0234, 0.0234], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        'name': 'wavesol_ThAr_I2',
        'class_name': 'HybridWavelengthSolution',
        'n_splines': 0, # Zero until I2 cell is implemented
        'spline': [-0.03, 0.0005, 0.03]
    }
}