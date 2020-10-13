import os
import numpy as np
import pychell.rvs

# Path to default templates for rvs
default_templates_path = pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep

#############################
####### Name and Site #######
#############################

spectrograph = 'NIRSPEC'
observatory = {"name" : 'Keck'}

####################################################################
####### Reduction / Extraction #####################################
####################################################################

redux_settings = {
    
    # Detector properties
    'detector_props' : [{'gain': 1.0, 'dark_current': 0.00, 'read_noise': 1.0}],
    
    # Calibration
    'dark_subtraction': False,
    'flat_division': False,
    'bias_subtraction': False,
    'wavelength_calibration': False,
    'flatfield_percentile': 0.85,
    
    # Pixels to mask on the top, bottom, left, and right edges
    'mask_left_edge': 20,
    'mask_right_edge': 20,
    'mask_top_edge': 20,
    'mask_bottom_edge': 20,
    
    # The height of an order is defined as where the flat is located.
    # This masks additional pixels on each side of the initial trace profile before moving forward.
    # The profile is further flagged after thes sky background is estimated.
    'mask_trace_edges':  3,
    
    # The degree of the polynomial to fit the individual order locations
    'trace_pos_polyorder' : 2,
    
    # Whether or not to perform a sky subtraction
    # The number of rows used to estimate the sky background (lowest n_sky_rows in the trace profile are used).
    'sky_subtraction': True,
    'n_sky_rows': 8,
    
    # The trace profile is constructed using oversampled data.
    # This is the oversample factor.
    'oversample': 4,
    
    # The optimal extraction algorithm
    'optx_alg': 'pmassey_wrapper',
    'order_map': {'source': 'empirical_unique', 'method': None}
    
}


####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# Default forward model settings
forward_model_settings = {
    
    # The cropped pixels
    'crop_data_pix': [100, 100],
    
    # The units for plotting
    'plot_wave_unit': 'microns',
    
    'observatory': observatory
}

# Forward model blueprints for RVs
# No default blueprints are defined.
forward_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'Star',
        'input_file': None,
        'vel': [-1000 * 300, 10, 1000 * 300]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'vis_tellurics',
        'class_name': 'TelluricsTAPAS',
        'vel': [-300, 0, 300],
        'water_depth': [0.01, 1.5, 4.0],
        'airmass_depth': [0.8, 1.2, 4.0],
        'min_range': 0.01,
        'input_files': {
            'water': 'telluric_water_tapas_maunakea.npz',
            'methane': 'telluric_methane_tapas_maunakea.npz',
            'nitrous_oxide': 'telluric_nitrous_oxide_tapas_maunakea.npz',
            'carbon_dioxide': 'telluric_carbon_dioxide_tapas_maunakea.npz',
            'oxygen' : 'telluric_oxygen_tapas_maunakea.npz',
            'ozone': 'telluric_ozone_tapas_maunakea.npz'
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class_name': 'SplineBlaze',
        'n_splines': 14,
        'poly_2': [-5.5E-5, -2E-6, 5.5E-5],
        'poly_1': [-0.001, 1E-5, 0.001],
        'poly_0': [0.96, 1.0, 1.08],
        'spline': [-0.135, 0.01, 0.135]
        
        # Blaze is centered on the blaze wavelength. under testing
        'blaze_wavelengths': [20500.0, 0, 0, 0]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'HermiteLSF',
        'hermdeg': 6,
        'n_delay': 0,
        'compress': 64,
        'width': [0.4, 0.4, 0.4], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        
        'name': 'lagrange_wavesol_splines',
        'class_name': 'SplineWavelengthSolution',
        
        # The three pixels to span the detector corresponding to the above wavelengths
        # They are chosen as such because we typically use pixels 200-1848 only.
        # These pixels must correspond to the wavelengths in the array wavesol_base_wave_set_points_i[order]
        'quad_pixel_set_points': [99, 512, 923],
        
        # Left most set point for the quadratic wavelength solution
        'quad_set_point_1': [20426.363142799542, 0, 0, 0],

        # Middle set point for the quadratic wavelength solution
        'quad_set_point_2': [20547.14884503126, 0, 0, 0],

        # Right most set point for the quadratic wavelength solution
        'quad_set_point_3': [20671.664557571028, 0, 0, 0],
        
        'n_splines': 6,
        'poly_lagrange': [-0.35, -0.05, 0.2],
        'spline': [-0.15, 0.01, 0.15]
    },
    
    # Fabry Perot cavity with two parameters
    'fringing': {
        'name': 'fringing',
        'class_name': 'BasicFringingModel',
        'd': [183900000.0, 183911000.0, 183930000.0],
        'fin': [0.01, 0.04, 0.08],
        'n_delay': 10000 # To delay indefinitely, user may wish to enable.
    }
}