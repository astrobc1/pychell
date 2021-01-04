import os
import numpy as np
import pychell.rvs

# Other notes for iSHELL:
# blaze_model parameters for a sinc model (full unmodified blaze)
# a: [1.02, 1.05, 1.08], b: [0.008, 0.01, 0.0115], c: [-5, 0.1, 5], d: [0.51, 0.7, 0.9]

#############################
####### Name and Site #######
#############################

spectrograph = 'iSHELL'
observatory = {
    'name': 'IRTF',
    'lat': 19.826218316666665,
    'lon': -155.4719987888889,
    'alt': 4168.066848
}

####################################################################
####### Reduction / Extraction #####################################
####################################################################

redux_settings = {
    
    # Detector properties
    'detector_props' : [{'gain': 1.8, 'dark_current': 0.05, 'read_noise': 8.0}],
    
    # Calibration
    'dark_subtraction': False,
    'flat_division': True,
    'bias_subtraction': False,
    'wavelength_calibration': False,
    'flatfield_percentile': 0.75,
    
    # Pixels to mask on the top, bottom, left, and right edges
    'mask_image_left': 200,
    'mask_image_right': 200,
    'mask_image_top': 20,
    'mask_image_bottom': 20,
    
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
    'order_map': {'source': 'empirical_from_flat_fields', 'method': None}
    
}


####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# Default forward model settings
forward_model_settings = {
    
    # X corr options
    'xcorr_options': {'method': 'weighted_brute_force', 'weights': ['tellurics', 'flux_unc'], 'n_bs': 1000, 'step': 50, 'range': 1E4},
    
    # The cropped pixels
    'crop_data_pix': [200, 200],
    
    # The units for plotting
    'plot_wave_unit': 'microns',
    
    "n_chunks": 1,

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
        'augmenter': 'cubic_spline_lsq',
        'input_file': None,
        'vel': [-1000 * 300, 10, 1000 * 300]
    },
    
    # The methane gas cell
    'gas_cell': {
        'name': 'methane_gas_cell', # NOTE: full parameter names are name + base_name.
        'class': 'DynamicGasCell',
        'input_file': 'methane_gas_cell_ishell_kgas.npz',
        'shift': [0, 0, 0],
        'depth': [0.97, 0.97, 0.97]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'kband_tellurics',
        'class': 'TelluricsTAPAS',
        'vel': [-500, -100, 500],
        'water_depth': [0.01, 1.2, 5.0],
        'airmass_depth': [0.8, 1.2, 4.0],
        'min_range': 0.01,
        'flag_thresh': [0.05, 0.5], # below this level of norm flux is flagged
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
    
    # The default blaze is a quadratic + splines.
    'continuum': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class': 'SplineContinuum',
        'n_splines': 10,
        'poly_order': 2,
        'poly_6': [-5.5E-9, -2E-6, 5.5E-9],
        'poly_5': [-5.5E-8, -2E-6, 5.5E-8],
        'poly_4': [-5.5E-7, -2E-6, 5.5E-7],
        'poly_3': [-5.5E-6, -2E-6, 5.5E-6],
        'poly_2': [-5.5E-5, -2E-6, 5.5E-5],
        'poly_1': [-0.001, 1E-5, 0.001],
        'poly_0': [0.96, 1.0, 1.1],
        'spline': [0.3, 0.95, 1.2],
        'n_delay': 0,
        'n_delay_splines': 0,
        
        # Blaze is centered on the blaze wavelength, ideally.
        'blaze_wavelengths': [24623.42005657, 24509.67655586, 24396.84451226, 24284.92392579, 24173.91479643, 24063.81712419, 23954.63090907, 23846.35615107, 23738.99285018, 23632.54100641, 23527.00061976, 23422.37169023, 23318.65421781, 23215.84820252, 23113.95364434, 23012.97054327, 22912.89889933, 22813.7387125,  22715.48998279, 22618.1527102, 22521.72689473, 22426.21253637, 22331.60963514, 22237.91819101, 22145.13820401, 22053.26967413, 21962.31260136, 21872.26698571, 21783.13282718]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class': 'HermiteLSF',
        'hermdeg': 6,
        'n_delay': 0,
        'width': [0.08, 0.11, 0.15], # LSF width, in angstroms
        #'width': [0.11037, 0.11037, 0.11037],
        'ak': [-0.05, 0.001, 0.1] # Hermite polynomial coefficients
    },
    
    # Determined by splines
    'wavelength_solution': {
        
        'name': 'csplines_wavesol',
        'class': 'SplineWavelengthSolution',
        'name': 'spline_wavesol',
        'poly_order': 4,
        'n_splines': 6,
        
        # The three pixels to span the detector corresponding to the above wavelengths
        # They are chosen as such because we typically use pixels 200-1848 only.
        # These pixels must correspond to the wavelengths in the array wavesol_base_wave_set_points_i[order]
        'quad_pixel_set_points': [199, 1023.5, 1847],
        
        # Left most set point for the quadratic wavelength solution
        'quad_set_point_1': [24545.57561435, 24431.48444449, 24318.40830764, 24206.35776048, 24095.33986576, 23985.37381209, 23876.43046386, 23768.48974584, 23661.54443537, 23555.56359209, 23450.55136357, 23346.4923953, 23243.38904298, 23141.19183839, 23039.90272625, 22939.50127095, 22840.00907242, 22741.40344225, 22643.6481698, 22546.74892171, 22450.70934177, 22355.49187891, 22261.08953053, 22167.42305394, 22074.72848136, 21982.75611957, 21891.49178289, 21801.07332421, 21711.43496504],

        # Middle set point for the quadratic wavelength solution
        'quad_set_point_2': [24628.37672608, 24513.79686837, 24400.32734124, 24287.85495107, 24176.4424356, 24066.07880622, 23956.7243081, 23848.39610577, 23741.05658955, 23634.68688897, 23529.29771645, 23424.86836784, 23321.379387, 23218.80573474, 23117.1876433, 23016.4487031, 22916.61245655, 22817.65768889, 22719.56466802, 22622.34315996, 22525.96723597, 22430.41612825, 22335.71472399, 22241.83394135, 22148.73680381, 22056.42903627, 21964.91093944, 21874.20764171, 21784.20091295],

        # Right most set point for the quadratic wavelength solution
        'quad_set_point_3': [24705.72472863, 24590.91231465, 24476.99298677, 24364.12010878, 24252.31443701, 24141.55527091, 24031.82506843, 23923.12291214, 23815.40789995, 23708.70106907, 23602.95596074, 23498.18607941, 23394.35163611, 23291.44815827, 23189.49231662, 23088.42080084, 22988.26540094, 22888.97654584, 22790.57559244, 22693.02942496, 22596.33915038, 22500.49456757, 22405.49547495, 22311.25574559, 22217.91297633, 22125.33774808, 22033.50356525, 21942.41058186, 21852.24253555],
        
        'poly_lagrange': [-0.35, -0.05, 0.35],
        'spline': [-0.35, 0.01, 0.35]
    },
    
    # Fabry Perot cavity with two parameters
    'fringing': {
        'name': 'fringing',
        'class': 'FPCavityFringing',
        'logd': [19.02990269, 19.0299625 , 19.03006581],
        'fin': [0.01, 0.04, 0.08],
        'n_delay': 1000 # To delay indefinitely, user may wish to enable.
    }
}