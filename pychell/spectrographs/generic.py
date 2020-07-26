import os
import numpy as np
import pychell.rvs

# Path to default templates for rvs
default_templates_path = pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep

def parse_filename(filename):
    return NotImplemented

#############################
####### General Stuff #######
#############################

# Some general parameters
general_settings = {
    
    # The spectrograph name. Can be anything.
    'spectrograph': 'GENERIC',
    
    # The name of the observatory.
    # Must be a recognized astropy EarthLocation if not computing own barycenter info.
    'observatory': 'GENERIC',
    
    # Gain of primary detector
    'gain': 1.0,
    
    # Dark current of primary detector
    'dark_current': 0.0,
    
    # Read noise of the primary detector
    'read_noise': 0.0,
    
     # The orientation of the spectral axis for 2d images
    'orientation': 'x',
    
    # The number of data pixels for forward modeling (includes cropped pix on the ends)
    'n_data_pix': NotImplemented,
    
    # increasing => left to right, decreasing => right to left
    'wave_direction': 'decreasing',
    
    # The time offset used in the headers
    'time_offset': 2400000.5,
    
    # The tags to recognize science, bias, dark, and flat field images
    'sci_tag': 'data',
    'bias_tag': 'bias',
    'darks_tag': 'dark',
    'flats_tag': 'flat',
    
    # The filename parser
    'filename_parser': parse_filename
}

# Header keys for reduction
# NOTE: For now, this is only suedin reduction.
# The keys are common to all instruments
# The items are lists.
# item[0] = actual key in the header
# item[1] = default values
header_keys = {
    'target': ['TCS_OBJ', 'STAR'],
    'RA': ['TCS_RA', '00:00:00.0'],
    'DEC': ['TCS_DEC', '00:00:00.0'],
    'slit': ['SLIT', '0.0'],
    'wavelength_range': ['XDTILT', 'NA'],
    'gas_cell': ['GASCELL', 'NA'],
    'exp_time': ['ITIME', 'NA'],
    'time_of_obs': ['TCS_UTC', 'NA'],
    'NDR': ['NDR', 1],
    'BZERO': ['BZERO', 0],
    'BSCALE': ['BSCALE', 1]
}

####################################################################
####### Reduction / Extraction #####################################
####################################################################

# calibration settings
# flat_correlation options are
# 'closest_time' for flats to be applied from the closest in time, 'single' (single set)
# 'closest_space' for flats to be applied from the closest in space angular sepration,
# 'single' for a single set.
calibration_settings = {
    'dark_subtraction': False,
    'flat_division': False,
    'bias_subtraction': False,
    'wavelength_calibration': False
}

# Extraction settings
extraction_settings = {
    
    # Order map algorithm (options: 'from_flats, 'empirical')
    'order_map': 'empirical',
    
    # Pixels to mask on the top, bottom, left, and right edges
    'mask_left_edge': 10,
    'mask_right_edge': 10,
    'mask_top_edge': 10,
    'mask_bottom_edge': 10,
    
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
    'oversample': 4
}


####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# Default forward model settings
forward_model_settings = NotImplemented

# Forward model blueprints for RVs
# No default blueprints are defined.
forward_model_blueprints = NotImplemented

# #forward_model_blueprints = {
    
#     # The star
#     'star': {
#         'name': 'star',
#         'class_name': 'StarModel',
#         'input_file': None,
#         'vel': [-1000 * 300, 10, 1000 * 300]
#     },
    
#     # Tellurics (from TAPAS)
#     'tellurics': {
#         'name': 'kband_tellurics',
#         'class_name': 'TelluricModelTAPAS',
#         'vel': [-250, -100, 100],
#         'species': {
#             'water': {
#                 'input_file': default_templates_path + 'telluric_water_tapas_maunakea.npz',
#                 'depth':[0.01, 1.5, 4.0]
#             },
#             'methane': {
#                 'input_file': default_templates_path + 'telluric_methane_tapas_maunakea.npz',
#                 'depth': [0.1, 1.0, 3.0]
#             },
#             'nitrous_oxide': {
#                 'input_file': default_templates_path + 'telluric_nitrous_oxide_tapas_maunakea.npz',
#                 'depth': [0.05, 0.65, 3.0]
#             },
#             'carbon_dioxide': {
#                 'input_file': default_templates_path + 'telluric_carbon_dioxide_tapas_maunakea.npz',
#                 'depth': [0.05, 0.65, 3.0]
#             }
#         }
#     },
    
#     # The default blaze is a quadratic + splines.
#     'blaze': {
#         'name': 'residual_blaze', # The blaze model after a division from a flat field
#         'class_name': 'ResidualBlazeModel',
#         'n_splines': 14,
#         'base_quad': [-5.5E-5, -2E-6, 5.5E-5],
#         'base_lin': [-0.001, 1E-5, 0.001],
#         'base_zero': [0.96, 1.0, 1.08],
#         'spline': [-0.135, 0.01, 0.135],
#         'n_delay_splines': 0,
        
#         # Blaze is centered on the blaze wavelength. Crude estimates
#         'blaze_wavelengths': 
#     },
    
#     # Hermite Gaussian LSF
#     'lsf': {
#         'name': 'lsf_hermite',
#         'class_name': 'LSFHermiteModel',
#         'hermdeg': 6,
#         'n_delay': 0,
#         'compress': 64,
#         'width': [0.055, 0.12, 0.2], # LSF width, in angstroms
#         'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
#     },
    
#     # Quadratic (Lagrange points) + splines
#     'wavelength_solution': {
        
#         'name': 'lagrange_wavesol_splines',
#         'class_name': 'WaveSolModelFull',
        
#         # The three pixels to span the detector corresponding to the above wavelengths
#         # They are chosen as such because we typically use pixels 200-1848 only.
#         # These pixels must correspond to the wavelengths in the array wavesol_base_wave_set_points_i[order]
#         'base_pixel_set_points': [199, 1023.5, 1847],
        
#         # Left most set point for the quadratic wavelength solution
#         'base_set_point_1': [24545.57561435, 24431.48444449, 24318.40830764, 24206.35776048, 24095.33986576, 23985.37381209, 23876.43046386, 23768.48974584, 23661.54443537, 23555.56359209, 23450.55136357, 23346.4923953, 23243.38904298, 23141.19183839, 23039.90272625, 22939.50127095, 22840.00907242, 22741.40344225, 22643.6481698, 22546.74892171, 22450.70934177, 22355.49187891, 22261.08953053, 22167.42305394, 22074.72848136, 21982.75611957, 21891.49178289, 21801.07332421, 21711.43496504],

#         # Middle set point for the quadratic wavelength solution
#         'base_set_point_2': [24628.37672608, 24513.79686837, 24400.32734124, 24287.85495107, 24176.4424356, 24066.07880622, 23956.7243081, 23848.39610577, 23741.05658955, 23634.68688897, 23529.29771645, 23424.86836784, 23321.379387, 23218.80573474, 23117.1876433, 23016.4487031, 22916.61245655, 22817.65768889, 22719.56466802, 22622.34315996, 22525.96723597, 22430.41612825, 22335.71472399, 22241.83394135, 22148.73680381, 22056.42903627, 21964.91093944, 21874.20764171, 21784.20091295],

#         # Right most set point for the quadratic wavelength solution
#         'base_set_point_3': [24705.72472863, 24590.91231465, 24476.99298677, 24364.12010878, 24252.31443701, 24141.55527091, 24031.82506843, 23923.12291214, 23815.40789995, 23708.70106907, 23602.95596074, 23498.18607941, 23394.35163611, 23291.44815827, 23189.49231662, 23088.42080084, 22988.26540094, 22888.97654584, 22790.57559244, 22693.02942496, 22596.33915038, 22500.49456757, 22405.49547495, 22311.25574559, 22217.91297633, 22125.33774808, 22033.50356525, 21942.41058186, 21852.24253555],
        
#         'n_splines': 6,
#         'n_delay_splines': 0,
#         'base': [-0.35, -0.05, 0.2],
#         'spline': [-0.15, 0.01, 0.15]
#     }
# }