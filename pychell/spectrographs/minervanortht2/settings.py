import os
import numpy as np
import pychell.rvs

# Path to default templates for rvs
default_templates_path = pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep

#############################
####### Name and Site #######
#############################

spectrograph = 'MinervaNorthT2'
observatory = 'Whipple'

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
    'crop_data_pix': [20, 20],
    
    # The units for plotting
    'plot_wave_unit': 'nm',
    
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
    
    # The methane gas cell
    'gas_cell': {
        'name': 'iodine_gas_cell',
        'class_name': 'GasCellModel',
        'input_file': default_templates_path + 'minerva_north_iodine_template_nist.npz',
        'shift': [0, 0, 0],
        'depth': [1, 1, 1]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'vis_tellurics',
        'class_name': 'TelluricModelTAPAS',
        'vel': [-500, -100, 500],
        'species': {
            'water': {
                'input_file': default_templates_path + 'telluric_water_tapas_whipple_vis.npz,
                'depth':[0.01, 1.5, 4.0]
            },
            'ozone': {
                'input_file': default_templates_path + 'telluric_ozone_tapas_whipple_vis.npz',
                'depth': [0.1, 1.0, 3.0]
            },
            'oxygen': {
                'input_file': default_templates_path + 'telluric_oxygen_tapas_whipple_vis.npz',
                'depth': [0.05, 0.65, 3.0]
            }
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'full_blaze', # The blaze model after a division from a flat field
        'class_name': 'FullBlazeModel',
        'n_splines': 0,
        'n_delay_splines': 0,
        'base_amp': [1.0, 1.05, 1.4],
        'base_b': [0.008, 0.01, 0.04],
        'base_c': [-10, 0.01, 10],
        'base_d': [0.51, 0.7, 0.9],
        'spline': [-0.135, 0.01, 0.135],
        
        # Blaze is centered on the blaze wavelength.
        'blaze_wavelengths': np.array([5025.0, 5057.142857142857, 5089.285714285715, 5121.428571428572, 5153.571428571428, 5185.714285714285, 5217.857142857143, 5250.0, 5282.142857142857, 5314.285714285715, 5346.428571428572, 5378.571428571428, 5410.714285714286, 5442.857142857143, 5475.0, 5507.142857142857, 5539.285714285715, 5571.428571428572, 5603.571428571428, 5635.714285714286, 5667.857142857143, 5700.0, 5732.142857142857, 5764.285714285715, 5796.428571428572, 5828.571428571428, 5860.714285714286, 5892.857142857143, 5925.0])
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
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        
        'name': 'lagrange_wavesol_splines',
        'class_name': 'WaveSolModelFull',
        
        # The three pixels to span the detector corresponding to the above wavelengths
        # They are chosen as such because we typically use pixels 200-1848 only.
        # These pixels must correspond to the wavelengths in the array wavesol_base_wave_set_points_i[order]
        'base_pixel_set_points': [99, 1023.5, 1947],
        
        # Left most set point for the quadratic wavelength solution
        'base_set_point_1': [5000.0, 5032.142857142857, 5064.285714285715, 5096.428571428572, 5128.571428571428, 5160.714285714285, 5192.857142857143, 5225.0, 5257.142857142857, 5289.285714285715, 5321.428571428572, 5353.571428571428, 5385.714285714286, 5417.857142857143, 5450.0, 5482.142857142857, 5514.285714285715, 5546.428571428572, 5578.571428571428, 5610.714285714286, 5642.857142857143, 5675.0, 5707.142857142857, 5739.285714285715, 5771.428571428572, 5803.571428571428, 5835.714285714286, 5867.857142857143, 5900.0],

        # Middle set point for the quadratic wavelength solution
        'base_set_point_2': [5025.0, 5057.142857142857, 5089.285714285715, 5121.428571428572, 5153.571428571428, 5185.714285714285, 5217.857142857143, 5250.0, 5282.142857142857, 5314.285714285715, 5346.428571428572, 5378.571428571428, 5410.714285714286, 5442.857142857143, 5475.0, 5507.142857142857, 5539.285714285715, 5571.428571428572, 5603.571428571428, 5635.714285714286, 5667.857142857143, 5700.0, 5732.142857142857, 5764.285714285715, 5796.428571428572, 5828.571428571428, 5860.714285714286, 5892.857142857143, 5925.0],

        # Right most set point for the quadratic wavelength solution
        'base_set_point_3': [5050.0, 5082.142857142857, 5114.285714285715, 5146.428571428572, 5178.571428571428, 5210.714285714285, 5242.857142857143, 5275.0, 5307.142857142857, 5339.285714285715, 5371.428571428572, 5403.571428571428, 5435.714285714286, 5467.857142857143, 5500.0, 5532.142857142857, 5564.285714285715, 5596.428571428572, 5628.571428571428, 5660.714285714286, 5692.857142857143, 5725.0, 5757.142857142857, 5789.285714285715, 5821.428571428572, 5853.571428571428, 5885.714285714286, 5917.857142857143, 5950.0],
        
        'n_splines': 6,
        'n_delay_splines': 0,
        'base': [-0.35, -0.05, 0.2],
        'spline': [-0.15, 0.01, 0.15]
    }
}