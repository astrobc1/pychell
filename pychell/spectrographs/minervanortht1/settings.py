import os
import numpy as np
import pychell.rvs

# Path to default templates for rvs
default_templates_path = pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep

#############################
####### Name and Site #######
#############################

spectrograph = 'MinervaNorthT1'
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
                'input_file': default_templates_path + 'telluric_water_tapas_whipple_vis.npz',
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
        'base_c': [-2, 0.01, 2],
        'base_d': [0.51, 0.7, 0.9],
        'spline': [-0.135, 0.01, 0.135],
        
        # Blaze is centered on the blaze wavelength.
        'blaze_wavelengths': [NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, 5095.27251992, 5138.06717456, 5181.58172007, 5225.91306865, 5270.91367833, 5316.74231748, 5363.31375096, 5410.80093697, 5459.1389083 , 5508.26062352, 5558.28621769, 5609.30469297, 5661.18981344, 5714.08104978, 5768.00209798, 5822.85465602, 5878.80712302, 5935.8355058, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented]
    },
    
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 0,
        'n_delay': 0,
        'compress': 64,
        'width': [0.010, 0.014, 0.018], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        
        'name': 'lagrange_wavesol_splines',
        'class_name': 'WaveSolModelFull',
        
        # The three pixels to span the detector
        'base_pixel_set_points': [19, 1023.5, 2027],
        
        # Left most set point for the quadratic wavelength solution
        #'base_set_point_1': [NotImplemented, NotImplemented, 5074.45624048, 5117.06409566, 5160.36846328, 5204.4086953, 5249.29954288, 5294.98064148, 5341.33622943, 5388.68457122, 5436.73709434, 5485.67105667, 5535.53619346, 5586.32149207, 5638.02502112, 5690.68013838, 5744.27794608, 5799.07700679, 5854.8232137 , 5911.63095474, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented],

        # Middle set point for the quadratic wavelength solution
        #'base_set_point_2': [NotImplemented, NotImplemented, 5095.27251992, 5138.06717456, 5181.58172007, 5225.91306865, 5270.91367833, 5316.74231748, 5363.31375096, 5410.80093697, 5459.1389083 , 5508.26062352, 5558.28621769, 5609.30469297, 5661.18981344, 5714.08104978, 5768.00209798, 5822.85465602, 5878.80712302, 5935.8355058, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented],

        # Right most set point for the quadratic wavelength solution
        #'base_set_point_3': [NotImplemented, NotImplemented, 5114.15170385, 5157.1102395 , 5200.73723567, 5245.00176499, 5290.22850819, 5336.22402975, 5383.11334993, 5430.72915821, 5479.1049699 , 5528.34996057, 5578.92845383, 5629.71798286, 5681.84042672, 5735.0674727 , 5788.94554486, 5844.28397302, 5900.54814152, 5957.86374969, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented],
        
        
        # Left most set point for the quadratic wavelength solution
        'base_set_point_1': [NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, 5074.45624048, 5117.06409566, 5160.36846328, 5204.4086953, 5249.29954288, 5294.98064148, 5341.33622943, 5388.68457122, 5436.73709434, 5485.67105667, 5535.53619346, 5586.32149207, 5638.02502112, 5690.68013838, 5744.27794608, 5799.07700679, 5854.8232137 , 5911.63095474, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented],

        # Middle set point for the quadratic wavelength solution
        'base_set_point_2': [NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, 5095.27251992, 5138.06717456, 5181.58172007, 5225.91306865, 5270.91367833, 5316.74231748, 5363.31375096, 5410.80093697, 5459.1389083 , 5508.26062352, 5558.28621769, 5609.30469297, 5661.18981344, 5714.08104978, 5768.00209798, 5822.85465602, 5878.80712302, 5935.8355058, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented],

        # Right most set point for the quadratic wavelength solution
        'base_set_point_3': [NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, 5114.15170385, 5157.1102395 , 5200.73723567, 5245.00176499, 5290.22850819, 5336.22402975, 5383.11334993, 5430.72915821, 5479.1049699 , 5528.34996057, 5578.92845383, 5629.71798286, 5681.84042672, 5735.0674727 , 5788.94554486, 5844.28397302, 5900.54814152, 5957.86374969, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented, NotImplemented],
        
        'n_splines': 0,
        'n_delay_splines': 0,
        'base': [-0.1, -0.01, 0.1],
        'spline': [-0.15, 0.01, 0.15]
    }
}