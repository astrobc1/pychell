import os
import numpy as np
import pychell.rvs
# Other notes for iSHELL:
# blaze_model parameters for a sinc model (full unmodified blaze)
# a: [1.02, 1.05, 1.08], b: [0.008, 0.01, 0.0115], c: [-5, 0.1, 5], d: [0.51, 0.7, 0.9]

# Path to default templates for rvs
default_templates_path = pychell.rvs.__file__[0:-11] + 'default_templates' + os.sep

# Return the file number and date
# ex: icm.2019B047.191109.data.00071.a.fits
# "data" could be anything set by the user, "a" could be b, c, d.
def parse_filename(filename):
    return NotImplemented

#############################
####### General Stuff #######
#############################

# Some general parameters
general_settings = {
    
    # The spectrograph name. Can be anything.
    'spectrograph': 'MinervaAustralis',
    
    # The name of the observatory, not recognized by EarthLocation
    'observatory': 'Mt. Kent',
    
    # Gain of primary detector
    'gain': NotImplemented,
    
    # Dark current of primary detector
    'dark_current': NotImplemented,
    
    # Read noise of the primary detector
    'read_noise': NotImplemented,
    
     # The orientation of the spectral axis for 2d images
    'orientation': NotImplemented,
    
    # The number of data pixels for forward modeling (includes cropped pix on the ends)
    'n_data_pix': 2047,
    
    # increasing => left to right, decreasing => right to left
    'wave_direction': 'increasing',
    
    # The time offset used in the headers
    'time_offset': NotImplemented,
    
    # The tags to recognize science, bias, dark, and flat field images
    'sci_tag': NotImplemented,
    'bias_tag': NotImplemented,
    'darks_tag': NotImplemented,
    'flats_tag': NotImplemented,
    
    # The filename parser
    'filename_parser': NotImplemented,
}

# Header keys for reduction and forward modeling
# The keys are common to all instruments
# The items are lists.
# item[0] = actual key in the header
# item[1] = default values
header_keys = NotImplemented

####################################################################
####### Reduction / Extraction #####################################
####################################################################

# calibration settings
# flat_correlation options are
# 'closest_time' for flats to be applied from the closest in time, 'single' (single set)
# 'closest_space' for flats to be applied from the closest in space angular sepration,
# 'single' for a single set.
calibration_settings = NotImplemented

# Extraction settings
extraction_settings = NotImplemented


####################################################################
####### RADIAL VELOCITIES ##########################################
####################################################################

# Default forward model settings
forward_model_settings = {
    
    # The cropped pixels
    'crop_data_pix': [200, 200],
    
    # The units for plotting
    'plot_wave_unit': 'microns'
}

# Forward model blueprints for RVs
forward_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'StarModelOrderDependent',
        'input_dir': None,
        'vel': [-np.inf, 0, np.inf]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'vis_tellurics', # NOTE: full parameter names are name + component + base_name.
        'class_name': 'TelluricModelTAPAS',
        'vel': [-250, -100, 250],
        'components': {
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
        'class_name': 'ResidualBlazeModel',
        'n_splines': 14,
        'base_quad': [-5.5E-5, -2E-6, 5.5E-5],
        'base_lin': [-0.001, 1E-5, 0.001],
        'base_zero': [0.96, 1.0, 1.08],
        'spline': [-0.135, 0.01, 0.135],
        'n_delay_splines': 0,
        
        # Blaze is centered on the blaze wavelength. Crude estimates
        'blaze_wavelengths': [4858.091694040058, 4896.964858182707, 4936.465079384465, 4976.607650024426, 5017.40836614558, 5058.88354743527, 5101.050061797753, 5143.9253397166585, 5187.527408353689, 5231.87491060088, 5276.98712989741, 5322.884028578407, 5369.586262921349, 5417.11522691744, 5465.493074938935, 5514.742760771861, 5564.888075329751, 5615.953682999512, 5667.96515950171, 5720.949036590132, 5774.932851929652, 5829.94518764045, 5886.015725989253, 5943.1753026380065, 6001.455961651197, 6060.891016560821, 6121.515108109428, 6183.364282120176, 6246.47605505618]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 0,
        'compress': 64,
        'n_delay': 0,
        'width': [0.055, 0.12, 0.2], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        'name': 'wavesol_ThAr_I2',
        'class_name': 'WaveModelKnown',
        'n_splines': 0, # Zero until I2 cell is implemented
        'n_delay_splines': 0,
        'spline': [-0.03, 0.0005, 0.03]
    }
}