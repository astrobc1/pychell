import pychell.rvs.driver
import os

# This dictionary must contain the following required args.
forward_model_settings = {
    
    # REQUIRED:
    
    # Name of the spectrograph.
    "spectrograph": "iSHELL",
    
    # Input path to the data
    "input_path": "data/",
    
    # Filelist contains base filenames of all spectra to be used in this run and must exist in input_path
    "flist_file": "filelist_example.txt",
    
    # Output path to store the results
    "output_path_root": os.path.dirname(os.path.realpath(__file__)) + os.sep, # For current directory
    
    # Name of the star, must be recognized recognized by SIMBAD if queurying
    "star_name": "GJ_699",
    
    # Appended to the front of all filenames
    "tag": "defaul_test_run",
    
    # The echelle orders to run
    "do_orders": [11, 13, 15],
    
    # Some optional arguments:
    
    # Number of times to fit with a real stellar template
    "n_template_fits": 3,
    
    # Number of cores, parallelized over spectra
    "n_cores": 1,
    
    # Helpful diagnistics
    "verbose_plot": True,
    "verbose_print": True,
    
    # Which nights to use when augmenting the template, empty list means use all nights
    "nights_for_template": [],
    
    # Which function to use in augmenting the first set of residuals to generate a stellar template
    # cslsq = cubic spline regression
    "template_augmenter": 'cubic_spline_lsq',
    
    # Once a first stellar template is generated, it may be updated via Adam (gradient-based) optimizing (akine to Wobble).
    # Here, we still use the default of not optimizing the star with Adam.
    "templates_to_optimize": [],
    
    # n_model_pix = model_resolution * n_data_pix
    "model_resolution": 8,
    
    # If true, only computes bc info and exits
    "compute_bc_only": False
}

# This dictionary can be empty to use the instrument provided default model
# Here we overwrite some settings to make the run finish in a timely manner.
model_blueprints = {
    
    # The star
    'star': {
        'input_file': None
    },
    
    # The default blaze is a quadratic + splines for iSHELL.
    'blaze': {
        'n_splines': 6,
        'n_delay_splines': 1
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'hermdeg': 2
    },
    
    # Quadratic (3 Lagrange points) + splines
    'wavelength_solution': {
        'n_splines': 6,
        'n_delay': 0
    },
    
    # Fabry Perot cavity with two parameters
    # Disable completely, since this example data relies on fringing removed through flat division
    'fringing': {
        'n_delay': 100
    }
}

pychell.rvs.driver.fit_target(forward_model_settings, model_blueprints)