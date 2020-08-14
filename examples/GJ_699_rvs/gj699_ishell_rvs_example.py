import pychell.rvs.driver
import os

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
    "tag": "default_test_run",
    
    # The echelle orders to run
    "do_orders": [11, 13, 15],
    
    # The templates required for fitting (a one time download)
    "templates_path": os.path.dirname(os.path.realpath(__file__)) + os.sep, # For current directory
    
    # This is optional and defaults to False, but for the first time it must be set to True, so we include it here.
    "force_download_templates": True,
    
    # OPTIONAL (and have defaults):
    
    # Number of times to fit with a real stellar template.
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
    
    # Once a first stellar template is generated, it may be updated via Adam optimizing (gradient based) (akine to Wobble).
    # Possible entries are 'star' and/or 'lab'
    # Here, we still use the default of not optimizing the star with Adam.
    "templates_to_optimize": [],
    
    # n_model_pix = model_resolution * n_data_pix
    "model_resolution": 8,
    
    # If true, only computes bc info and exits (by default is false)
    "compute_bc_only": False,
    
}

# This dictionary can be empty to use the instrument provided default model
# Here we overwrite some settings to make the run finish in a timely manner.
model_blueprints = {
    
    # The star
    'star': {
        'input_file': None # To start from a flat template (no star)
    },
    
    # The default blaze is defined with cubic splines for iSHELL.
    'blaze': {
        'n_splines': 6,
        "poly_order": None
    },
    
    # Hermite Gaussian LSF of degree 2 (three terms)
    'lsf': {
        'hermdeg': 2
    },
    
    # iSHELL wls is determined by cubic splines
    'wavelength_solution': {
        'n_splines': 6,
        "poly_order": None # Purely splines
    },
    
    # Fabry Perot cavity with two parameters
    # Disable completely, since this example data relies on fringing removed through flat division
    'fringing': {
        'n_delay': 100
    }
}

pychell.rvs.driver.fit_target(forward_model_settings, model_blueprints)