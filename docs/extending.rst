Extending
*********

Overview of Pychell
===================

For both spectral reduction/extraction, and generating RVs, pychell provides basic default settings / config to get things started. These settings is stored in config.py and are instrument independent. A given instrument must define some additional configuration and will possibly override some of the default settings. In a user's config file for a given use case, they will provide any final necessary information specific for that run, as well as override either the default config, or instrument specific default config. The following dictionary is loaded into memory for any run.

.. code-block:: python

    general_settings = {

        # Number of cores to use where appropriate. Defaults to 1.
        'n_cores': <int>,

        # Plots helpful diagnostics. Defaults to False.
        'verbose_plot': <bool>,

        # Prints helpful diagnostics. Defaults to False.
        'verbose_print': <bool>,

        # The wavelength units for plotting spectra. Options are nm, ang, microns.
        'plot_wave_unit': <str>,

        # The pipeline path, auto generated.
        'pychell_path': <str>
    }

Define A New Spectrograph
=========================

Each implemented spectrograph must live in a file named insname.py in the spectrographs directory. This file must define the following dictionaries and keys for each use case. If a given implementation is not desired, it is best to set the variable to NotImplemented.

**redux_settings**

.. code-block:: python

    redux_settings = {

        ## USER LEVEL ##
        
        # The name of the spectrograph
        'spectrograph': <str>

        # The full input path.
        'input_path': <str>,

        # The root output path.
        'output_path_root': <str>,

        
        ## INSTRUMENT LEVEL ##

        # A list of dictionaries for each detector. Must define gain, dark_current, and read_noise. If multiple detectors, one must also provide the coordinates (xmin, xmax, ymin, ymax).
        # Ex: [{'gain': 1.8, 'dark_current': 0.05, 'read_noise': 8.0}],
        'detector_props' : <list>,

        ## BOTTOM LEVEL ##
        
        # Whether or not to perform dark subtraction.
        'dark_subtraction': <bool>,

        # Whether or not to perform flat division.
        'flat_division': <bool>,

        # Whether or not to perform bias subtraction.
        'bias_subtraction': <bool>,

        # Whether or not to perform wavelength calibration.
        'wavelength_calibration': <bool>,

        # The percentile in the flat field images to consider as 1. Defaults to 0.75
        'flatfield_percentile': <float>,
        
        # Pixels to mask on the top, bottom, left, and right edges. Defaults to 10 all around.
        'mask_left_edge': <int>,
        'mask_right_edge': <int>,
        'mask_top_edge': <int>,
        'mask_bottom_edge': <int>,
        
        # The height of an order is defined as where the flat is located.
        # This masks additional pixels on each side of the initial trace profile before moving forward.
        # The profile is further flagged after thes sky background is estimated.
        'mask_trace_edges':  <int>,
        
        # The degree of the polynomial to fit the individual order locations
        'trace_pos_polyorder' : <int>,
        
        # Whether or not to perform a sky subtraction
        # The number of rows used to estimate the sky background (lowest n_sky_rows in the trace profile are used).
        'sky_subtraction': <bool>,
        'n_sky_rows': <int>,
        
        # The trace profile is constructed using oversampled data and to properly interpolate the profile for each column.
        'oversample': <int>,
        
        # The optimal extraction algorithm options. Defaults is pmassey_wrapper.
        'optx_alg': <str>,

        # The order map options.
        # Ex: {'source': 'empirical_from_flat_fields', 'method': None}
        'order_map': <dict>
        
    }


**forward_model_settings**

.. code-block:: python

    forward_model_settings = {

        ## USER LEVEL ##

        # The name of the spectrograph
        'spectrograph': <str>

        # The full input path.
        'input_path': <str>,

        # The root output path.
        'output_path_root': <str>,

        # The name of the star (for spaces, use and underscore). Must me found by simbad if calculating BC vels via barycorrpy.
        'star_name': <str>,

        # The base filename containing the base files to consider on each line. Must live in input_path.
        'flist_file': <str>,

        # The unique tag for this run. tag_star_name_insname is included in all filenames and is the name of the output directory for this run.
        'tag': <str>,

        # A list of which orders to run.
        "do_orders": <list>,

        ## INSTRUMENT LEVEL ##

        # The name of the observatory, potentially used to compute the barycenter corrections. Must be a recognized astroy.EarthLocation.
        'observatory': <str>,

        ## BOTTOM LEVEL ##
    
        # The number of pixels to crop on each side of the spectrum. List of two integers. Defaults to [10, 10]
        'crop_data_pix': <list>,
        
        # If the user only wishes to compute the BJDS and barycorrs then exit. Defaults to False.
        'compute_bc_only': <bool>,
        
        # Path of the default provided templates (tellurics, gas cell). Defaults to the pychell default_templates path.
        'default_templates_path': <str>,
        
        # Barycenter file (col1=BJD, col2=BC_VEL, comma delimited). Order must match flist_file.txt. Default is None, and is therefore generated via barycorrpy.
        'bary_corr_file': <str>,
        
        # The target function to optimize the model in the Nelder Mead call. Must live in target_functions.py Default is simple_rms.
        'target_function': <str>,
        
        # The number of bad pixels to flag in fitting. Default is 20.
        'flag_n_worst_pixels': <int>,
        
        # Function to augment the stellar (and/or) lab frame templates. Must live in template_augmenter.py Default is cubic_spline_lsq.
        'template_augmenter': <str>,

        # A list of which nights to consider in augmenting the template (first night = 1) Empty list = all nights. 'best' uses the night with the highest total S/N (lowest summation over RMS). Default is [] for all nights.
        'nights_for_template': <list, str>

        # A list of which templates to optimize. If empty, this parameter is ignored. Possible entries in the list are 'star' and/or 'lab'. If non-empty, the globalfit method is called to utilize PyTorch / ADAM to optimize the templates. Default is an empty list, so template_augmenter is called on all iterations. If starting from no stellar template, the template_augmenter function is still called the first time.
        'templates_to_optimize': <list>,
        
        # Number of iterations to update the stellar template. A zeroth iteration (flat template) does not count towards this number. Default is 5.
        'n_template_fits': <int>,
        
        # Cross correlation / bisector span options for each iteration.
        # A cross correlation will still be run to estimate the correct overall RV before fitting if starting from a synthetic template.
        # If method is None, then xcorr is not performed. Default is below.
        # Ex: {'method': 'weighted_brute_force', 'weights': [], 'n_bs': 1000, 'step': 50, 'range': 1E4},
        'xcorr_options': <dict>,
        
        # Whether or not to crudely remove the continuum from the data before any optimizing. Default is False.
        'remove_continuum': False,

        # Model Resolution (n_model_pixels = model_resolution * n_data_pixels)
        # This is only important because of the instrument line profile (LSF).
        # Default is 8.
        'model_resolution': <int>
    }
    

**forward_model_blueprints**

.. code-block:: python

    forward_model_blueprints = {

        # Recognized keys:
        'star', 'gas_cell', 'blaze', 'tellurics', 'wavelength_solution', 'lsf'

        # Additional models with any key (e.g. - 'fringing') may be defined.
    }

Each entry (sub-dictionary) in forward_model_blueprints defines the blueprints on how to construct the class for this model component, and will correspond to a class in model_components.py which extends the SpectralComponent Class. Each of these components will define several model-specific entries, but one must also define a few basic things. When a given model is constructed, it is given the appropriate corresponding sub-dictionary. An example of a typical stellar model is shown below.

Raw example:

.. code-block:: python

    'model_component': {
             
            # REQUIRED

            # The name of the model. May be anything.
            'name': <str>,

            # The corresponding class in model_components.py
            'class_name': <str>,

            # ADDITIONAL SETTINGS FOR ALL MODELS

            # The number of times to delay this model. Default is 0.
            'n_delay': <int>
    }

Specific example:

.. code-block:: python

    'star': {

        'name': 'star',
        'class_name': 'StarModel',

        # MODEL SPECIFIC SETTINGS

        # The full path to the input file, defaults to None to start from a flat stellar template.
        'input_file': None,

        # The single parameter for the star (Doppler velocity).
        'vel': [-1000 * 300, 10, 1000 * 300]
    }