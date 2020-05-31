Extending
*********

Defining New Instruments
========================

It is recommended a user define a new instrument if there is no current support. **First**, create a new file, *insname.py*, within the *spectrographs* folder. This file must contain several dictionaries. Some of these dictionaries overwrite settings in *config.py*. Every option found here may be overwritten at user run time in their config file, with similar dictionaries.

First we import some modules and define the default templates path which will be helpful for later.

.. code-block:: python

        # Basic imports
        import os
        import numpy as np
        import pychell.rvs


**general_settings** stores instrument defaults which will be used in one or both of the RVs and reduction code. Required and optional keys are listed. If one only wishes to implement one of reduction or RVs, then certain settings may be anything, however the recommended is ``NotImplemented``.

.. code-block:: python

        general_settings = {

            ## REQUIRED
            
            # The spectrograph name. Can be anything.
            'spectrograph': <str>,
            
            # The name of the observatory.
            # Must be a recognized astropy EarthLocation if not computing own barycenter info.
            'observatory': <str>,
            
            ## OPTIONAL, but highly recommended for readability.

            # The tags to recognize science, bias, dark, and flat field images
            'sci_tag': <str>,
            'bias_tag': <str>,
            'darks_tag': <str>,
            'flats_tag': <str>

            # The name of the filename parser function (see above).
            # This function must be defined in 
            'filename_parser': <str> or <function>,

            # Gain of primary detector
            # Reduction
            'gain': <float>,
            
            # Dark current of primary detector
            # Only required if performing dark subtraction, default=1.0
            'dark_current': <float>,
            
            # Read noise of the primary detector
            # Reduction, default=1
            'read_noise': <float>,
            
            # The orientation of the spectral axis for 2d images. 'x' for aligned with detector rows, 'y' for columns.
            # Reduction, default='x'
            'orientation': <str>,
            
            # increasing => left to right, decreasing => right to left
            # Reduction and RVs, default='increasing'
            'wave_direction': ,<str>,
            
            # The time offset used in the headers
            # Reduction, default=0
            'time_offset': <float>
        }

**header_keys** are only used in reduction as of now, since the data load method is truly instrument specific and must be implemented for every instrument. The keys of this dictionary are used throughout the code. The items they point to (lists), contain the actual header key found in the instrument header, item[0], and a default value if that key is not found, item[1]. A default header_keys instrument is provided in config.py and copied below, however it is highly recommended the user implement their own.

.. code-block:: python

    header_keys = {
        'target': ['TCS_OBJ', 'NA'],
        'RA': ['TCS_RA', '00:00:00'],
        'DEC': ['TCS_DEC', '00:00:00'],
        'slit': ['SLIT', 'NA'],
        'wavelength_range': ['WAVELENGTH', 'NA'],
        'gas_cell': ['GASCELL', 'NA'],
        'exp_time': ['ITIME', 0],
        'time_of_obs': ['TIME', 2457000],
        'NDR': ['NDR', 1],
        'BZERO': ['BZERO', 0],
        'BSCALE': ['BSCALE', 1]
    }

**calibration_settings** contains the default calibration settings for this instrument. As of now, only the usual bias, dark, flat corrections are implemented. The default is False for all settings.


.. code-block:: python

        calibration_settings = {
            'dark_subtraction': <bool>,
            'flat_division': <bool>,
            'bias_subtraction': <bool>,
            'wavelength_calibration': <bool>
        }


**extraction_settings** contains the settings used in the actual optimal extraction process.

.. code-block:: python

        extraction_settings = {
            
            # Order map algorithm (options: 'from_flats, 'empirical'), default='empirical'
            'order_map': <str>,
            
            # Pixels to mask on the top, bottom, left, and right edges, default=20 for all.
            'mask_left_edge': <int>,
            'mask_right_edge': <int>,
            'mask_top_edge': <int>,
            'mask_bottom_edge': <int>,
            
            # The height of an order is defined as where the flat is located.
            # This masks additional pixels on each side of the initial trace profile before moving forward.
            # The profile is further masked after the sky background is estimated and data extracted.
            # Default=3
            'mask_trace_edges':  <int>,
            
            # The degree of the polynomial to fit the individual order locations, default=2
            'trace_pos_polyorder' : <int>,
            
            # Whether or not to perform a sky subtraction, default=True
            # The number of rows used to estimate the sky background (lowest n_sky_rows in the trace profile are used), default=8
            'sky_subtraction': <bool>,
            'n_sky_rows': <int>,
            
            # The trace profile is constructed using oversampled data via interpolation with cubic splines.
            # Use 1 to not oversample.
            # This is the oversample factor, default=16
            'oversample': <int>
        }

**forward_model_settings** contains basic information for forward modeling the spectra to generate RVs. Nothing is required here.


.. code-block:: python

**model_blueprints** *coming soon...*