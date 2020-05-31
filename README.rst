=======
pychell
=======

The tldr;
=========

A environment to:

1. Reduce single trace multi-order echelle spectra via optimal extraction.
2. Generate radial velocities for stellar sources via forward modeling reduced 1-dimensional spectra.

Install with ``pip install .`` from the head directory.

Reduction
=========

As of now, reduction can be performed on well-behaved spectrographs with a single trace per echelle order.  Orders are traced with a density clustering algorithm (sklearn.cluster.DBSCAN). Flats are grouped together according to their "density" (metric ~ angular separation + time separation). Objects are mapped to a master flat according to the closest flat in space and time.

Calibration
+++++++++++

Flat, bias, and dark calibration is performed when provided. Wavelength calibartion via ThAr lamps or LFC's are not currently provided.

Extraction
++++++++++

The trace profile (seeing profile) is estimated by rectifying the order and taking a median crunch in the spectral direction on a high resolution grid. The background sky, *sky(x)* is computed by considering regions of low flux within the trace profile. The profile is then interpolated back into 2d space according to the order locations, *y(x)*. An optimal extraction is iteratively performed on the non-rectified data. Depending on the nature of the user's work, this may not be suitable and one should rely on using an instrument specific reduction package.

Tested Support Status:

1. iSHELL (Kgas, K2, J2 modes, empirical and flat field order tracing)
2. CHIRON (highres mode, R~136k, *under development*)
3. NIRSPEC (K band, *under development*)
4. Generic (single trace per order, minimal support)

Radial Velocities
=================

Computes radial velocities from reduced echelle spectra by forward modeling the individual orders. Only certain instruments are supported, however adding support for a new instrument is relatively straightforward (see below).

Tested Support Status:

1. iSHELL (Kgas mode, methane gas cell)
2. CHIRON (highres mode, R~136k, iodine gas cell)
3. Minerva-Australis (ThAr Lamp calibrated, soon iodine gas cell)
4. NIRSPEC (K band, *under development*)


Quick Tutorials
=========

Reduction
+++++++++

Below is a quick-start guide which reduces a single spectrum of Vega using raw spectra from the iSHELL spectrograph in Kgas mode (~2.3 microns). Flat division is performed. Neither dark or bias subtraction are performed. Copy the example folder Vega_reduction to a new location of your choice. Open a terminal window in this new location. Run:

``python vega_ishell_reduction_example.py``

The code will print helpful messages as it processes the single image. An output directory will be created in an output folder ``reduce/Vega``. Each sub folder contains the following:

1. **calib** - Master calibration images. Darks are named according to exposure times. Flats are named according to the individual image numbers it was created from.
2. **previews** - Images of the full frame reduced spectra (all orders).
3. **spectra** - Reduced and extracted 1-dimensional spectra stored in .fits files. shape=(n_orders, n_pixels, 3). The last index is for flux, flux unc, and bad pixels (1=good, 0=bad). The flux and unc are in units of photoelectrons.
4. **trace_profiles** - Trace profiles (seeing profiles) for all full frame science images.

Radial Velocities
+++++++++++++++++

Below is a quick-start guide which fits 4 nights (8 spectra) of Barnard's Star spectra using reduced spectra from the iSHELL spectrograph. Only orders 11, 13, and 15 are fit. The wavelength solution and LSF are constrained from the isotopic methane gas cell, which is provided in the default_templates folder.

Copy the example folder GJ_699_rvs to a new location of your choice. Open a terminal window in this new location. Run:

``python gj699_ishell_rvs_example.py``

The code will immediately provide a summary of the run. Summaries of fits are printed after each fit. An output directory will also be created in an output folder ``GJ_699_default_test_run``. This folder contains the global parameters dictionary (stored in an .npz file) used throughout the code and sub folders for each order. Each sub folder contains the following:

1. **Fits** - Contains the forward model plots for each iteration. The spectral numbers are in chronological order. Also contains .npz files with the following keys:
    - *wave* : The wavelegnth solutions for this spectrum, np.ndarray, shape=(n_data_pix, n_template_fits)
    - *models* : The best fit constructed forward models for this spectrum, np.ndarray, shape=(n_data_pix, n_template_fits)
    - *residuals* : The residuals for this spectrum (data - model), np.ndarray, shape=(n_data_pix, n_template_fits)
    - *data* : The data for this observation, np.ndarray, shape=(n_data_pix, 3), col1 is flux, col2 is flux_unc, col3 is badpix

    Lastly, this folder contains pickled forward model objects that can be loaded in with full access to the forward models builds. The above and below arrays are also stored here as instance variables.

2. **Opt** - Contains the optimization results from the Nelder-Mead fitting stored in .npz files. This must be loaded with allow_pickle=True. Keys are:
    - *best_fit_pars* : The best fit parameters, np.ndarray, shape=(n_template_fits,). Each entry is a Parameters object.
    - *opt* : np.ndarray, shape=(n_template_fits, 2). col1=final RMS returned by the solver. col2=total target function calls.

3. **RVs** - Contains the RVs for this order. Fits for each iteration are in .png files. The RVs are stored in the per iteration .npz files with the following keys:
    - *rvs* : The best fit RVs, np.ndarray, shape=(n_spec, n_template_fits)
    - *rvs_nightly* : The co-added ("nightly") RVs, np.ndarray, shape=(n_nights, n_template_fits)
    - *rvs_unc_nightly* : The corresponding 1 sigma error bars for the nightly RVs, shape=(n_nights, n_template_fits)
    - *BJDS* : The bary-centric Julian dates which correspond to the single RVs.
    - *BJDS_nightly* : The nightly BJDS which correspond to the nightly RVs.
    - *n_obs_nights* : The number of observation observed on each night, np.ndarray, shape=(n_nights,)
    - *rvs_xcorr* : The cross-correlation RVs if do_xcorr=True, np.ndarray, shape=(n_spec, n_template_fits)
    - *rvs_xcorr_nightly* : The co-added ("nightly") cross-correlation RVs if do_xcorr=True, np.ndarray, shape=(n_nights, n_template_fits)
    - *rvs_xcorr_unc_nightly* : The corresponding 1 sigma error bars for the nightly cross-correlation RVs, shape=(n_nights, n_template_fits)
    - *xcorr_vels* : The cross correlation velocity grid, shape=(n_vels, n_template_fits)
    - *xcorrs* : The corresponding cross correlations, shape=(n_vels, n_template_fits)
    - *line_bisectors* : The line bisectors as a function of ccf depth, shape=(n_bs, n_template_fits)
    - *bisector_spans* : The corresponding line bisectors (BIS), shape=(n_template_fits,)

4. **Stellar_Templates** - Contains the stellar template over iterations. Contains a single .npz file with key:
    - *stellar_templates* : The stellar template used in each iteration, np.ndarray, shape=(n_model_pix, n_template_fits+1). col1 is wavelength, remaining cols are flux.


Custom Runs
===========

Reduction
+++++++++

All input data must be stored in .fits or .fz files and be the primary HDU for now. Other HDUs are ignored. For a given night (or partial-night), the data must be stored in a single directory, *input_dir*.

A .py run file is created containing the following dictionaries and keys:

general_settings
++++++++++++++++

**REQUIRED**

- *input_dir* : The input directory (str)
- *output_dir* : The output directory. Must live elsewhere since it has the same name as the input directory. (str)
- *instrument* : The instrument used. If not directly supported, try "generic" (str)

**OPTIONAL**

- *n_cores* : The number of computing cores used. (int) Default=1


extraction_settings
+++++++++++++++++++

See instrument files for defaults.

**OPTIONAL**

- *mask_left_edge*: Masks the left edge of the frame (int)
- *mask_right_edge*: Masks the right edge of the frame (int)
- *mask_top_edge* : Masks the top edge of the frame (int)
- *mask_bottom_edge* : Masks the bottom edge of the frame (int)
- *order_map* : Order tracing algorithm. Options are: 'from_flats' to use flat fields (flat_division must be True), or 'empirical' to determine it from the data. The trace is further refined via cross correlation. (str)

calib_settings
++++++++++++++

See isntrument files for defaults.

**OPTIONAL**

- *bias_subtraction* : Whether or not to perform bias subtraction. (bool)
- *dark_subtraction* : Whether or not to perform dark subtraction. (bool)
- *flat_division* : Whether or not to perform flat division. (bool)

header_keys (*OPTIONAL*)
++++++++++++++++++++++++

If using a generic instrument, this must be provided to extract header info.

To kick things off, include ``import import pychell.reduce.driver`` and run:

``pychell.reduce.driver.reduce_night(general_settings, extraction_settings, calib_settings, header_keys=None)``

Radial Velocities
+++++++++++++++++

For all supported instruments, each full frame image (all orders) must be stored in a single fits file. This file contains header information (including time info to compute the exposure midpoint). The data is formatted as a single array with shape=(n_orders, n_data_pix, K), where K>=1 is an integer specific to the data. Specific information below:

1. iSHELL - K=3; flux, flux unc, badpix (1=good, 0=bad). **As of now, wavelength is assumed decreasing and is internally flipped, this will be changed in a future update.**
2. CHIRON - K=2; wave, flux. The wavelength grid is obtained from the ThAr lamp and is further constrained with iodine cell.
3. PARVI - K=2; wave, flux. The wavelength grid is obtained from the ThAr lamp and is further constrained with iodine cell.

User Config File
++++++++++++++++

For a given run, a config file with the following dictionaries and entries must be created. They could be of any name, but the following are recommended for readability.

forward_model_settings
++++++++++++++++++++++


**REQUIRED**

- *instrument* : The spectrograph the data was taken with. Must be in the supported instruments - iSHELL, PARVI, CHIRON, NIRSPEC. (str).
- *data_input_path* : The data input path. All spectra must be stored in this single directory. (str)
- *filelist* : The text file containing the files (one per line) to be used in this run. This file must be stored in data_input_path. Order is not important (str).
- *output_path* : The output path to store the run in. A single directory is created per run. (str).
- *star_name* : The name of the star. If fetching bary-center info from barycorrpy, it must be searchable on simbad with this entry. FOr spaces, use an underscore. (str)
- *tag* : A tag to uniquely identfiy this run. The main level path for this run will be called star_name + tag. All files will include star_name + tag.
- *do_orders* : Which echelle orders to do. e.g., np.arange(1, 30).astype(int) would do all 29 iSHELL orders. Or a list of orders [4, 5, 6] will only fit orders 4-6. Orders are fit in numerical order, not the order they are provided. (iterable)

**OPTIONAL**

- *bary_corr_file* : A csv file in data_input_path containing the bary-center info. col1=BJDS, col2=bc_vels. The order must be consistent with the order provided in filelist. Lines that begin with '#' are ignored.  (str), DEFAULT: None, and bc info is calculated with barycorrpy.
- *n_cores* : The number of cores used in the Nelder-Mead fitting and possible cross corr analysis. (int). DEFAULT: 1
- *verbose_plot* : Whether or not to add templates to the plots. (bool) DEFAULT: False
- *verbose_print* : Whether or not to print the optimization results after each fit. (bool) DEFAULT: False
- *nights_for_template* : Which nights to include when updating the stellar template. e.g., [1,2] will only use the first and second nights. Use an empty list to use all nights. Or use 'best' to use the night with the highest co-added S/N. (list or str). DEFAULT: [] (empty list) for all nights.
- *n_template_fits* : The number of times a real stellar template is fit to the data. DEFAULT: 10
- *model_resolution* : The resolution of the model. It's important this is greater than 1 to ensure the convolution with the LSF is accurate. n_model_pix = n_data_pix * model_resolution. (int) DEFAULT: 8
- *do_xcorr* : Whether or not a cross correlation analysis is performed after the fit. This takes time, but provides the bisector span of the ccf function which can be useful. If True, additional keys are added to the RV output files (see above). (bool).  DEFAULT: False
- *flag_n_worst_pixels* : The number of worst pixels to flag in the forward model (after weights are applied) (int). DEFAULT: 20
- *plot_wave_unit* : The wavelength units in plots (str). Option are 'nm', 'ang', 'microns'. DEFAULT: 'nm'
- *compute_bc_only* : If True, the bary-center information is computed via barycorrpy and written to a file in the provided output directory.
- *crop_pix* : The number of pixels cropped on the ends each order; [crop_from_left, crop_from_right]. If the bad pix array provided with the data allows for a wider window, the window is still cropped according to this entry. If the bad pix array is smaller, the entry is irrelevant. (list). DEFAULT: [10, 10]
- *target_function* : The optimization function that minimizes some helpful quantity to fit the spectra. See ``pychell_target_functions.py`` for more info. (str)
- *template_augmenter* : The function to augment the stellar template after Nelder Mead fitting. See ``pychell_target_functions.py`` for more info. (str)
- Any other key found in the instrument forward_model_settings dictionary or ``config.py``.


model_blueprints
++++++++++++++++

Each instrument defines its own default ``model_blueprints`` dictionary, stored in ``pychell/spectrographs/parameters_insname.py.`` This dictionary contains the blueprints to construct the forward model. A few keys in this dictionary are special. It must contain a *star* and *wavelength_solution* as keys, which are already provided in the default settings and don't need to be provided in the user config file, unless the user wishes to override settings. The iems are then dictionaries which contains helpful info to construct that model component. Each model component must be tied to a class which implements/extends the SpectralComponent abstract class in ``pychell/rvs/model_components.py.`` For a given run, the user may wish to overwrite some of these defaults. This is done through defining the user_model_blueprints dictionary in their run file. From here, the user can add new model components by adding new keys, or updating existing ones by redefining an existing key. Three cases exist:

1. Key is common to both dictionaries - The item will only be updated according to the user's sub keys. Existing sub keys remain with their default values.
2. Key exists only in the user blueprints but not the default - The new model is added and must contain all information necessary (see below on defnining new models).
3. Key exists only in the default blueprints - Default settings are used.

Example of overriding blueprints model to start from a synthetic stellar template. The default setting was ``None`` - to start from a flat stellar template. This will now start things from a real template.

.. code-block:: python

    'star' : {
        'input_file' : '/path/to/input_file/'
    }
 

There are a few special keys required for each entry in this dictionary (see defining new models below). The format of each sub dictionary can be anything that the model supports. So, to know how to override settings for other mode components, one must look at the default model information to see what is available. To kick things off, include ``import import pychell.rvs.driver`` and run:

``pychell.rvs.driver.fit_target(forward_model_settings, model_blueprints)``


Templates
+++++++++

Custom (synthetic or empirical) templates may be used. Templates must be stored in csv files with col1=wave (increasing, in Angstroms), col2=flux (normalized). Comments are assumed to start with ``#``. Templates are always cropped to the order (with small padding). Stellar templates specifically are padded to account for the bary-center velocity bias if using the StarModel class. Templates for custom can be loaded in by implementing a function ``load_template(self, gpars)`` in the relevant SpectralModel class.


Support for New Instruments
===========================

Each instrument utilizes a special file located in ``pychell/spectrographs/insname.py``. For now, look to the other instruments to see specifically which keys must be defined. Each entry is commented. If one only wishes to implement rvs or reduction, then only the relevant dictionaries need be defined.

1. general_settings - For both Reduction and RVs, contains information that will likely never change.
2. calibration_settings - Reduction, default calibatrion settings (bias, dark, flat)
3. header_keys - Reduction, maps header keys to keys used in the code
4. extraction_settings - Reduction, pixels to mask, sky subtraction.
5. forward_model_settings - RVs, forward model settings.
6. forward_model_blueprints - RVs, how to construct the forward model dictionary.

Reduction
+++++++++

*Coming Soon*

Radial Velocities
+++++++++++++++++

To implement a new instrument for RVs, the following must be defined:

**1. In the  .py file located in pychell/spectrographs/ with a unique name, insname.py.**

The forward model settings should define:
- *spectrograph* : Mandatory, should match (not case-sensitive) the name of this file. (str)
- *observatory* : The name of the observatory, must be recognized by astropy EarthLocations (str)
- *crop_pix* : The number of detector pixels to crop on each side of the 1d spectrum. (list; [int, int], e.g. [10, 10])
- *plot_wave_unit* : The wavelength units. Options are 'nm', 'ang', 'microns'  (str)

**2. A class in pychell/rvs/data1d.py with the name SpecData[insname] (no brackets) which extends the SpecData1d class**

The super class will store the *order_num*; 0...n_orders-1 (int), *spec_num*; 0...n_spec-1 (int), and *input_file* (str) which is the full path to the file.

This class must define a parse method where the data for a specific order is read in. The instance members *flux*, *flux_unc*, and *badpix* must be defined. If they are not provided by the input file, defaults can be created. The syntax is:

.. code-block:: python

    # Function signature
    # wave_direction is a setting available in forward_model_settings
    # No value is returned
    def parse(self, wave_direction='increasing'):
        
        # Example of loading in the data for this observation
        fits_data = fits.open(self.input_file)
        # Store flux, flux_unc, badpix in appropriate instance variables.
        self.flux = ...
        ...

        # If spectra are not normalized, one may use:
        continuum = pcmath.weighted_median(self.flux, med_val=0.99)
        self.flux /= continuum
        self.flux_unc /= continuum
        ...

New Models
++++++++++

*Coming Soon*