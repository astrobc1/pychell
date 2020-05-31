.. _quickstart:

Quickstart
**********

pychell may be run in a modular sort of way, where the user executes each sub-command to run the analyses, or from a python script. The follwoing tutorials execute pre-generated scripts from the command line. To see more on running the code from the command line, see the API.

Reduction
=========

Below is a quick-start guide which reduces a single spectrum of Vega using raw spectra from the iSHELL spectrograph in Kgas mode (~2.3 microns). Flat division is performed. Neither dark or bias subtraction are performed. Copy the example folder Vega_reduction to a new location of your choice. Open a terminal window in this new location. Run:

``python vega_ishell_reduction_example.py``

The code will print helpful messages as it processes the single image. An output directory will be created in an output folder ``reduce/Vega``. Each sub folder contains the following:

1. **calib** - Master calibration images. Darks are named according to exposure times. Flats are named according to the individual image numbers it was created from.
2. **previews** - Images of the full frame reduced spectra (all orders).
3. **spectra** - Reduced and extracted 1-dimensional spectra stored in .fits files. shape=(n_orders, n_pixels, 3). The last index is for flux, flux unc, and bad pixels (1=good, 0=bad). The flux and unc are in units of photoelectrons.
4. **trace_profiles** - Trace profiles (seeing profiles) for all full frame science images.

Radial Velocities
=================

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