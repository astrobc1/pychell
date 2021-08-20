.. _quickstart:

Tutorials
*********

Reduction
=========

Below is an example which reduces a single spectrum of Vega using raw spectra from the iSHELL spectrograph in Kgas mode (~2.3 microns). Flat division is performed, but not dark or bias subtraction. Copy the example folder Vega_reduction to a new location of your choice. Open a terminal window in this new location. Run:

``python vega_ishell_reduction_example.py``

An output directory will be created in an output folder ``Vega``. Each sub folder contains the following:

1. **calib** - Any master calibration images.
3. **spectra** - Reduced and extracted 1-dimensional spectra stored in .fits files. shape=(n_orders, n_traces, n_pixels, 3). The last index is for flux, flux unc, and bad pixels (1=good, 0=bad). The flux and unc are in units of photoelectrons, but may still contain blaze modulation.
4. **trace** - Trace profiles (seeing profiles) and order map information for all relevant full frame science images.


Radial Velocitiy Generation
===========================

Below is an example which fits 4 nights (8 spectra) of Barnard's Star spectra using reduced spectra from iSHELL. Only orders 6, 8, 12, and 15 are fit. The wavelength solution and LSF are constrained from the isotopic methane gas cell.

Copy the example folder GJ_699_rvs to a new location of your choice. Open a terminal window in this new location. Run:

``python gj699_ishell_rvs_example.py``

Summaries of fits are printed after each fit. An output directory will also be created in an output folder ``ishell_gj699_example``, with each order contained in its own folder. Each order contains the following sub folders:

1. **ForwardModels** - The spectral model fit for each observation and iteration as PNG plots.
2. **RVs** - Plots of the individual and per-night (co-added) RVs for each iteration, plots of BIS vs. RVs for each iteration, and the RVs stored in a ``.npz`` file.
3. **Templates** - Not currently used.

A ``.pkl`` file of the SpectralRVProb instance is also saved which may be loaded in order to generate wavelength solutions, look at best fit parameters, etc.


Radial Velocity Fitting
=======================

