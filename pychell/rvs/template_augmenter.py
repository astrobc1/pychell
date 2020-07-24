# Python built in modules
import copy
import glob # File searching
import os # Making directories
import importlib.util # importing other modules from files
import warnings # ignore warnings
import sys # sys utils
import pickle
import warnings
from sys import platform # plotting backend
from pdb import set_trace as stop # debugging

# Graphics
import matplotlib # to set the backend
import matplotlib.pyplot as plt # Plotting
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Science/math
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np # Math, Arrays
try:
    import torch
except:
    warnings.warn("Could not import pytorch!")
import scipy.interpolate # Cubic interpolation, Akima interpolation

# llvm
from numba import njit, jit, prange

# Pychell modules
import pychell.config as pcconfig
import pychell.maths as pcmath # mathy equations
import pychell.rvs.forward_models as pcforwardmodels # the various forward model implementations
import pychell.rvs.data1d as pcdata # the data objects
import pychell.rvs.model_components as pcmodelcomponents # the data objects
import pychell.rvs.target_functions as pctargetfuns
import pychell.utils as pcutils
import pychell.rvs.rvcalc as pcrvcalc


def cubic_spline_lsq(forward_models, iter_index=None, nights_for_template=None, templates_to_optimize=None):
    """Augments the stellar template by fitting the residuals with cubic spline least squares regression. The knot-points are spaced roughly according to the detector grid. The weighting scheme includes (possible inversly) the rms of the fit, the amount of telluric absorption. Weights are also applied such that the barycenter sampling is approximately uniform from vmin to vmax.

    Args:
        forward_models (ForwardModels): The list of forward model objects
        iter_index (int): The iteration to use.
        nights_for_template (str or list): The nights to consider for averaging residuals to update the stellar template. Options are 'best' to use the night with the highest co-added S/N, a list of indices for specific nights, or an empty list to use all nights. defaults to [] for all nights.
    """
    if nights_for_template is None:
        nights_for_template = forward_models.nights_for_template

    current_stellar_template = np.copy(forward_models.templates_dict['star'])
    
    # Storage Arrays for the low res grid
    # This is for the low res reiduals where the star is constructed via a least squares cubic spline.
    # Before the residuals are added, they are normalized.
    waves_shifted_lr = np.empty(shape=(forward_models[0].data.flux.size, forward_models.n_spec), dtype=np.float64)
    residuals_lr = np.empty(shape=(forward_models[0].data.flux.size, forward_models.n_spec), dtype=np.float64)
    tot_weights_lr = np.empty(shape=(forward_models[0].data.flux.size, forward_models.n_spec), dtype=np.float64)
    
    # Weight by 1 / rms^2
    rms = np.array([forward_models[ispec].opt[-1][0] for ispec in range(forward_models.n_spec)], dtype=float)
    rms_weights = 1 / rms**2
    good = np.where(np.isfinite(rms_weights))[0]
    bad = np.where(~np.isfinite(rms_weights))[0]
    if good.size == 0:
        rms_weights = np.ones(forward_models.n_spec)

    # All nights
    if nights_for_template is None or len(nights_for_template) == 0:
        template_spec_indices = np.arange(forward_models.n_spec).astype(int)
    # Night with highest co-added S/N
    if nights_for_template == 'best':
        night_index = determine_best_night(rms, forward_models.n_obs_nights)
        template_spec_indices = list(forward_models.get_all_spec_indices_from_night(night_index, forward_models.n_obs_nights))
    # User specified nights
    else:
        template_spec_indices = []
        for night in nights_for_template:
            template_spec_indices += list(forward_models.get_all_spec_indices_from_night(night - 1, forward_models.n_obs_nights))
            
    # Loop over spectra
    for ispec in range(forward_models.n_spec):

        # De-shift residual wavelength scale according to the barycenter correction
        # Or best doppler shift if using a non flat initial template
        if forward_models[0].models_dict['star'].from_synthetic:
            waves_shifted_lr[:, ispec] = forward_models[ispec].wavelength_solutions[-1] * np.exp(-1 * forward_models[ispec].best_fit_pars[-1][forward_models[ispec].models_dict['star'].par_names[0]].value / cs.c)
        else:
            waves_shifted_lr[:, ispec] = forward_models[ispec].wavelength_solutions[-1] * np.exp(forward_models[ispec].data.bc_vel / cs.c)
            
        residuals_lr[:, ispec] = np.copy(forward_models[ispec].residuals[-1])

        # Telluric weights
        tell_flux_hr = forward_models[ispec].models_dict['tellurics'].build(forward_models[ispec].best_fit_pars[-1], forward_models.templates_dict['tellurics'], current_stellar_template[:, 0])
        tell_flux_hr_convolved = forward_models[ispec].models_dict['lsf'].convolve_flux(tell_flux_hr, pars=forward_models[ispec].best_fit_pars[-1])
        tell_flux_lr_convolved = np.interp(forward_models[ispec].wavelength_solutions[-1], current_stellar_template[:, 0], tell_flux_hr_convolved, left=np.nan, right=np.nan)
        tell_weights = tell_flux_lr_convolved**2
        
        tot_weights_lr[:, ispec] = forward_models[ispec].data.badpix * rms_weights[ispec]
        
        # Final weights
        if len(nights_for_template) != 1:
            tot_weights_lr[:, ispec] = tot_weights_lr[:, ispec] # * tell_weights
            
        
    # Generate the histogram
    bc_vels = np.array([forward_models[ispec].data.bc_vel for ispec in range(forward_models.n_spec)], dtype=float)
    hist_counts, histx = np.histogram(bc_vels, bins=int(np.min([forward_models.n_spec, 10])), range=(np.min(bc_vels)-1, np.max(bc_vels)+1))
    
    # Check where we have no spectra (no observations in this bin)
    hist_counts = hist_counts.astype(np.float64)
    bad = np.where(hist_counts == 0)[0]
    if bad.size > 0:
        hist_counts[bad] = np.nan
    number_weights = 1 / hist_counts
    number_weights = number_weights / np.nansum(number_weights)

    # Loop over spectra and also weight spectra according to the barycenter sampling
    # Here we explicitly use a multiplicative combination of weights.
    if len(nights_for_template) == forward_models.n_nights:
        for ispec in range(forward_models.n_spec):
            vbc = forward_models[ispec].data.bc_vel
            y = np.where(histx >= vbc)[0][0] - 1
            tot_weights_lr[:, ispec] = tot_weights_lr[:, ispec] * number_weights[y]
            
    # Now to co-add residuals according to a least squares cubic spline
    # Flatten the arrays
    waves_shifted_lr_flat = waves_shifted_lr.flatten()
    residuals_lr_flat = residuals_lr.flatten()
    tot_weights_lr_flat = tot_weights_lr.flatten()
    
    # Remove all bad pixels.
    good = np.where(np.isfinite(waves_shifted_lr_flat) & np.isfinite(residuals_lr_flat) & (tot_weights_lr_flat > 0))[0]
    waves_shifted_lr_flat, residuals_lr_flat, tot_weights_lr_flat = waves_shifted_lr_flat[good], residuals_lr_flat[good], tot_weights_lr_flat[good]

    # Sort the wavelengths
    sorted_inds = np.argsort(waves_shifted_lr_flat)
    waves_shifted_lr_flat, residuals_lr_flat, tot_weights_lr_flat = waves_shifted_lr_flat[sorted_inds], residuals_lr_flat[sorted_inds], tot_weights_lr_flat[sorted_inds]
    
    # Knot points are roughly the detector grid.
    knots_init = np.linspace(waves_shifted_lr_flat[0]+0.01, waves_shifted_lr_flat[-1]-0.01, num=forward_models[0].data.flux.size)
    bad_knots = []
    for iknot in range(len(knots_init) - 1):
        n = np.where((waves_shifted_lr_flat > knots_init[iknot]) & (waves_shifted_lr_flat < knots_init[iknot+1]))[0].size
        if n == 0:
            bad_knots.append(iknot)
    bad_knots = np.array(bad_knots)
    knots = np.delete(knots_init, bad_knots)
    
    # Do the fit
    tot_weights_lr_flat /= np.nansum(tot_weights_lr_flat)
    spline_fitter = scipy.interpolate.LSQUnivariateSpline(waves_shifted_lr_flat, residuals_lr_flat, t=knots[1:-1], w=tot_weights_lr_flat, k=3, ext=1, bbox=[waves_shifted_lr_flat[0], waves_shifted_lr_flat[-1]], check_finite=True)
    
    # Use the fit to determine the hr residuals to add
    residuals_hr_fit = spline_fitter(current_stellar_template[:, 0])

    # Remove bad regions
    bad = np.where((current_stellar_template[:, 0] <= knots[0]) | (current_stellar_template[:, 0] >= knots[-1]))[0]
    if bad.size > 0:
        residuals_hr_fit[bad] = 0

    # Augment the template
    new_flux = current_stellar_template[:, 1] + residuals_hr_fit
    
    locs = np.where(new_flux > 1)[0]
    if locs.size > 0:
        new_flux[locs] = 1

    forward_models.templates_dict['star'][:, 1] = new_flux
    
    
def cubic_spline_lsq_nobcweights(forward_models, iter_index, nights_for_template=None, templates_to_optimize=None):
    """Augments the stellar template by fitting the residuals with cubic spline least squares regression. The knot-points are spaced roughly according to the detector grid. This function is identical to cubic_spline_lsq but does not include barycenter weighting.

    Args:
        forward_models (ForwardModels): The list of forward model objects
        iter_index (int): The iteration to use.
        nights_for_template (str or list): The nights to consider for averaging residuals to update the stellar template. Options are 'best' to use the night with the highest co-added S/N, a list of indices for specific nights, or an empty list to use all nights. defaults to [] for all nights.
    """

    current_stellar_template = np.copy(forward_models.templates_dict['star'])
    
    # Storage Arrays for the low res grid
    # This is for the low res reiduals where the star is constructed via a least squares cubic spline.
    # Before the residuals are added, they are normalized.
    waves_shifted_lr = np.empty(shape=(forward_models[0].data.flux.size, forward_models.n_spec), dtype=np.float64)
    residuals_lr = np.empty(shape=(forward_models[0].data.flux.size, forward_models.n_spec), dtype=np.float64)
    tot_weights_lr = np.empty(shape=(forward_models[0].data.flux.size, forward_models.n_spec), dtype=np.float64)
    
    # Weight by 1 / rms^2
    rms = np.array([forward_models[ispec].opt[-1][0] for ispec in range(forward_models.n_spec)])
    rms_weights = 1 / rms**2
    good = np.where(np.isfinite(rms_weights))[0]
    bad = np.where(~np.isfinite(rms_weights))[0]
    if good.size == 0:
        rms_weights = np.ones(forward_models.n_spec)

    # All nights
    if nights_for_template is None or len(nights_for_template) == 0:
        template_spec_indices = np.arange(forward_models.n_spec).astype(int)
    # Night with highest co-added S/N
    if nights_for_template == 'best':
        night_index = determine_best_night(rms, forward_models.n_obs_nights)
        template_spec_indices = list(forward_models.get_all_spec_indices_from_night(night_index, forward_models.n_obs_nights))
    # User specified nights
    else:
        template_spec_indices = []
        for night in nights_for_template:
            template_spec_indices += list(forward_models.get_all_spec_indices_from_night(night - 1, forward_models.n_obs_nights))
            
    # Loop over spectra
    for ispec in range(forward_models.n_spec):

        # De-shift residual wavelength scale according to the barycenter correction
        # Or best doppler shift if using a non flat initial template
        if forward_models[0].models_dict['star'].from_synthetic:
            waves_shifted_lr[:, ispec] = forward_models[ispec].wavelength_solutions[-1] * np.exp(-1 * forward_models[ispec].best_fit_pars[-1][forward_models[ispec].models_dict['star'].par_names[0]].value / cs.c)
        else:
            waves_shifted_lr[:, ispec] = forward_models[ispec].wavelength_solutions[-1] * np.exp(forward_models[ispec].data.bc_vel / cs.c)
            
        residuals_lr[:, ispec] = np.copy(forward_models[ispec].residuals[-1])
        

        # Telluric weights
        tell_flux_hr = forward_models[ispec].models_dict['tellurics'].build(forward_models[ispec].best_fit_pars[-1], forward_models.templates_dict['tellurics'], current_stellar_template[:, 0])
        tell_flux_hr_convolved = forward_models[ispec].models_dict['lsf'].convolve_flux(tell_flux_hr, pars=forward_models[ispec].best_fit_pars[-1])
        tell_flux_lr_convolved = np.interp(forward_models[ispec].wavelength_solutions[-1], current_stellar_template[:, 0], tell_flux_hr_convolved, left=np.nan, right=np.nan)
        tell_weights = tell_flux_lr_convolved**2
        
        tot_weights_lr[:, ispec] = forward_models[ispec].data.badpix * rms_weights[ispec]
        
        # Final weights
        if len(nights_for_template) != 1:
            tot_weights_lr[:, ispec] = tot_weights_lr[:, ispec] * tell_weights
            
    # Now to co-add residuals according to a least squares cubic spline
    # Flatten the arrays
    waves_shifted_lr_flat = waves_shifted_lr.flatten()
    residuals_lr_flat = residuals_lr.flatten()
    tot_weights_lr_flat = tot_weights_lr.flatten()
    
    # Remove all bad pixels.
    good = np.where(np.isfinite(waves_shifted_lr_flat) & np.isfinite(residuals_lr_flat) & (tot_weights_lr_flat > 0))[0]
    waves_shifted_lr_flat, residuals_lr_flat, tot_weights_lr_flat = waves_shifted_lr_flat[good], residuals_lr_flat[good], tot_weights_lr_flat[good]

    # Sort the wavelengths
    sorted_inds = np.argsort(waves_shifted_lr_flat)
    waves_shifted_lr_flat, residuals_lr_flat, tot_weights_lr_flat = waves_shifted_lr_flat[sorted_inds], residuals_lr_flat[sorted_inds], tot_weights_lr_flat[sorted_inds]
    
    # Knot points are roughly the detector grid.
    knots_init = np.linspace(waves_shifted_lr_flat[0]+0.01, waves_shifted_lr_flat[-1]-0.01, num=forward_models[0].data.flux.size)
    bad_knots = []
    for iknot in range(len(knots_init) - 1):
        n = np.where((waves_shifted_lr_flat > knots_init[iknot]) & (waves_shifted_lr_flat < knots_init[iknot+1]))[0].size
        if n == 0:
            bad_knots.append(iknot)
    bad_knots = np.array(bad_knots)
    knots = np.delete(knots_init, bad_knots)
    

    # Do the fit
    tot_weights_lr_flat /= np.nansum(tot_weights_lr_flat)
    spline_fitter = scipy.interpolate.LSQUnivariateSpline(waves_shifted_lr_flat, residuals_lr_flat, t=knots[1:-1], w=tot_weights_lr_flat, k=3, ext=1, bbox=[waves_shifted_lr_flat[0], waves_shifted_lr_flat[-1]], check_finite=True)
    
    # Use the fit to determine the hr residuals to add
    residuals_hr_fit = spline_fitter(current_stellar_template[:, 0])

    # Remove bad regions
    bad = np.where((current_stellar_template[:, 0] <= knots[0]) | (current_stellar_template[:, 0] >= knots[-1]))[0]
    if bad.size > 0:
        residuals_hr_fit[bad] = 0

    # Augment the template
    new_flux = current_stellar_template[:, 1] + residuals_hr_fit
    
    locs = np.where(new_flux > 1)[0]
    if locs.size > 0:
        new_flux[locs] = 1

    forward_models.templates_dict['star'][:, 1] = new_flux


def weighted_median(forward_models, iter_index=None, nights_for_template=None, templates_to_optimize=None):
    """Augments the stellar template by considering the weighted median of the residuals on a common high resolution grid.

    Args:
        forward_models (ForwardModels): The list of forward model objects
        iter_index (int): The iteration to use.
        nights_for_template (str or list): The nights to consider for averaging residuals to update the stellar template. Options are 'best' to use the night with the highest co-added S/N, a list of indices for specific nights, or an empty list to use all nights. defaults to [] for all nights.
    """
    current_stellar_template = np.copy(forward_models.templates_dict['star'])

    # Stores the shifted high resolution residuals (all on the star grid)
    residuals_hr = np.empty(shape=(forward_models.n_model_pix, forward_models.n_spec), dtype=np.float64)
    bad_pix_hr = np.empty(shape=(forward_models.n_model_pix, forward_models.n_spec), dtype=bool)
    tot_weights_hr = np.zeros(shape=(forward_models.n_model_pix, forward_models.n_spec), dtype=np.float64)
    
    # Stores the weighted median grid. Is set via loop, so pre-allocate.
    residuals_median = np.empty(forward_models.n_model_pix, dtype=np.float64)
    
    # These show the min and max of of the residuals for all observations, useful for plotting if desired.
    residuals_max = np.empty(forward_models.n_model_pix, dtype=np.float64)
    residuals_min = np.empty(forward_models.n_model_pix, dtype=np.float64)
    
    
    # Weight by 1 / rms^2
    rms = np.array([forward_models[ispec].opt[iter_index][0] for ispec in range(forward_models.n_spec)]) 
    rms_weights = 1 / rms**2
    good = np.where(np.isfinite(rms_weights))[0]
    bad = np.where(~np.isfinite(rms_weights))[0]
    if good.size == 0:
        rms_weights = np.ones(forward_models.n_spec)
    
    # bc vels
    bc_vels = np.array([fwm.data.bc_vel for fwm in forward_models], dtype=np.float64)
    
    # All nights
    if nights_for_template is None or type(nights_for_template) is list and len(nights_for_template) == 0:
        template_spec_indices = np.arange(forward_models.n_spec).astype(int)
    # Night with highest co-added S/N
    elif nights_for_template == 'best':
        night_index = determine_best_night(rms, forward_models.n_obs_nights)
        template_spec_indices = list(forward_models.get_all_spec_indices_from_night(night_index, forward_models.n_obs_nights))
    # User specified nights
    else:
        template_spec_indices = []
        for night in nights_for_template:
            template_spec_indices += list(forward_models.get_all_spec_indices_from_night(night - 1, forward_models.n_obs_nights))

    # Loop over spectra
    for ispec in range(forward_models.n_spec):

        # De-shift residual wavelength scale according to the barycenter correction
        # Or best doppler shift if using a non flat initial template
        if forward_models[0].models_dict['star'].from_synthetic:
            wave_stellar_frame = forward_models[ispec].wavelength_solutions[-1] * np.exp(-1 * forward_models[ispec].best_fit_pars[-1][forward_models[ispec].models_dict['star'].par_names[0]].value / cs.c)
        else:
            wave_stellar_frame = forward_models[ispec].wavelength_solutions[-1] * np.exp(forward_models[ispec].data.bc_vel / cs.c)

        # Telluric Weights
        tell_flux_hr = forward_models[ispec].models_dict['tellurics'].build(forward_models[ispec].best_fit_pars[-1], forward_models.templates_dict['tellurics'], current_stellar_template[:, 0])
        tell_flux_hr_convolved = forward_models[ispec].models_dict['lsf'].convolve_flux(tell_flux_hr, pars=forward_models[ispec].best_fit_pars[-1])
        tell_weights_hr = tell_flux_hr_convolved**2

        # For the high res grid, we need to interpolate the bad pixel mask onto high res grid.
        # Any pixels not equal to 1 after interpolation are considered bad.
        bad_pix_hr[:, ispec] = np.interp(current_stellar_template[:, 0], wave_stellar_frame, forward_models[ispec].data.badpix, left=0, right=0)
        bad = np.where(bad_pix_hr[:, ispec] < 1)[0]
        if bad.size > 0:
            bad_pix_hr[bad, ispec] = 0

        # Weights for the high res residuals
        tot_weights_hr[:, ispec] = rms_weights[ispec] * bad_pix_hr[:, ispec] * tell_weights_hr

        # Only use finite values and known good pixels for interpolating up to the high res grid.
        # Even though bad pixels are ignored later when median combining residuals,
        # they will still affect interpolation in unwanted ways.
        good = np.where(np.isfinite(forward_models[ispec].residuals[-1]) & (forward_models[ispec].data.badpix == 1))
        residuals_interp_hr = scipy.interpolate.CubicSpline(wave_stellar_frame[good], forward_models[ispec].residuals[-1][good].flatten(), bc_type='not-a-knot', extrapolate=False)(current_stellar_template[:, 0])

        # Determine values with np.nans and set weights equal to zero
        bad = np.where(~np.isfinite(residuals_interp_hr))[0]
        if bad.size > 0:
            tot_weights_hr[bad, ispec] = 0
            bad_pix_hr[bad, ispec] = 0

        # Also ensure all bad pix in hr residuals are nans, even though they have zero weight
        bad = np.where(tot_weights_hr[:, ispec] == 0)[0]
        if bad.size > 0:
            residuals_interp_hr[bad] = np.nan

        # Pass to final storage array
        residuals_hr[:, ispec] = residuals_interp_hr

    # Additional Weights:
    # Up-weight spectra with poor BC sampling.
    # In other words, we weight by the inverse of the histogram values of the BC distribution
    # Generate the histogram
    hist_counts, histx = np.histogram(bc_vels, bins=int(np.min([forward_models.n_spec, 10])), range=(np.min(bc_vels)-1, np.max(bc_vels)+1))
    
    # Check where we have no spectra (no observations in this bin)
    hist_counts = hist_counts.astype(np.float64)
    bad = np.where(hist_counts == 0)[0]
    if bad.size > 0:
        hist_counts[bad] = np.nan
    number_weights = 1 / hist_counts
    number_weights = number_weights / np.nansum(number_weights)

    # Loop over spectra and check which bin an observation belongs to
    # Then update the weights accordingly.
    if len(nights_for_template) == 0:
        for ispec in range(forward_models.n_spec):
            vbc = forward_models[ispec].data.bc_vel
            y = np.where(histx >= vbc)[0][0] - 1
            tot_weights_hr[:, ispec] = tot_weights_hr[:, ispec] * number_weights[y]

    # Only use specified nights
    tot_weights_hr = tot_weights_hr[:, template_spec_indices]
    bad_pix_hr = bad_pix_hr[:, template_spec_indices]
    residuals_hr = residuals_hr[:, template_spec_indices]

    # Co-add residuals according to a weighted median crunch
    # 1. If all weights at a given pixel are zero, set median value to zero.
    # 2. If there's more than one spectrum, compute the weighted median
    # 3. If there's only one spectrum, use those residuals, unless it's nan.
    for ix in range(forward_models.n_model_pix):
        if np.nansum(tot_weights_hr[ix, :]) == 0:
            residuals_median[ix] = 0
        else:
            if forward_models.n_spec > 1:
                residuals_median[ix] = pcmath.weighted_median(residuals_hr[ix, :], weights=tot_weights_hr[ix, :]/np.nansum(tot_weights_hr[ix, :]))
            elif np.isfinite(residuals_hr[ix, 0]):
                residuals_median[ix] = residuals_hr[ix, 0]
            else:
                residuals_median[ix] = 0

        # Store the min and max
        residuals_max[ix] = np.nanmax(residuals_hr[ix, :] * bad_pix_hr[ix, :])
        residuals_min[ix] = np.nanmin(residuals_hr[ix, :] * bad_pix_hr[ix, :])
        
    # Change any nans to zero
    bad = np.where(~np.isfinite(residuals_median))[0]
    if bad.size > 0:
        residuals_median[bad] = 0

    # Augment the template
    new_flux = current_stellar_template[:, 1] + residuals_median

    # Force the max to be less than 1.
    locs = np.where(new_flux > 1)[0]
    if locs.size > 0:
        new_flux[locs] = 1

    forward_models.templates_dict['star'][:, 1] = new_flux


def weighted_average(forward_models, iter_index=None, nights_for_template=None, templates_to_optimize=None):
    """Augments the stellar template by considering the weighted average of the residuals on a common high resolution grid.

    Args:
        forward_models (ForwardModels): The list of forward model objects
        iter_index (int): The iteration to use.
        nights_for_template (str or list): The nights to consider for averaging residuals to update the stellar template. Options are 'best' to use the night with the highest co-added S/N, a list of indices for specific nights, or an empty list to use all nights. defaults to [] for all nights.
    """
    current_stellar_template = np.copy(forward_models.templates_dict['star'])

    # Stores the shifted high resolution residuals (all on the star grid)
    residuals_hr = np.empty(shape=(forward_models.n_model_pix, forward_models.n_spec), dtype=np.float64) + np.nan
    bad_pix_hr = np.empty(shape=(forward_models.n_model_pix, forward_models.n_spec), dtype=bool)
    tot_weights_hr = np.zeros(shape=(forward_models.n_model_pix, forward_models.n_spec), dtype=np.float64)
    
    # Stores the weighted median grid. Is set via loop, so pre-allocate.
    residuals_average = np.empty(forward_models.n_model_pix, dtype=np.float64) + np.nan
    
    # These show the min and max of of the residuals for all observations, useful for plotting if desired.
    residuals_max = np.empty(forward_models.n_model_pix, dtype=np.float64) + np.nan
    residuals_min = np.empty(forward_models.n_model_pix, dtype=np.float64) + np.nan
    
    # Weight by 1 / rms^2
    rms = np.array([forward_models[ispec].opt[iter_index][0] for ispec in range(forward_models.n_spec)]) 
    rms_weights = 1 / rms**2
    good = np.where(np.isfinite(rms_weights))[0]
    bad = np.where(~np.isfinite(rms_weights))[0]
    if good.size == 0:
        rms_weights = np.ones(forward_models.n_spec)
    
    # bc vels
    bc_vels = np.array([fwm.data.bc_vel for fwm in forward_models], dtype=np.float64)
    
    # All nights
    if nights_for_template is None or type(nights_for_template) is list and len(nights_for_template) == 0:
        template_spec_indices = np.arange(forward_models.n_spec).astype(int)
    # Night with highest co-added S/N
    elif nights_for_template == 'best':
        night_index = determine_best_night(rms, forward_models.n_obs_nights)
        template_spec_indices = list(forward_models.get_all_spec_indices_from_night(night_index, forward_models.n_obs_nights))
    # User specified nights
    else:
        template_spec_indices = []
        for night in nights_for_template:
            template_spec_indices += list(forward_models.get_all_spec_indices_from_night(night - 1, forward_models.n_obs_nights))

    # Loop over spectra
    for ispec in range(forward_models.n_spec):

        # De-shift residual wavelength scale according to the barycenter correction
        # Or best doppler shift if using a non flat initial template
        if forward_models[0].models_dict['star'].from_synthetic:
            wave_stellar_frame = forward_models[ispec].wavelength_solutions[-1] * np.exp(-1 * forward_models[ispec].best_fit_pars[-1][forward_models[ispec].models_dict['star'].par_names[0]].value / cs.c)
        else:
            wave_stellar_frame = forward_models[ispec].wavelength_solutions[-1] * np.exp(forward_models[ispec].data.bc_vel / cs.c)

        # Telluric Weights
        tell_flux_hr = forward_models[ispec].models_dict['tellurics'].build(forward_models[ispec].best_fit_pars[-1], forward_models.templates_dict['tellurics'], current_stellar_template[:, 0])
        tell_flux_hr_convolved = forward_models[ispec].models_dict['lsf'].convolve_flux(tell_flux_hr, pars=forward_models[ispec].best_fit_pars[-1])
        tell_weights_hr = tell_flux_hr_convolved**2

        # For the high res grid, we need to interpolate the bad pixel mask onto high res grid.
        # Any pixels not equal to 1 after interpolation are considered bad.
        bad_pix_hr[:, ispec] = np.interp(current_stellar_template[:, 0], wave_stellar_frame, forward_models[ispec].data.badpix, left=0, right=0)
        bad = np.where(bad_pix_hr[:, ispec] < 1)[0]
        if bad.size > 0:
            bad_pix_hr[bad, ispec] = 0

        # Weights for the high res residuals
        tot_weights_hr[:, ispec] = rms_weights[ispec] * bad_pix_hr[:, ispec] * tell_weights_hr

        # Only use finite values and known good pixels for interpolating up to the high res grid.
        # Even though bad pixels are ignored later when median combining residuals,
        # they will still affect interpolation in unwanted ways.
        good = np.where(np.isfinite(forward_models[ispec].residuals[-1]) & (forward_models[ispec].data.badpix == 1))
        residuals_interp_hr = scipy.interpolate.CubicSpline(wave_stellar_frame[good], forward_models[ispec].residuals[-1][good].flatten(), bc_type='not-a-knot', extrapolate=False)(current_stellar_template[:, 0])

        # Determine values with np.nans and set weights equal to zero
        bad = np.where(~np.isfinite(residuals_interp_hr))[0]
        if bad.size > 0:
            tot_weights_hr[bad, ispec] = 0
            bad_pix_hr[bad, ispec] = 0

        # Also ensure all bad pix in hr residuals are nans, even though they have zero weight
        bad = np.where(tot_weights_hr[:, ispec] == 0)[0]
        if bad.size > 0:
            residuals_interp_hr[bad] = np.nan

        # Pass to final storage array
        residuals_hr[:, ispec] = residuals_interp_hr

    # Additional Weights:
    # Up-weight spectra with poor BC sampling.
    # In other words, we weight by the inverse of the histogram values of the BC distribution
    # Generate the histogram
    hist_counts, histx = np.histogram(bc_vels, bins=int(np.min([forward_models.n_spec, 10])), range=(np.min(bc_vels)-1, np.max(bc_vels)+1))
    
    # Check where we have no spectra (no observations in this bin)
    hist_counts = hist_counts.astype(np.float64)
    bad = np.where(hist_counts == 0)[0]
    if bad.size > 0:
        hist_counts[bad] = np.nan
    number_weights = 1 / hist_counts
    number_weights = number_weights / np.nansum(number_weights)

    # Loop over spectra and check which bin an observation belongs to
    # Then update the weights accordingly.
    if len(nights_for_template) == 0:
        for ispec in range(forward_models.n_spec):
            vbc = forward_models[ispec].data.bc_vel
            y = np.where(histx >= vbc)[0][0] - 1
            tot_weights_hr[:, ispec] = tot_weights_hr[:, ispec] * number_weights[y]

    # Only use specified nights
    tot_weights_hr = tot_weights_hr[:, template_spec_indices]
    bad_pix_hr = bad_pix_hr[:, template_spec_indices]
    residuals_hr = residuals_hr[:, template_spec_indices]
    
    # Co-add residuals according to a weighted median crunch
    # 1. If all weights at a given pixel are zero, set median value to zero.
    # 2. If there's more than one spectrum, compute the weighted median
    # 3. If there's only one spectrum, use those residuals, unless it's nan.
    for ix in range(forward_models.n_model_pix):
        if np.nansum(tot_weights_hr[ix, :]) == 0:
            residuals_average[ix] = 0
        else:
            if forward_models.n_spec > 1:
                # Further flag any pixels larger than 3*wstddev from a weighted average.
                wavg = pcmath.weighted_mean(residuals_hr[ix, :], tot_weights_hr[ix, :])
                wstddev = pcmath.weighted_stddev(residuals_hr[ix, :], tot_weights_hr[ix, :])
                diffs = np.abs(wavg - residuals_hr[ix, :])
                bad = np.where(diffs > 3*wstddev)[0]
                if bad.size > 0:
                    tot_weights_hr[ix, bad] = 0
                    bad_pix_hr[ix, bad] = 0
                residuals_average[ix] = pcmath.weighted_mean(residuals_hr[ix, :], tot_weights_hr[ix, :])
            elif np.isfinite(residuals_hr[ix, 0]):
                residuals_average[ix] = residuals_hr[ix, 0]
            else:
                residuals_average[ix] = 0

        # Store the min and max
        residuals_max[ix] = np.nanmax(residuals_hr[ix, :] * bad_pix_hr[ix, :])
        residuals_min[ix] = np.nanmin(residuals_hr[ix, :] * bad_pix_hr[ix, :])
        
    # Change any nans to zero
    bad = np.where(~np.isfinite(residuals_average))[0]
    if bad.size > 0:
        residuals_average[bad] = 0

    # Augment the template
    new_flux = current_stellar_template[:, 1] + residuals_average

    # Force the max to be less than 1.
    locs = np.where(new_flux > 1)[0]
    if locs.size > 0:
        new_flux[locs] = 1
        
    forward_models.templates_dict['star'][:, 1] = new_flux


# Uses pytorch to optimize the template
def global_fit(forward_models, templates_to_optimize=None, iter_index=None, nights_for_template=None):
    """Similar to Wobble, this will update the stellar template and lab frames by performing a gradient-based optimization via ADAM in pytorch considering all observations. Here, the template(s) are a parameter of thousands of points. The star is implemented in such a way that it will modify the current template, Doppler shift each observation and multiply them into the model. The lab frame is implemented in such a way that it will modify a zero based array, and then add this into each observation before convolution is performed.

    Args:
        forward_models (ForwardModels): The list of forward model objects
        iter_index (int): The iteration to use.
        templates_to_optimize (list of strings): valid entries are 'star', 'lab'.
        nights_for_template (None): May be set, but not used here.
    """
    
    # The number of lsf points
    n_lsf_pts = forward_models[0].models_dict['lsf'].nx
    
    # The high resolution master grid (also stellar template grid)
    wave_hr_master = torch.from_numpy(forward_models.templates_dict['star'][:, 0])
    
    # Grids to optimize
    if 'star' in templates_to_optimize:
        star_flux = torch.nn.Parameter(torch.from_numpy(forward_models.templates_dict['star'][:, 1].astype(np.float64)))
        
        # The current best fit stellar velocities
        star_vels = torch.from_numpy(np.array([forward_models[ispec].best_fit_pars[-1][forward_models[ispec].models_dict['star'].par_names[0]].value for ispec in range(forward_models.n_spec)]).astype(np.float64))
    else:
        star_flux = None
        star_vels = None
        
    if 'lab' in templates_to_optimize:
        residual_lab_flux = torch.nn.Parameter(torch.zeros(wave_hr_master.size()[0], dtype=torch.float64) + 1E-4)
    else:
        residual_lab_flux = None
    
    # The partial built forward model flux
    base_flux_models = torch.empty((forward_models.templates_dict['star'][:, 0].size, forward_models.n_spec), dtype=torch.float64)
    
    # The best fit LSF for each spec (optional)
    if 'lsf' in forward_models[0].models_dict and forward_models[0].models_dict['lsf'].enabled:
        lsfs = torch.empty((n_lsf_pts, forward_models.n_spec), dtype=torch.float64)
    else:
        lsfs = None
    
    # The data flux
    data_flux = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)
    
    # Bad pixel arrays for the data
    badpix = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)
    
    # The wavelength solutions
    waves_lr = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)
    
    # Weights, may just be binary mask
    weights = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)

    # Loop over spectra and extract to the above arrays
    for ispec in range(forward_models.n_spec):

        # Get the pars for this iteration and spectrum
        pars = forward_models[ispec].best_fit_pars[-1]

        if 'star' in templates_to_optimize:
            x, y = forward_models[ispec].build_hr_nostar(pars, iter_index)
        else:
            x, y = forward_models[ispec].build_hr(pars, iter_index)
            
        waves_lr[:, ispec], base_flux_models[:, ispec] = torch.from_numpy(x), torch.from_numpy(y)

        # Fetch lsf and flip for torch. As of now torch does not support the negative step
        if 'lsf' in forward_models[0].models_dict and forward_models[0].models_dict['lsf'].enabled:
            lsfs[:, ispec] = torch.from_numpy(forward_models[ispec].models_dict['lsf'].build(pars))
            lsfs[:, ispec] = torch.from_numpy(np.flip(lsfs[:, ispec].numpy(), axis=0).copy())

        # The data and weights, change bad vals to nan
        data_flux[:, ispec] = torch.from_numpy(np.copy(forward_models[ispec].data.flux))
        weights[:, ispec] = torch.from_numpy(np.copy(forward_models[ispec].data.badpix))
        bad = torch.where(~torch.isfinite(data_flux[:, ispec]) | ~torch.isfinite(weights[:, ispec]) | (weights[:, ispec] <= 0))[0]
        if len(bad) > 0:
            data_flux[bad, ispec] = np.nan
            weights[bad, ispec] = 0

    # CPU or GPU
    if torch.cuda.is_available():
        torch.device('cuda')
    else:
        torch.device('cpu')

    # Create the Torch model object
    model = TemplateOptimizer(base_flux_models, waves_lr, weights, data_flux, wave_hr_master, star_flux=star_flux, star_vels=star_vels, residual_lab_flux=residual_lab_flux, lsfs=lsfs)

    # Create the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print('Optimizing Template(s) ...', flush=True)

    for epoch in range(500):

        # Generate the model
        optimizer.zero_grad()
        loss = model.forward()

        # Back propagation (gradient calculation)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('epoch {}, loss {}'.format(epoch + 1, loss.item()), flush=True)

    if 'star' in templates_to_optimize:
        new_star_flux = model.star_flux.detach().numpy()
        locs = np.where(new_star_flux > 1)[0]
        if locs.size > 0:
            new_star_flux[locs] = 1
        forward_models.templates_dict['star'][:, 1] = new_star_flux

    if 'lab' in templates_to_optimize:
        residual_lab_flux_fit = model.residual_lab_flux.detach().numpy()
        forward_models.templates_dict['residual_lab'] = np.array([np.copy(forward_models.templates_dict['star'][:, 0]), residual_lab_flux_fit]).T
    



        class Interp1d(torch.autograd.Function):
            def __call__(self, x, y, xnew, out=None):
                return self.forward(x, y, xnew, out)

            def forward(ctx, x, y, xnew, out=None):

                # making the vectors at least 2D
                is_flat = {}
                require_grad = {}
                v = {}
                device = []
                for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
                    assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                                'at most 2-D.'
                    if len(vec.shape) == 1:
                        v[name] = vec[None, :]
                    else:
                        v[name] = vec
                    is_flat[name] = v[name].shape[0] == 1
                    require_grad[name] = vec.requires_grad
                    device = list(set(device + [str(vec.device)]))
                assert len(device) == 1, 'All parameters must be on the same device.'
                device = device[0]

                # Checking for the dimensions
                assert (v['x'].shape[1] == v['y'].shape[1]
                        and (
                            v['x'].shape[0] == v['y'].shape[0]
                            or v['x'].shape[0] == 1
                            or v['y'].shape[0] == 1
                            )
                        ), ("x and y must have the same number of columns, and either "
                            "the same number of row or one of them having only one "
                            "row.")

                reshaped_xnew = False
                if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
                and (v['xnew'].shape[0] > 1)):
                    # if there is only one row for both x and y, there is no need to
                    # loop over the rows of xnew because they will all have to face the
                    # same interpolation problem. We should just stack them together to
                    # call interp1d and put them back in place afterwards.
                    original_xnew_shape = v['xnew'].shape
                    v['xnew'] = v['xnew'].contiguous().view(1, -1)
                    reshaped_xnew = True

                # identify the dimensions of output and check if the one provided is ok
                D = max(v['x'].shape[0], v['xnew'].shape[0])
                shape_ynew = (D, v['xnew'].shape[-1])
                if out is not None:
                    if out.numel() != shape_ynew[0]*shape_ynew[1]:
                        # The output provided is of incorrect shape.
                        # Going for a new one
                        out = None
                    else:
                        ynew = out.reshape(shape_ynew)
                if out is None:
                    ynew = torch.zeros(*shape_ynew, dtype=y.dtype, device=device)

                # moving everything to the desired device in case it was not there
                # already (not handling the case things do not fit entirely, user will
                # do it if required.)
                for name in v:
                    v[name] = v[name].to(device)

                # calling searchsorted on the x values.
                #ind = ynew
                #searchsorted(v['x'].contiguous(), v['xnew'].contiguous(), ind)
                ind = np.searchsorted(v['x'].contiguous().numpy().flatten(), v['xnew'].contiguous().numpy().flatten())
                ind = torch.tensor(ind)
                # the `-1` is because searchsorted looks for the index where the values
                # must be inserted to preserve order. And we want the index of the
                # preceeding value.
                ind -= 1
                # we clamp the index, because the number of intervals is x.shape-1,
                # and the left neighbour should hence be at most number of intervals
                # -1, i.e. number of columns in x -2
                ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1).long()

                # helper function to select stuff according to the found indices.
                def sel(name):
                    if is_flat[name]:
                        return v[name].contiguous().view(-1)[ind]
                    return torch.gather(v[name], 1, ind)

                # activating gradient storing for everything now
                enable_grad = False
                saved_inputs = []
                for name in ['x', 'y', 'xnew']:
                    if require_grad[name]:
                        enable_grad = True
                        saved_inputs += [v[name]]
                    else:
                        saved_inputs += [None, ]
                # assuming x are sorted in the dimension 1, computing the slopes for
                # the segments
                is_flat['slopes'] = is_flat['x']
                # now we have found the indices of the neighbors, we start building the
                # output. Hence, we start also activating gradient tracking
                with torch.enable_grad() if enable_grad else contextlib.suppress():
                    v['slopes'] = (
                            (v['y'][:, 1:]-v['y'][:, :-1])
                            /
                            (v['x'][:, 1:]-v['x'][:, :-1])
                        )

                    # now build the linear interpolation
                    ynew = sel('y') + sel('slopes')*(
                                            v['xnew'] - sel('x'))

                    if reshaped_xnew:
                        ynew = ynew.view(original_xnew_shape)

                ctx.save_for_backward(ynew, *saved_inputs)
                return ynew

            @staticmethod
            def backward(ctx, grad_out):
                inputs = ctx.saved_tensors[1:]
                gradients = torch.autograd.grad(
                                ctx.saved_tensors[0],
                                [i for i in inputs if i is not None],
                                grad_out, retain_graph=True)
                result = [None, ] * 5
                pos = 0
                for index in range(len(inputs)):
                    if inputs[index] is not None:
                        result[index] = gradients[pos]
                        pos += 1
                return (*result,)
        
      
        
# Class to optimize the forward model
if 'torch' in sys.modules:
    class TemplateOptimizer(torch.nn.Module):

        def __init__(self, base_flux_models, waves_lr, weights, data_flux, wave_hr_master, star_flux=None, star_vels=None, residual_lab_flux=None, lsfs=None):
            
            # Parent init
            super(TemplateOptimizer, self).__init__()
            
            # Shape information
            self.nx_data, self.n_spec = data_flux.shape
            
            # High res master wave grid for all observations
            self.wave_hr_master = wave_hr_master # shape=(nmx,)
            
            # The number of model pixels
            self.nx_model = self.wave_hr_master.size()[0]
            
            # Actual data
            self.data_flux = data_flux # shape=(nx, n_spec)

            # Potential parameters to optimize
            self.star_flux = star_flux # coherence in stellar (quasi) rest frame, shape=(nmx,)
            self.residual_lab_flux = residual_lab_flux # coherence in lab frame, shape=(nmx,)
            
            # The base flux
            self.base_flux_models = base_flux_models # shape=(nx, n_spec)
            
            # Current wavelength solutions
            self.waves_lr = waves_lr # shape=(nx, n_spec)
            
            # Current best fit stellar velocities
            self.star_vels = star_vels # shape=(n_spec,)
            
            # Optimization weights
            self.weights = weights # shape=(nx, n_spec)
            
            # The lsf (optional)
            if lsfs is not None:
                self.nx_lsf = lsfs.shape[0]
                self.lsfs = torch.ones(1, 1, self.nx_lsf, self.n_spec, dtype=torch.float64)
                self.lsfs[0, 0, :, :] = lsfs
                self.nx_pad1 = int(self.nx_lsf / 2) - 1
                self.nx_pad2 = int(self.nx_lsf / 2)
            else:
                self.lsfs = None

        def forward(self):
        
            # Stores all low res models
            models_lr = torch.empty((self.nx_data, self.n_spec), dtype=torch.float64) + np.nan
            
            # Loop over observations
            for ispec in range(self.n_spec):
                
                # Doppler shift the stellar wavelength grid used for this observation.
                if self.star_flux is not None and self.residual_lab_flux is not None:
                    wave_hr_star_shifted = self.wave_hr_master * torch.exp(self.star_vels[ispec] / cs.c)
                    star = self.Interp1d()(wave_hr_star_shifted, self.star_flux, self.wave_hr_master).flatten()
                    model = self.base_flux_models[:, ispec] * star + self.residual_lab_flux
                elif self.star_flux is not None and self.residual_lab_flux is None:
                    wave_hr_star_shifted = self.wave_hr_master * torch.exp(self.star_vels[ispec] / cs.c)
                    star = self.Interp1d()(wave_hr_star_shifted, self.star_flux, self.wave_hr_master).flatten()
                    model = self.base_flux_models[:, ispec] * star
                else:
                    model = self.base_flux_models[:, ispec] + self.residual_lab_flux
                    
                # Convolution. NOTE: PyTorch convolution is a pain in the ass
                if self.lsfs is not None:
                    model_p = torch.ones((1, 1, self.nx_model + self.nx_pad1 + self.nx_pad2), dtype=torch.float64)
                    model_p[0, 0, self.nx_pad1:(-self.nx_pad2)] = model
                    conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
                    conv.weight.data = self.lsfs[:, :, :, ispec]
                    model = conv(model_p).flatten()

                # Interpolate onto data grid
                good = torch.where(torch.isfinite(self.weights[:, ispec]) & (self.weights[:, ispec] > 0))[0]
                models_lr[good, ispec] = self.Interp1d()(self.wave_hr_master, model, self.waves_lr[good, ispec])

            # Weighted RMS
            good = torch.where(torch.isfinite(self.weights) & (self.weights > 0) & torch.isfinite(models_lr))[0]
            wdiffs2 = (models_lr[good] - self.data_flux[good])**2 * self.weights[good]
            loss = torch.sqrt(torch.sum(wdiffs2) / (torch.sum(self.weights[good])))

            return loss
        
        
        @staticmethod
        def h_poly_helper(tt):
            A = torch.tensor([
                [1, 0, -3, 2],
                [0, 1, -2, 1],
                [0, 0, 3, -2],
                [0, 0, -1, 1]
                ], dtype=tt[-1].dtype)
            return [
                sum( A[i, j]*tt[j] for j in range(4) )
                for i in range(4) ]
            
        @classmethod
        def h_poly(cls, t):
            tt = [ None for _ in range(4) ]
            tt[0] = 1
            for i in range(1, 4):
                tt[i] = tt[i-1]*t
            return cls.h_poly_helper(tt)

        @classmethod
        def H_poly(cls, t):
            tt = [ None for _ in range(4) ]
            tt[0] = t
            for i in range(1, 4):
                tt[i] = tt[i-1]*t*i/(i+1)
            return cls.h_poly_helper(tt)

        @classmethod
        def interpcs(cls, x, y, xs):
            m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
            m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
            I = np.searchsorted(x[1:], xs)
            dx = (x[I+1]-x[I])
            hh = cls.h_poly((xs-x[I])/dx)
            return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

        class Interp1d(torch.autograd.Function):
            def __call__(self, x, y, xnew, out=None):
                return self.forward(x, y, xnew, out)

            def forward(ctx, x, y, xnew, out=None):

                # making the vectors at least 2D
                is_flat = {}
                require_grad = {}
                v = {}
                device = []
                for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
                    assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                                'at most 2-D.'
                    if len(vec.shape) == 1:
                        v[name] = vec[None, :]
                    else:
                        v[name] = vec
                    is_flat[name] = v[name].shape[0] == 1
                    require_grad[name] = vec.requires_grad
                    device = list(set(device + [str(vec.device)]))
                assert len(device) == 1, 'All parameters must be on the same device.'
                device = device[0]

                # Checking for the dimensions
                assert (v['x'].shape[1] == v['y'].shape[1]
                        and (
                            v['x'].shape[0] == v['y'].shape[0]
                            or v['x'].shape[0] == 1
                            or v['y'].shape[0] == 1
                            )
                        ), ("x and y must have the same number of columns, and either "
                            "the same number of row or one of them having only one "
                            "row.")

                reshaped_xnew = False
                if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
                and (v['xnew'].shape[0] > 1)):
                    # if there is only one row for both x and y, there is no need to
                    # loop over the rows of xnew because they will all have to face the
                    # same interpolation problem. We should just stack them together to
                    # call interp1d and put them back in place afterwards.
                    original_xnew_shape = v['xnew'].shape
                    v['xnew'] = v['xnew'].contiguous().view(1, -1)
                    reshaped_xnew = True

                # identify the dimensions of output and check if the one provided is ok
                D = max(v['x'].shape[0], v['xnew'].shape[0])
                shape_ynew = (D, v['xnew'].shape[-1])
                if out is not None:
                    if out.numel() != shape_ynew[0]*shape_ynew[1]:
                        # The output provided is of incorrect shape.
                        # Going for a new one
                        out = None
                    else:
                        ynew = out.reshape(shape_ynew)
                if out is None:
                    ynew = torch.zeros(*shape_ynew, dtype=y.dtype, device=device)

                # moving everything to the desired device in case it was not there
                # already (not handling the case things do not fit entirely, user will
                # do it if required.)
                for name in v:
                    v[name] = v[name].to(device)

                # calling searchsorted on the x values.
                #ind = ynew
                #searchsorted(v['x'].contiguous(), v['xnew'].contiguous(), ind)
                ind = np.searchsorted(v['x'].contiguous().numpy().flatten(), v['xnew'].contiguous().numpy().flatten())
                ind = torch.tensor(ind)
                # the `-1` is because searchsorted looks for the index where the values
                # must be inserted to preserve order. And we want the index of the
                # preceeding value.
                ind -= 1
                # we clamp the index, because the number of intervals is x.shape-1,
                # and the left neighbour should hence be at most number of intervals
                # -1, i.e. number of columns in x -2
                ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1).long()

                # helper function to select stuff according to the found indices.
                def sel(name):
                    if is_flat[name]:
                        return v[name].contiguous().view(-1)[ind]
                    return torch.gather(v[name], 1, ind)

                # activating gradient storing for everything now
                enable_grad = False
                saved_inputs = []
                for name in ['x', 'y', 'xnew']:
                    if require_grad[name]:
                        enable_grad = True
                        saved_inputs += [v[name]]
                    else:
                        saved_inputs += [None, ]
                # assuming x are sorted in the dimension 1, computing the slopes for
                # the segments
                is_flat['slopes'] = is_flat['x']
                # now we have found the indices of the neighbors, we start building the
                # output. Hence, we start also activating gradient tracking
                with torch.enable_grad() if enable_grad else contextlib.suppress():
                    v['slopes'] = (
                            (v['y'][:, 1:]-v['y'][:, :-1])
                            /
                            (v['x'][:, 1:]-v['x'][:, :-1])
                        )

                    # now build the linear interpolation
                    ynew = sel('y') + sel('slopes')*(
                                            v['xnew'] - sel('x'))

                    if reshaped_xnew:
                        ynew = ynew.view(original_xnew_shape)

                ctx.save_for_backward(ynew, *saved_inputs)
                return ynew

            @staticmethod
            def backward(ctx, grad_out):
                inputs = ctx.saved_tensors[1:]
                gradients = torch.autograd.grad(
                                ctx.saved_tensors[0],
                                [i for i in inputs if i is not None],
                                grad_out, retain_graph=True)
                result = [None, ] * 5
                pos = 0
                for index in range(len(inputs)):
                    if inputs[index] is not None:
                        result[index] = gradients[pos]
                        pos += 1
                return (*result,)
        

################################
#### More Helpful functions ####
################################

def determine_best_night(rms, n_obs_nights):
    """Determines the night with the highest co-added S/N given the RMS of the fits.

    Args:
        rms (np.ndarray): The array of RMS values from fitting.
        n_obs_nights (np.ndarray): The number of observations on each night, has length = total number of nights.
        templates_to_optimize: For now, only the star is able to be optimized. Future updates will include a lab-frame coherence  simultaneous fit.
    """
    f = 0
    l = n_obs_nights[0]
    n_nights = len(n_obs_nights)
    nightly_snrs = np.empty(n_nights, dtype=float)
    for inight in range(n_nights):
        
        nightly_snrs[inight] = np.sqrt(np.nansum((1 / rms[f:l]**2)))
        
        if inight < n_nights - 1:
                f += n_obs_nights[inight]
                l += n_obs_nights[inight+1]
                
    best_night_index = np.nanargmax(nightly_snrs)
    return best_night_index
        

# This calculates the weighted median of a data set for rolling calculations
def estimate_continuum(x, y, width=7, n_knots=8, cont_val=0.98, smooth=True):
    """This will estimate the continuum with adjustable spline knots.

    Args:
        x (np.ndarray): The wavelength array.
        y (np.ndarray): The flux array.
        width (float, optional): The width of the window in units of x. Defaults to 7.
        n_knots (int, optional): The number of spline knots. Defaults to 8.
        cont_val (float, optional): The estimate of the percentile of the continuum. Defaults to 0.98.
        smooth (bool, optional): Whether or not to smooth the input spectrum. Defaults to True.

    Returns:
        np.ndarray: The estimate of the continuum
    """
    nx = x.size
    continuum_coarse = np.ones(nx, dtype=np.float64)
    if smooth:
        ys = pcmath.median_filter1d(y, width=7)
    else:
        ys = np.copy(y)
    for ix in range(nx):
        use = np.where((x > x[ix]-width/2) & (x < x[ix]+width/2) & np.isfinite(y))[0]
        if use.size == 0 or np.all(~np.isfinite(ys[use])):
            continuum_coarse[ix] = np.nan
        else:
            continuum_coarse[ix] = pcmath.weighted_median(ys[use], weights=None, med_val=cont_val)
    good = np.where(np.isfinite(ys))[0]
    knot_points = x[np.linspace(good[0], good[-1], num=n_knots).astype(int)]
    cspline = scipy.interpolate.CubicSpline(knot_points, continuum_coarse[np.linspace(good[0], good[-1], num=n_knots).astype(int)], extrapolate=False, bc_type='not-a-knot')
    continuum = cspline(x)
    return continuum


def fit_continuum_wobble(x, y, mask, order=6, nsigma=[0.3,3.0], maxniter=50):
    """Fit the continuum using sigma clipping. This function is a modified version from Megan Bedell's Wobble code.
    Args:
        x: The wavelengths.
        y: The log-fluxes.
        order: The polynomial order to use
        nsigma: The sigma clipping threshold: tuple (low, high)
        maxniter: The maximum number of iterations to do
    Returns:
        The value of the continuum at the wavelengths in x in log space.
    """
    good = np.where(np.isfinite(x) & np.isfinite(y))[0]
    xx, yy = x[good], y[good]
    yy = np.copy(y)
    yy = pcmath.median_filter1d(yy, 7)
    A = np.vander(x - np.nanmean(x), order+1)
    m = np.ones(len(x), dtype=bool)
    for i in range(maxniter):
        m[mask == 0] = 0  # mask out the bad pixels
        w = np.linalg.solve(np.dot(A[m].T, A[m]), np.dot(A[m].T, yy[m]))
        mu = np.dot(A, w)
        resid = yy - mu
        sigma = np.sqrt(np.nanmedian(resid**2))
        m_new = (resid > -nsigma[0]*sigma) & (resid < nsigma[1]*sigma)
        if m.sum() == m_new.sum():
            m = m_new
            break
        m = m_new
    return mu
