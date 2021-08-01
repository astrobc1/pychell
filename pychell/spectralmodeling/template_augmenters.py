# Base Python
import os
import sys
import warnings

# Pychell deps
import pychell
import pychell.maths as pcmath

# Graphics
import matplotlib.pyplot as plt
try:
    plt.style.use(f"{os.path.dirname(pychell.__file__)}{os.sep}gadfly_stylesheet.mplstyle")
except:
    warnings.warn("Could not locate gadfly stylesheet, using default matplotlib stylesheet.")

# Maths
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np
import scipy.interpolate # Cubic spline LSQ fitting

class TemplateAugmenter:
    
    def __init__(self, use_nights=None, downweight_tellurics=True, max_thresh=None):
        self.use_nights = use_nights
        self.downweight_tellurics = downweight_tellurics
        self.max_thresh = max_thresh
        
    def augment_templates(self, specrvprob, iter_index):
        pass

class CubicSplineLSQ(TemplateAugmenter):
    
    def augment_templates(self, specrvprob, iter_index):
        
        # Which nights / spectra to consider
        if self.use_nights is None:
            
            # Default use all nights
            self.use_nights = np.arange(specrvprob.n_nights).astype(int)

        # Unpack the current stellar template
        current_stellar_template = np.copy(specrvprob.spectral_model.templates_dict["star"])
        
        # Get the fit metric
        fit_metrics = np.full(specrvprob.n_spec, np.nan)
        for ispec in range(specrvprob.n_spec):
            fit_metrics[ispec] = np.abs(specrvprob.opt_results[ispec, iter_index]["fbest"])
        
        # Weights according to fit metric
        fit_weights = 1 / fit_metrics**2
        good = np.where(np.isfinite(fit_weights))[0]
        bad = np.where(~np.isfinite(fit_weights))[0]
        if good.size == 0:
            fit_weights = np.ones(specrvprob.n_spec)
            
        wave_star_rest = []
        residuals = []
        weights = []
        
        # Loop over data and build residuals
        for ispec in range(specrvprob.n_spec):
                
            # Best fit pars
            pars = specrvprob.opt_results[ispec, iter_index]["pbest"]
        
            # Init
            specrvprob.spectral_model.initialize(pars, specrvprob.data[ispec], iter_index=iter_index)
        
            # Generate the low res model
            wave_data, model_lr = specrvprob.spectral_model.build(pars)
            
            # Residuals
            residuals_lr = specrvprob.data[ispec].flux - model_lr
            residuals += residuals_lr.tolist()

            # Shift to a pseudo rest frame. All must start from same frame
            if specrvprob.spectral_model.star.from_flat and iter_index == 0:
                vel = specrvprob.data[ispec].bc_vel
            else:
                vel = -1 * pars[specrvprob.spectral_model.star.par_names[0]].value
                
            _wave_star_rest = pcmath.doppler_shift(wave_data, vel, flux=None, wave_out=None, interp=None).tolist()
            wave_star_rest += _wave_star_rest
            
            # Telluric weights, must doppler shift them as well.
            if self.downweight_tellurics:
                tell_flux = specrvprob.spectral_model.tellurics.build(pars, specrvprob.spectral_model.templates_dict["tellurics"], wave_data)
                if specrvprob.spectral_model.lsf is not None:
                    tell_flux = specrvprob.spectral_model.lsf.convolve_flux(tell_flux, pars)
                tell_flux = pcmath.doppler_shift(wave_data, vel, flux=tell_flux)
                tell_weights = tell_flux**2
                _weights = specrvprob.data[ispec].mask * fit_weights[ispec] * tell_weights
            else:    
                _weights = specrvprob.data[ispec].mask * fit_weights[ispec]
                
            weights += _weights.tolist()
            
        # Convert to numpy arrays
        wave_star_rest = np.array(wave_star_rest)
        residuals = np.array(residuals)
        weights = np.array(weights)
    
        # Remove all bad pixels.
        good = np.where(np.isfinite(wave_star_rest) & np.isfinite(residuals) & (weights > 0))[0]
        wave_star_rest, residuals, weights = wave_star_rest[good], residuals[good], weights[good]

        # Sort the wavelengths
        ss = np.argsort(wave_star_rest)
        wave_star_rest, residuals, weights = wave_star_rest[ss], residuals[ss], weights[ss]
    
        # Knot points are roughly the detector grid.
        # Ensure the data surrounds the knots.
        knots_init = np.linspace(wave_star_rest[0] + 0.001, wave_star_rest[-1] - 0.001, num=specrvprob.spectral_model.sregion.pix_len())
    
        # Remove bad knots
        bad_knots = []
        for iknot in range(len(knots_init) - 1):
            n = np.where((wave_star_rest > knots_init[iknot]) & (wave_star_rest < knots_init[iknot+1]))[0].size
            if n == 0:
                bad_knots.append(iknot)
        knots = np.delete(knots_init, bad_knots)
    
        # Do the fit
        weights /= np.nansum(weights) # probably irrelevant
        spline_fitter = scipy.interpolate.LSQUnivariateSpline(wave_star_rest, residuals, t=knots, w=weights, k=3, ext=1)
    
        # Use the fit to determine the hr residuals to add
        residuals_hr_fit = spline_fitter(current_stellar_template[:, 0])

        # Remove bad regions
        bad = np.where((current_stellar_template[:, 0] <= knots[0]) | (current_stellar_template[:, 0] >= knots[-1]))[0]
        if bad.size > 0:
            residuals_hr_fit[bad] = 0

        # Augment the template
        new_flux = current_stellar_template[:, 1] + residuals_hr_fit
    
        # Force the max to be less than 1.
        if self.max_thresh is not None:
            bad = np.where(new_flux > self.max_thresh)[0]
            if bad.size > 0:
                new_flux[bad] = self.max_thresh
    
        # Update the template
        specrvprob.spectral_model.templates_dict['star'][:, 1] = new_flux
    
class WeightedMedian(TemplateAugmenter):

    def augment_templates(self, specrvprob, iter_index):
    
        # Which nights / spectra to consider
        if self.use_nights is None:
            
            # Default use all nights
            self.use_nights = np.arange(specrvprob.n_nights).astype(int)

        # Unpack the current stellar template
        current_stellar_template = np.copy(specrvprob.spectral_model.templates_dict["star"])
        
        # Get the fit metric
        fit_metrics = np.full(specrvprob.n_spec, np.nan)
        for ispec in range(specrvprob.n_spec):
            fit_metrics[ispec] = np.abs(specrvprob.opt_results[ispec, iter_index]["fbest"])
        
        # Weights according to fit metric
        fit_weights = 1 / fit_metrics**2
        good = np.where(np.isfinite(fit_weights))[0]
        bad = np.where(~np.isfinite(fit_weights))[0]
        if good.size == 0:
            fit_weights = np.ones(specrvprob.n_spec)

        # Storage arrays
        nx  = len(current_stellar_template[:, 0])
        residuals_median = np.zeros(nx)
        residuals = np.zeros(shape=(nx, specrvprob.n_spec), dtype=float)
        weights = np.zeros(shape=(nx, specrvprob.n_spec), dtype=float)

        # Loop over spectra
        for ispec in range(specrvprob.n_spec):
            
            # Best fit pars
            pars = specrvprob.opt_results[ispec, iter_index]["pbest"]
        
            # Init the chunk
            specrvprob.spectral_model.initialize(pars, specrvprob.data[ispec], iter_index=iter_index)
        
            # Generate the low res model
            wave_data, model_lr = specrvprob.spectral_model.build(pars)
            
            # Residuals
            residuals_lr = specrvprob.data[ispec].flux - model_lr

            # Shift to a pseudo rest frame. All must start from same frame
            if specrvprob.spectral_model.star.from_flat and iter_index == 0:
                vel = specrvprob.data[ispec].bc_vel
            else:
                vel = -1 * pars[specrvprob.spectral_model.star.par_names[0]].value
                
            # Shift residuals
            wave_star_rest = pcmath.doppler_shift(wave_data, vel, flux=None, wave_out=None, interp=None)
            residuals[:, ispec] = pcmath.cspline_interp(wave_star_rest, residuals_lr, current_stellar_template[:, 0])

            # Telluric weights, must doppler shift them as well.
            if self.downweight_tellurics:
                tell_flux = specrvprob.spectral_model.tellurics.build(pars, specrvprob.spectral_model.templates_dict["tellurics"], wave_data)
                if specrvprob.spectral_model.lsf is not None:
                    tell_flux = specrvprob.spectral_model.lsf.convolve_flux(tell_flux, pars)
                tell_flux = pcmath.doppler_shift(wave_data, vel, flux=tell_flux)
                tell_weights = tell_flux**2
                weights_lr = specrvprob.data[ispec].mask * fit_weights[ispec] * tell_weights
            else:
                weights_lr = specrvprob.data[ispec].mask * fit_weights[ispec]
            
            # Interpolate to a high res grid
            weights_hr = pcmath.lin_interp(wave_star_rest, weights_lr, current_stellar_template[:, 0])
            bad = np.where((weights_hr < 0) | ~np.isfinite(weights_hr))[0]
            if bad.size > 0:
                weights_hr[bad] = 0
            weights[:, ispec] = weights_hr


        # Co-add residuals according to a weighted median crunch
        # 1. If all weights at a given pixel are zero, set median value to zero.
        # 2. If there's more than one spectrum, compute the weighted median
        # 3. If there's only one spectrum, use those residuals, unless it's nan.
        for ix in range(nx):
            ww, rr = weights[ix, :], residuals[ix, :]
            if np.nansum(ww) == 0:
                residuals_median[ix] = 0
            else:
                good = np.where((ww > 0) & np.isfinite(ww))[0]
                if good.size == 0:
                    residuals_median[ix] = 0
                elif good.size == 1:
                    residuals_median[ix] = rr[good[0]]
                else:
                    residuals_median[ix] = pcmath.weighted_median(rr, weights=ww)

        # Change any nans to zero just in case
        bad = np.where(~np.isfinite(residuals_median))[0]
        if bad.size > 0:
            residuals_median[bad] = 0

        # Augment the template
        new_flux = current_stellar_template[:, 1] + residuals_median
        
        # Perform cspline lsq regression
        # Generate knots
        good = np.where(np.isfinite(new_flux))[0]
        wave_min, wave_max = current_stellar_template[good[0], 0], current_stellar_template[good[-1], 0]
        knots = np.linspace(wave_min, wave_max, num=specrvprob.spectral_model.sregion.pix_len())
        
        # Remove bad knots
        bad_knots = []
        for iknot in range(len(knots) - 1):
            n = np.where((wave_star_rest > knots[iknot]) & (wave_star_rest < knots[iknot+1]))[0].size
            if n == 0:
                bad_knots.append(iknot)
        knots = np.delete(knots, bad_knots)
        
        # Fit with cubic spline
        spline_fitter = scipy.interpolate.LSQUnivariateSpline(current_stellar_template[:, 0], new_flux, t=knots[1:-1], k=3, ext=0)
        new_flux = spline_fitter(current_stellar_template[:, 0])
        
        # Force the max to be less than 1.
        if self.max_thresh is not None:
            bad = np.where(new_flux > self.max_thresh)[0]
            if bad.size > 0:
                new_flux[bad] = self.max_thresh
    
        # Update the template
        specrvprob.spectral_model.templates_dict['star'][:, 1] = new_flux
        
        
class WeightedMean(TemplateAugmenter):

    def augment_templates(self, specrvprob, iter_index):
    
        # Which nights / spectra to consider
        if self.use_nights is None:
            
            # Default use all nights
            self.use_nights = np.arange(specrvprob.n_nights).astype(int)

        # Unpack the current stellar template
        current_stellar_template = np.copy(specrvprob.spectral_model.templates_dict["star"])
        
        # Get the fit metric
        fit_metrics = np.full(specrvprob.n_spec, np.nan)
        for ispec in range(specrvprob.n_spec):
            fit_metrics[ispec] = np.abs(specrvprob.opt_results[ispec, iter_index]["fbest"])
        
        # Weights according to fit metric
        fit_weights = 1 / fit_metrics**2
        good = np.where(np.isfinite(fit_weights))[0]
        bad = np.where(~np.isfinite(fit_weights))[0]
        if good.size == 0:
            fit_weights = np.ones(specrvprob.n_spec)

        # Storage arrays
        nx  = len(current_stellar_template[:, 0])
        residuals_median = np.zeros(nx)
        residuals = np.zeros(shape=(nx, specrvprob.n_spec), dtype=float)
        weights = np.zeros(shape=(nx, specrvprob.n_spec), dtype=float)

        # Loop over spectra
        for ispec in range(specrvprob.n_spec):
            
            # Best fit pars
            pars = specrvprob.opt_results[ispec, iter_index]["pbest"]
        
            # Init the chunk
            specrvprob.spectral_model.initialize(pars, specrvprob.data[ispec], iter_index=iter_index)
        
            # Generate the low res model
            wave_data, model_lr = specrvprob.spectral_model.build(pars)
            
            # Residuals
            residuals_lr = specrvprob.data[ispec].flux - model_lr

            # Shift to a pseudo rest frame. All must start from same frame
            if specrvprob.spectral_model.star.from_flat and iter_index == 0:
                vel = specrvprob.data[ispec].bc_vel
            else:
                vel = -1 * pars[specrvprob.spectral_model.star.par_names[0]].value
                
            # Shift residuals
            wave_star_rest = pcmath.doppler_shift(wave_data, vel, flux=None, wave_out=None, interp=None)
            residuals[:, ispec] = pcmath.cspline_interp(wave_star_rest, residuals_lr, current_stellar_template[:, 0])

            # Telluric weights, must doppler shift them as well.
            if self.downweight_tellurics:
                tell_flux = specrvprob.spectral_model.tellurics.build(pars, specrvprob.spectral_model.templates_dict["tellurics"], wave_data)
                if specrvprob.spectral_model.lsf is not None:
                    tell_flux = specrvprob.spectral_model.lsf.convolve_flux(tell_flux, pars)
                tell_flux = pcmath.doppler_shift(wave_data, vel, flux=tell_flux)
                tell_weights = tell_flux**2
                weights_lr = specrvprob.data[ispec].mask * fit_weights[ispec] * tell_weights
            else:
                weights_lr = specrvprob.data[ispec].mask * fit_weights[ispec]
            
            # Final weights
            weights_hr = pcmath.cspline_interp(wave_star_rest, weights_lr, current_stellar_template[:, 0])
            bad = np.where(weights_hr < 0)[0]
            if bad.size > 0:
                weights_hr[bad] = 0
            weights[:, ispec] = weights_hr


        # Co-add residuals according to a weighted median crunch
        # 1. If all weights at a given pixel are zero, set median value to zero.
        # 2. If there's more than one spectrum, compute the weighted median
        # 3. If there's only one spectrum, use those residuals, unless it's nan.
        for ix in range(nx):
            ww, rr = weights[ix, :], residuals[ix, :]
            if np.nansum(ww) == 0:
                residuals_median[ix] = 0
            else:
                good = np.where((ww > 0) & np.isfinite(ww))[0]
                if good.size == 0:
                    residuals_median[ix] = 0
                elif good.size == 1:
                    residuals_median[ix] = rr[good[0]]
                else:
                    residuals_median[ix] = pcmath.weighted_mean(rr, ww)

        # Change any nans to zero just in case
        bad = np.where(~np.isfinite(residuals_median))[0]
        if bad.size > 0:
            residuals_median[bad] = 0

        # Augment the template
        new_flux = current_stellar_template[:, 1] + residuals_median
        
        # Force the max to be less than 1.
        if self.max_thresh is not None:
            bad = np.where(new_flux > self.max_thresh)[0]
            if bad.size > 0:
                new_flux[bad] = self.max_thresh
    
        # Update the template
        specrvprob.spectral_model.templates_dict['star'][:, 1] = new_flux


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
    nightly_snrs = np.array([np.sqrt(np.nansum((1 / rms[f:l]**2))) for i, f, l in pcutils.nightly_iteration(n_obs_nights)], dtype=float)
    best_night_index = np.nanargmax(nightly_snrs)
    return best_night_index