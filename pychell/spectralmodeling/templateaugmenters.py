# Base Python
import os
import sys
import warnings
import copy

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
    
    def __init__(self, use_nights=None, weight_tellurics=False, max_thresh=None, weight_fits=True):
        self.use_nights = use_nights
        self.weight_tellurics = weight_tellurics
        self.max_thresh = max_thresh
        self.weight_fits = weight_fits
        
    def augment_templates(self, specrvprob, iter_index):
        raise NotImplementedError(f"Must implement the method augment_templates for {self.__class__.__name__}")

# The nominal approach!
class CubicSplineLSQ(TemplateAugmenter):

    def __init__(self, use_nights=None, weight_fits=True, weight_tellurics=False, max_thresh=None, ideconv=True, mask_tellurics=False):
        super().__init__(use_nights, weight_tellurics, max_thresh, weight_fits)
        self.ideconv = ideconv
    
    def augment_templates(self, specrvprob, iter_index):

        # Idea:
        # IF ideconv:
        #    1. Compute and sample all residuals on common grid (low res, common reference frame).
        #    2. Compute median residuals.
        #    3. Flag outliers in residuals between between median residuals.
        #    4. Do cspline LSQ on low res residuals (not common grid).
        # ELSE:
        #    1. Correct each observation according to continuum, tellurics, gas cell (if needed).
        #    2. Sample all modified spectra on common grid (low res, common reference frame).
        #    3. Compute median spectrum.
        #    4. Flag outliers in modified spectra on common grid (low res, common reference frame).
        #    5. Do cspline LSQ on modified spectra (not common grid).
        # NOTES:
        #    1. Weights are ~ 1 / RMS^2 * TELL_FLUX^2 (both are optional)

        # Wave sampling (approx)
        wave_sampling = 1 / specrvprob.spectral_model.sregion.pix_per_wave()

        # High res stellar flux grid
        wave_star_hr = specrvprob.spectral_model.templates_dict['star'][:, 0]

        # The number of detector pixels (approx.)
        n_pix_lr = int(wave_star_hr.size * wave_sampling)

        # Common wavelength grid for low res data
        wave_star_coherent_lr = np.linspace(wave_star_hr.min(), wave_star_hr.max(), num=n_pix_lr)

        # Initialize arrs for data (flux or residuals) and weights
        data_only_star_coherent_lr = np.full((len(wave_star_coherent_lr), specrvprob.n_spec), np.nan)
        weights_coherent_lr = np.ones((specrvprob.data[0].flux.size, specrvprob.n_spec))

        # Vecs for lsq
        waves_vec = []
        data_vec = []
        weights_vec = []
        
        # Loop over data and build waves, data, and weights for median filter on both a common and unique grid
        for ispec in range(specrvprob.n_spec):

            # Initial weights
            _weights_vec = np.ones(specrvprob.data[0].flux.size)

            # Continue if not good
            if not specrvprob.data[ispec].is_good:
                weights_lr[:, ispec] = 0
                continue
                
            # Best fit pars
            pars = specrvprob.opt_results[ispec, iter_index]["pbest"]

            # Init
            spectral_model = copy.deepcopy(specrvprob.spectral_model)
            spectral_model.initialize(pars, specrvprob.data[ispec], iter_index, specrvprob.stellar_templates)

            # Vel to shift by
            if spectral_model.star.from_flat:
                vel = specrvprob.data[ispec].bc_vel
            else:
                vel = -1 * pars[spectral_model.star.par_names[0]].value
        
            # Generate the low res model
            wave_data, model_lr = spectral_model.build(pars)

            # If ideconv, store residuals, else, store flux after division of continuum, tellurics and gas cell
            if self.ideconv:

                # Simple data - model (no masking done here)
                residuals_lr = specrvprob.data[ispec].flux - model_lr

                # Interpolate and shift to common wavelength frame (unique)
                _wave, _data = pcmath.doppler_shift_flux(wave_data, residuals_lr, vel, wave_out=None)

                # Add to vecs
                waves_vec += _wave.tolist()
                data_vec += _data.tolist()

                # Interpolate and shift to common wavelength frame (common)
                data_only_star_coherent_lr[:, ispec] = pcmath.lin_interp(_wave, _data, wave_star_coherent_lr)

            else:

                # Copy flux
                flux_mod = np.copy(specrvprob.data[ispec].flux)

                # Continuum
                if hasattr(spectral_model, "continuum") and spectral_model.continuum is not None:
                    flux_mod /= spectral_model.continuum.build(pars, wave_data)

                # Tellurics
                if hasattr(spectral_model, "tellurics") and spectral_model.tellurics is not None:
                    flux_mod /= spectral_model.tellurics.build(pars, spectral_model.templates_dict['tellurics'], wave_data)

                # Gas Cell
                if hasattr(spectral_model, "gas_cell") and spectral_model.gas_cell is not None:
                    flux_mod /= spectral_model.continuum.build(pars, spectral_model.templates_dict['gas_cell'], wave_data)

                # Interpolate and shift to common wavelength frame (unique)
                _wave, _data = pcmath.doppler_shift_flux(wave_data, flux_mod, vel, wave_out=None)

                # Add to vecs
                waves_vec += _wave.tolist()
                data_vec += _data.tolist()

                # Interpolate and shift to common wavelength frame (common)
                data_only_star_coherent_lr[:, ispec] = pcmath.lin_interp(_wave, _data, wave_star_coherent_lr)

            
            # Telluric weights
            if self.weight_tellurics:

                # Build telluric spectrum
                tell_flux = spectral_model.tellurics.build(pars, specrvprob.spectral_model.templates_dict["tellurics"], wave_data)

                # Convolve
                if self.ideconv is not None and hasattr(specrvprob.spectral_model, "lsf") and specrvprob.spectral_model.lsf is not None:
                    tell_flux = specrvprob.spectral_model.lsf.convolve_flux(tell_flux, pars)

                # Doppler shift
                tell_wave, tell_flux = pcmath.doppler_shift_flux(wave_data, tell_flux, vel, wave_out=None)

                # Weights are flux^2 (more flux => more weight)
                tell_weights = tell_flux**2

                # Combine with decoherent weights
                _weights_vec *= tell_weights

                # Combine with coherent weights
                weights_coherent_lr[:, ispec] *= tell_weights

            # Mask weights and RMS weights
            if self.weight_fits:
                fit_weight = 1 / np.abs(specrvprob.opt_results[ispec, iter_index]["fbest"])
                weights_coherent_lr[:, ispec] *= fit_weight
                _weights_vec *= fit_weight

            
            # Final weights vector
            weights_vec += _weights_vec.tolist()

        # Flag bad pixels
        data_median = np.full(data_only_star_coherent_lr.shape[0], np.nan)
        for ix in range(data_only_star_coherent_lr.shape[0]):

            # Data and weights
            yy, ww = data_only_star_coherent_lr[ix, :], weights_coherent_lr[ix, :]
            good = np.where(np.isfinite(yy) & (ww > 0))[0]

            # Check each case
            if good.size >= 0:
                ymed = pcmath.weighted_median(yy[good], ww[good])
                data_median[ix] = ymed
        
        # Compute a cubic spline object from the median fit
        good = np.where(np.isfinite(wave_star_coherent_lr) & np.isfinite(data_median))[0]
        cspline_median = scipy.interpolate.CubicSpline(wave_star_coherent_lr[good], data_median[good], extrapolate=False)

        # Flag in vecs
        waves_vec = np.array(waves_vec)
        data_vec = np.array(data_vec)
        weights_vec = np.array(weights_vec)
        res = data_vec - cspline_median(waves_vec)
        use = np.where(np.isfinite(res) & (res > 0) & (weights_vec > 0) & np.isfinite(data_vec))[0]
        bad = np.where(np.abs(res) > 4 * np.nanstd(res))[0]
        weights_vec[bad] = 0

        # Final fit
        cspline = self.cspline_fit(waves_vec, data_vec, wave_sampling, weights_vec)

        # Compute new stellar template flux
        if self.ideconv:
            new_flux = specrvprob.spectral_model.templates_dict['star'][:, 1] + cspline(specrvprob.spectral_model.templates_dict['star'][:, 0])
        else:
            new_flux = cspline(specrvprob.spectral_model.templates_dict['star'][:, 0])

        # Bad
        bad = np.where(new_flux > self.max_thresh)[0]
        new_flux[bad] = self.max_thresh

        # Update flux
        specrvprob.spectral_model.templates_dict['star'][:, 1] = new_flux



    @staticmethod
    def cspline_fit(waves_vec, data_vec, wave_sampling, weights_vec=None):
        """Performs the actual cubic spline least squares fitting routine via scipy.

        Args:
            wave_vec (np.ndarray): The wavelength vector.
            data_vec (np.ndarray): The data vector (either flux or residuals) of the same length as wave_vec
            wave_sampling (float): The sampling of wavelength per detector pixel (Angstroms / detector pixel). This is used to generate the knot sampling and may be approximate.
            weight_vec (np.ndarray): The weights vector of the same length as wave_vec.

        Returns:
            (scipy.interpolate.LSQUnivariateSpline): The nominal fitted scipy cubic spline object.
        """

        # Weights
        if weights_vec is None:
            weights_vec = np.ones(len(waves_vec), dtype=float)
        else:
            weights_vec = np.copy(weights_vec)
        
        # Waves and data
        waves_vec = np.copy(waves_vec)
        data_vec = np.copy(data_vec)

        # Flag bad vals
        good = np.where(np.isfinite(waves_vec) & np.isfinite(data_vec) & np.isfinite(weights_vec) & (weights_vec > 0))[0]
        if good.size == 0:
            raise ValueError("No good data found in data to augment templates.")
        waves, data, weights = waves_vec[good], data_vec[good], weights_vec[good]

        # Sort
        ss = np.argsort(waves)
        waves, data, weights = waves[ss], data[ss], weights[ss]

        # Region
        wi, wf = waves.min(), waves.max()
        n_knots_init = int((wf - wi) / wave_sampling) # estimate, close enough to determine appropriate knot sampling for a large enough chunk where the sampling is nearly constant relative to the wavelengths

        # Knot points are roughly the detector grid.
        # Ensure the data surrounds the knots.
        knots = np.linspace(wi + 0.001, wf - 0.001, num=n_knots_init)
    
        # Remove bad knots
        bad_knots = []
        for i in range(100):
            for iknot in range(len(knots) - 1):
                n = np.where((waves > knots[iknot]) & (waves < knots[iknot+1]))[0].size
                if n == 0:
                    bad_knots.append(iknot)
            if len(bad_knots) > 0:
                knots = np.delete(knots, bad_knots)
            else:
                break

        # Normalize weights (probably not necessary)
        weights /= np.nansum(weights)

        # Fit
        spline_fitter = scipy.interpolate.LSQUnivariateSpline(waves, data, t=knots, w=weights, k=3, ext=1)
        
        # Return
        return spline_fitter





# TEMP FOR OLDER CODES, NEED TO REMOVE.
# A weighted median is more robust but may not be sufficient in providing robust "average" fluxes.
# With a median, the zero point heavily favors a certain range of dates (which themselves likely exhibit correlated noise)
class WeightedMedian(TemplateAugmenter):
    pass

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
        residuals_mean = np.zeros(nx)
        residuals_median = np.zeros(nx)
        residuals = np.full((nx, specrvprob.n_spec), np.nan)
        weights = np.zeros(shape=(nx, specrvprob.n_spec), dtype=float)

        # Loop over spectra
        for ispec in range(specrvprob.n_spec):

            # Continue if not good
            if not specrvprob.data[ispec].is_good:
                continue
            
            # Best fit pars
            pars = specrvprob.opt_results[ispec, iter_index]["pbest"]
        
            # Init the chunk
            specrvprob.spectral_model.initialize(pars, specrvprob.data[ispec], iter_index, specrvprob.stellar_templates)
        
            # Generate the low res model
            wave_data, model_lr = specrvprob.spectral_model.build(pars)
            
            # Residuals
            residuals_lr = specrvprob.data[ispec].flux - model_lr

            # Shift to a pseudo rest frame
            if specrvprob.spectral_model.star.from_flat:
                vel = specrvprob.data[ispec].bc_vel
            else:
                vel = -1 * pars[specrvprob.spectral_model.star.par_names[0]].value

            wave_star_rest = pcmath.doppler_shift_wave(wave_data, vel)
            residuals[:, ispec] = pcmath.cspline_interp(wave_star_rest, residuals_lr, current_stellar_template[:, 0])

            # Telluric weights, must doppler shift them as well.
            if self.weight_tellurics:
                tell_flux = specrvprob.spectral_model.tellurics.build(pars, specrvprob.spectral_model.templates_dict["tellurics"], wave_data)
                if specrvprob.spectral_model.lsf is not None:
                    tell_flux = specrvprob.spectral_model.lsf.convolve_flux(tell_flux, pars)
                _, tell_flux = pcmath.doppler_shift_flux(wave_data, tell_flux, vel)
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


        # Co-add residuals. First do a weighted median and flag 4 sigma deviations.
        # 1. If all weights at a given pixel are zero, set median value to zero.
        # 2. If there's more than one spectrum, compute the weighted median
        # 3. If there's only one spectrum, use those residuals, unless it's nan.
        for ix in range(nx):
            ww, rr = weights[ix, :], residuals[ix, :]
            ww = np.ones(len(rr))
            bad = np.where(~np.isfinite(rr))[0]
            ww[bad] = 0
            if np.nansum(ww) == 0:
                residuals_median[ix] = 0
            else:
                good = np.where((ww > 0) & np.isfinite(ww))[0]
                if good.size == 0:
                    residuals_median[ix] = 0
                elif good.size == 1:
                    residuals_median[ix] = rr[good[0]]
                else:
                    residuals_median[ix] = pcmath.weighted_median(rr, ww)

        weights_new[ix] = 1 / (residuals - np.outer(residuals_median, np.ones(specrvprob.n_spec)))**2
        bad = np.where(~np.isfinite(weights_new))
        weights_new[bad] = 0
        for ix in range(nx):
            ww, rr = weights_new[ix, :], residuals[ix, :]
            if np.nansum(ww) == 0:
                residuals_mean[ix] = 0
            else:
                good = np.where((ww > 0) & np.isfinite(ww))[0]
                if good.size == 0:
                    residuals_mean[ix] = 0
                elif good.size == 1:
                    residuals_mean[ix] = rr[good[0]]
                else:
                    residuals_mean[ix] = pcmath.weighted_mean(rr, ww)

        # Change any nans to zero just in case
        bad = np.where(~np.isfinite(residuals_mean))[0]
        if bad.size > 0:
            residuals_mean[bad] = 0

        # Augment the template
        new_flux = current_stellar_template[:, 1] + residuals_mean
        
        # Force the max to be less than given thresh.
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