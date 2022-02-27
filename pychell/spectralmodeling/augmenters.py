# Base Python
import copy

# Pychell deps
import pychell.maths as pcmath

# Maths
import numpy as np
import scipy.interpolate

class TemplateAugmenter:
    
    def __init__(self, weight_tellurics=False, max_thresh=None, weight_fits=True):
        self.weight_tellurics = weight_tellurics
        self.max_thresh = max_thresh
        self.weight_fits = weight_fits
        
    def augment_template(self, specrvprob, iter_index):
        raise NotImplementedError(f"Must implement the method augment_templates for {self.__class__.__name__}")


class WeightedMeanAugmenter(TemplateAugmenter):

    def augment_template(self, specrvprob, opt_results, iter_index):

        # Unpack the current stellar template
        current_stellar_template = np.copy(specrvprob.model.templates["star"])
        
        # Get the fit metric
        fit_metrics = np.full(len(specrvprob), np.nan)
        for ispec in range(len(specrvprob)):
            fit_metrics[ispec] = np.abs(opt_results[ispec]["fbest"])
        
        # Weights according to fit metric
        fit_weights = 1 / fit_metrics**2
        good = np.where(np.isfinite(fit_weights))[0]
        bad = np.where(~np.isfinite(fit_weights) | (fit_weights > 200))[0]
        if good.size == 0:
            fit_weights = np.ones(len(specrvprob))

        # Storage arrays
        nx  = len(current_stellar_template)
        residuals_mean = np.zeros(nx)
        residuals_median = np.zeros(nx)
        residuals = np.full((nx, len(specrvprob)), np.nan)
        weights = np.zeros(shape=(nx, len(specrvprob)), dtype=float) 

        # Loop over spectra
        for ispec in range(len(specrvprob)):

            try:

                # Best fit pars
                pars = opt_results[ispec]["pbest"]
        
                # Generate the low res model
                wave_data, model_lr = specrvprob.model.build(pars, specrvprob.data[ispec])
            
                # Residuals
                residuals_lr = specrvprob.data[ispec].flux - model_lr

                # Shift to a pseudo rest frame
                if specrvprob.model.star.from_flat:
                    vel = float(specrvprob.data[ispec].header["bc_vel"])
                else:
                    vel = -1 * pars[specrvprob.model.star.par_names[0]].value

                wave_star_rest = pcmath.doppler_shift_wave(wave_data, vel)
                residuals[:, ispec] = pcmath.cspline_interp(wave_star_rest, residuals_lr, specrvprob.model.templates['wave'])
                weights_lr = specrvprob.data[ispec].mask * fit_weights[ispec]

                # Telluric weights, must doppler shift them as well.
                if self.weight_tellurics:
                    tell_flux = specrvprob.model.tellurics.build(pars, specrvprob.model.templates["tellurics"])
                    if specrvprob.model.lsf is not None:
                        tell_flux = specrvprob.model.lsf.convolve(tell_flux, pars=pars)
                    tell_flux = pcmath.doppler_shift_flux(wave_data, tell_flux, vel)
                    tell_weights = tell_flux**2
                    weights_lr *= tell_weights
            
            except:

                weights_lr = specrvprob.data[ispec].mask * fit_weights[ispec]
            
            # Interpolate to a high res grid
            weights_hr = pcmath.lin_interp(wave_star_rest, weights_lr, specrvprob.model.templates['wave'])
            bad = np.where((weights_hr < 0) | ~np.isfinite(weights_hr))[0]
            weights_hr[bad] = 0
            weights[:, ispec] = weights_hr

        
        # Sync
        bad = np.where(~np.isfinite(residuals) | (weights <= 0))
        residuals[bad] = np.nan
        weights[bad] = 0

        # Co-add residuals. First do a weighted median and flag 4 sigma deviations.
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
                    residuals_median[ix] = pcmath.weighted_median(rr, ww)

        # Flag outliers
        weights_final = np.copy(weights)
        for ix in range(nx):
            ww, rr = weights[ix, :], residuals[ix, :]
            n_good = np.where(ww > 0)[0].size
            if n_good >= 5:
                bad = np.where(residuals[ix, :] > 4 * np.nanstd(residuals[ix, :] - residuals_median[ix]))[0]
                weights_final[ix, bad] = 0
                residuals[ix, bad] = np.nan

        
        # Sync
        bad = np.where(~np.isfinite(residuals) | (weights_final <= 0))
        residuals[bad] = np.nan
        weights_final[bad] = 0

        # Final loop
        for ix in range(nx):
            ww, rr = weights_final[ix, :], residuals[ix, :]
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
        new_flux = current_stellar_template + residuals_mean
        
        # Force < 1
        bad = np.where(new_flux > 1)[0]
        if bad.size > 0:
            new_flux[bad] = 1
    
        # Update the template
        specrvprob.model.templates['star'] = new_flux
