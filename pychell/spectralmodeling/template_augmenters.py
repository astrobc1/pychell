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
try:
    import torch
except:
    warnings.warn("Could not import pytorch!")
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
                
            wave_star_rest += pcmath.doppler_shift(wave_data, vel, flux=None, wave_out=None, interp=None).tolist()

            # Telluric weights, must doppler shift them as well.
            if self.downweight_tellurics:
                tell_flux = specrvprob.spectral_model.tellurics.build(pars, specrvprob.spectral_model.templates_dict["tellurics"], wave_data)
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
                tell_flux = pcmath.doppler_shift(wave_data, vel, flux=tell_flux)
                tell_weights = tell_flux**2
            
            # Final weights
            if self.downweight_tellurics:
                weights_lr = specrvprob.data[ispec].mask * fit_weights[ispec] * tell_weights
            else:
                weights_lr = specrvprob.data[ispec].mask * fit_weights[ispec]
            weights_hr = pcmath.cspline_interp(wave_data, weights_lr, current_stellar_template[:, 0])
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
                    residuals_median[ix] = pcmath.weighted_median(rr, weights=ww)

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
                tell_flux = pcmath.doppler_shift(wave_data, vel, flux=tell_flux)
                tell_weights = tell_flux**2
            
            # Final weights
            weights_lr = specrvprob.data[ispec].mask * fit_weights[ispec] * tell_weights
            weights_hr = pcmath.cspline_interp(wave_data, weights_lr, current_stellar_template[:, 0])
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

# Uses pytorch to optimize the template
# def global_fit(forward_models, iter_index=None):
#     """Similar to Wobble, this will update the stellar template and lab frames by performing a gradient-based optimization via ADAM in pytorch considering all observations. Here, the template(s) are a parameter of thousands of points. The star is implemented in such a way that it will modify the current template, Doppler shift each observation and multiply them into the model. The lab frame is implemented in such a way that it will modify a zero based array, and then add this into each observation before convolution is performed.

#     Args:
#         forward_models (ForwardModels): The list of forward model objects
#         iter_index (int): The iteration to use.
#         templates_to_optimize (list of strings): valid entries are 'star', 'lab'.
#         nights_for_template (None): May be set, but not used here.
#     """
    
#     # The number of lsf points
#     n_lsf_pts = forward_models[0].models_dict['lsf'].nx_lsf
    
#     # The high resolution master grid (also stellar template grid)
#     wave_hr_master = torch.from_numpy(forward_models.templates_dict['star'][:, 0])
    
#     # Grids to optimize
#     if 'star' in templates_to_optimize:
#         star_flux = torch.nn.Parameter(torch.from_numpy(forward_models.templates_dict['star'][:, 1].astype(np.float64)))
        
#         # The current best fit stellar velocities
#         star_vels = torch.from_numpy(np.array([forward_models[ispec].best_fit_pars[-1][forward_models[ispec].models_dict['star'].par_names[0]].value for ispec in range(forward_models.n_spec)]).astype(np.float64))
#     else:
#         star_flux = None
#         star_vels = None
        
#     if 'lab' in templates_to_optimize:
#         residual_lab_flux = torch.nn.Parameter(torch.zeros(wave_hr_master.size()[0], dtype=torch.float64) + 1E-4)
#     else:
#         residual_lab_flux = None
    
#     # The partial built forward model flux
#     base_flux_models = torch.empty((forward_models.templates_dict['star'][:, 0].size, forward_models.n_spec), dtype=torch.float64)
    
#     # The best fit LSF for each spec (optional)
#     if 'lsf' in forward_models[0].models_dict and forward_models[0].models_dict['lsf'].enabled:
#         lsfs = torch.empty((n_lsf_pts, forward_models.n_spec), dtype=torch.float64)
#     else:
#         lsfs = None
    
#     # The data flux
#     data_flux = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)
    
#     # Bad pixel arrays for the data
#     badpix = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)
    
#     # The wavelength solutions
#     waves_lr = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)
    
#     # Weights, may just be binary mask
#     weights = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)

#     # Loop over spectra and extract to the above arrays
#     for ispec in range(forward_models.n_spec):

#         # Best fit pars
#         pars = forward_models[ispec].opt_results[-1][0]

#         if 'star' in templates_to_optimize:
#             x, y = forward_models[ispec].build_hr_nostar(pars, iter_index)
#         else:
#             x, y = forward_models[ispec].build_hr(pars, iter_index)
            
#         waves_lr[:, ispec], base_flux_models[:, ispec] = torch.from_numpy(x), torch.from_numpy(y)

#         # Fetch lsf and flip for torch. As of now torch does not support the negative step
#         if 'lsf' in forward_models[0].models_dict and forward_models[0].models_dict['lsf'].enabled:
#             lsfs[:, ispec] = torch.from_numpy(forward_models[ispec].models_dict['lsf'].build(pars))
#             lsfs[:, ispec] = torch.from_numpy(np.flip(lsfs[:, ispec].numpy(), axis=0).copy())

#         # The data and weights, change bad vals to nan
#         data_flux[:, ispec] = torch.from_numpy(np.copy(forward_models[ispec].data.flux))
#         weights[:, ispec] = torch.from_numpy(np.copy(forward_models[ispec].data.mask))
#         bad = torch.where(~torch.isfinite(data_flux[:, ispec]) | ~torch.isfinite(weights[:, ispec]) | (weights[:, ispec] <= 0))[0]
#         if len(bad) > 0:
#             data_flux[bad, ispec] = np.nan
#             weights[bad, ispec] = 0

#     # CPU or GPU
#     if torch.cuda.is_available():
#         torch.device('cuda')
#     else:
#         torch.device('cpu')

#     # Create the Torch model object
#     model = TemplateOptimizer(base_flux_models, waves_lr, weights, data_flux, wave_hr_master, star_flux=star_flux, star_vels=star_vels, residual_lab_flux=residual_lab_flux, lsfs=lsfs)

#     # Create the Adam optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#     print('Optimizing Template(s) ...', flush=True)

#     for epoch in range(500):

#         # Generate the model
#         optimizer.zero_grad()
#         loss = model.forward()

#         # Back propagation (gradient calculation)
#         loss.backward()
#         optimizer.step()

#         if (epoch + 1) % 10 == 0:
#             print('epoch {}, loss {}'.format(epoch + 1, loss.item()), flush=True)

#     if 'star' in templates_to_optimize:
#         new_star_flux = model.star_flux.detach().numpy()
#         locs = np.where(new_star_flux > 1)[0]
#         if locs.size > 0:
#             new_star_flux[locs] = 1
#         forward_models.templates_dict['star'][:, 1] = new_star_flux

#     if 'lab' in templates_to_optimize:
#         residual_lab_flux_fit = model.residual_lab_flux.detach().numpy()
#         forward_models.templates_dict['residual_lab'] = np.array([np.copy(forward_models.templates_dict['star'][:, 0]), residual_lab_flux_fit]).T
        
        
# # Class to optimize the forward model
# if 'torch' in sys.modules:
#     class TemplateOptimizer(torch.nn.Module):

#         def __init__(self, base_flux_models, waves_lr, weights, data_flux, wave_hr_master, star_flux=None, star_vels=None, residual_lab_flux=None, lsfs=None):
            
#             # Parent init
#             super(TemplateOptimizer, self).__init__()
            
#             # Shape information
#             self.nx_data, self.n_spec = data_flux.shape
            
#             # High res master wave grid for all observations
#             self.wave_hr_master = wave_hr_master # shape=(nmx,)
            
#             # The number of model pixels
#             self.nx_model = self.wave_hr_master.size()[0]
            
#             # Actual data
#             self.data_flux = data_flux # shape=(nx, n_spec)

#             # Potential parameters to optimize
#             self.star_flux = star_flux # coherence in stellar (quasi) rest frame, shape=(nmx,)
#             self.residual_lab_flux = residual_lab_flux # coherence in lab frame, shape=(nmx,)
            
#             # The base flux
#             self.base_flux_models = base_flux_models # shape=(nx, n_spec)
            
#             # Current wavelength solutions
#             self.waves_lr = waves_lr # shape=(nx, n_spec)
            
#             # Current best fit stellar velocities
#             self.star_vels = star_vels # shape=(n_spec,)
            
#             # Optimization weights
#             self.weights = weights # shape=(nx, n_spec)
            
#             # The lsf (optional)
#             if lsfs is not None:
#                 self.nx_lsf = lsfs.shape[0]
#                 self.n_pad_model = int(np.floor(self.nx_lsf / 2))
#                 self.lsfs = torch.ones(1, 1, self.nx_lsf, self.n_spec, dtype=torch.float64)
#                 self.lsfs[0, 0, :, :] = lsfs
#             else:
#                 self.lsfs = None

#         def forward(self):
        
#             # Stores all low res models
#             models_lr = torch.empty((self.nx_data, self.n_spec), dtype=torch.float64) + np.nan
            
#             # Loop over observations
#             for ispec in range(self.n_spec):
                
#                 # Doppler shift the stellar wavelength grid used for this observation.
#                 if self.star_flux is not None and self.residual_lab_flux is not None:
#                     wave_hr_star_shifted = self.wave_hr_master * torch.exp(self.star_vels[ispec] / cs.c)
#                     star = self.Interp1d()(wave_hr_star_shifted, self.star_flux, self.wave_hr_master).flatten()
#                     model = self.base_flux_models[:, ispec] * star + self.residual_lab_flux
#                 elif self.star_flux is not None and self.residual_lab_flux is None:
#                     wave_hr_star_shifted = self.wave_hr_master * torch.exp(self.star_vels[ispec] / cs.c)
#                     star = self.Interp1d()(wave_hr_star_shifted, self.star_flux, self.wave_hr_master).flatten()
#                     model = self.base_flux_models[:, ispec] * star
#                 else:
#                     model = self.base_flux_models[:, ispec] + self.residual_lab_flux
                    
#                 # Convolution. NOTE: PyTorch convolution is a pain in the ass
#                 # Second NOTE: Anything in PyTorch is a pain in the ass
#                 # Third NOTE: Wtf kind of language does the ML community even speak?
#                 if self.lsfs is not None:
#                     model_p = torch.ones((1, 1, self.nx_model + 2 * self.n_pad_model), dtype=torch.float64)
#                     model_p[0, 0, self.n_pad_model:-self.n_pad_model] = model
#                     conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
#                     conv.weight.data = self.lsfs[:, :, :, ispec]
#                     model = conv(model_p).flatten()

#                 # Interpolate onto data grid
#                 good = torch.where(torch.isfinite(self.weights[:, ispec]) & (self.weights[:, ispec] > 0))[0]
#                 models_lr[good, ispec] = self.Interp1d()(self.wave_hr_master, model, self.waves_lr[good, ispec])

#             # Weighted RMS
#             good = torch.where(torch.isfinite(self.weights) & (self.weights > 0) & torch.isfinite(models_lr))
#             wdiffs2 = (models_lr[good] - self.data_flux[good])**2 * self.weights[good]
#             loss = torch.sqrt(torch.sum(wdiffs2) / (torch.sum(self.weights[good])))

#             return loss
        
        
#         @staticmethod
#         def h_poly_helper(tt):
#             A = torch.tensor([
#                 [1, 0, -3, 2],
#                 [0, 1, -2, 1],
#                 [0, 0, 3, -2],
#                 [0, 0, -1, 1]
#                 ], dtype=tt[-1].dtype)
#             return [sum(A[i, j]*tt[j] for j in range(4)) for i in range(4)]
            
#         @classmethod
#         def h_poly(cls, t):
#             tt = [ None for _ in range(4) ]
#             tt[0] = 1
#             for i in range(1, 4):
#                 tt[i] = tt[i-1]*t
#             return cls.h_poly_helper(tt)

#         @classmethod
#         def H_poly(cls, t):
#             tt = [ None for _ in range(4) ]
#             tt[0] = t
#             for i in range(1, 4):
#                 tt[i] = tt[i-1]*t*i/(i+1)
#             return cls.h_poly_helper(tt)

#         @classmethod
#         def interpcs(cls, x, y, xs):
#             m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
#             m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
#             I = np.searchsorted(x[1:], xs)
#             dx = (x[I+1]-x[I])
#             hh = cls.h_poly((xs-x[I])/dx)
#             return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

#         class Interp1d(torch.autograd.Function):
#             def __call__(self, x, y, xnew, out=None):
#                 return self.forward(x, y, xnew, out)

#             def forward(ctx, x, y, xnew, out=None):

#                 # making the vectors at least 2D
#                 is_flat = {}
#                 require_grad = {}
#                 v = {}
#                 device = []
#                 for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
#                     assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
#                                                 'at most 2-D.'
#                     if len(vec.shape) == 1:
#                         v[name] = vec[None, :]
#                     else:
#                         v[name] = vec
#                     is_flat[name] = v[name].shape[0] == 1
#                     require_grad[name] = vec.requires_grad
#                     device = list(set(device + [str(vec.device)]))
#                 assert len(device) == 1, 'All parameters must be on the same device.'
#                 device = device[0]

#                 # Checking for the dimensions
#                 assert (v['x'].shape[1] == v['y'].shape[1]
#                         and (
#                             v['x'].shape[0] == v['y'].shape[0]
#                             or v['x'].shape[0] == 1
#                             or v['y'].shape[0] == 1
#                             )
#                         ), ("x and y must have the same number of columns, and either "
#                             "the same number of row or one of them having only one "
#                             "row.")

#                 reshaped_xnew = False
#                 if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
#                 and (v['xnew'].shape[0] > 1)):
#                     # if there is only one row for both x and y, there is no need to
#                     # loop over the rows of xnew because they will all have to face the
#                     # same interpolation problem. We should just stack them together to
#                     # call interp1d and put them back in place afterwards.
#                     original_xnew_shape = v['xnew'].shape
#                     v['xnew'] = v['xnew'].contiguous().view(1, -1)
#                     reshaped_xnew = True

#                 # identify the dimensions of output and check if the one provided is ok
#                 D = max(v['x'].shape[0], v['xnew'].shape[0])
#                 shape_ynew = (D, v['xnew'].shape[-1])
#                 if out is not None:
#                     if out.numel() != shape_ynew[0]*shape_ynew[1]:
#                         # The output provided is of incorrect shape.
#                         # Going for a new one
#                         out = None
#                     else:
#                         ynew = out.reshape(shape_ynew)
#                 if out is None:
#                     ynew = torch.zeros(*shape_ynew, dtype=y.dtype, device=device)

#                 # moving everything to the desired device in case it was not there
#                 # already (not handling the case things do not fit entirely, user will
#                 # do it if required.)
#                 for name in v:
#                     v[name] = v[name].to(device)

#                 # calling searchsorted on the x values.
#                 #ind = ynew
#                 #searchsorted(v['x'].contiguous(), v['xnew'].contiguous(), ind)
#                 ind = np.searchsorted(v['x'].contiguous().numpy().flatten(), v['xnew'].contiguous().numpy().flatten())
#                 ind = torch.tensor(ind)
#                 # the `-1` is because searchsorted looks for the index where the values
#                 # must be inserted to preserve order. And we want the index of the
#                 # preceeding value.
#                 ind -= 1
#                 # we clamp the index, because the number of intervals is x.shape-1,
#                 # and the left neighbour should hence be at most number of intervals
#                 # -1, i.e. number of columns in x -2
#                 ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1).long()

#                 # helper function to select stuff according to the found indices.
#                 def sel(name):
#                     if is_flat[name]:
#                         return v[name].contiguous().view(-1)[ind]
#                     return torch.gather(v[name], 1, ind)

#                 # activating gradient storing for everything now
#                 enable_grad = False
#                 saved_inputs = []
#                 for name in ['x', 'y', 'xnew']:
#                     if require_grad[name]:
#                         enable_grad = True
#                         saved_inputs += [v[name]]
#                     else:
#                         saved_inputs += [None, ]
#                 # assuming x are sorted in the dimension 1, computing the slopes for
#                 # the segments
#                 is_flat['slopes'] = is_flat['x']
#                 # now we have found the indices of the neighbors, we start building the
#                 # output. Hence, we start also activating gradient tracking
#                 with torch.enable_grad() if enable_grad else contextlib.suppress():
#                     v['slopes'] = (
#                             (v['y'][:, 1:]-v['y'][:, :-1])
#                             /
#                             (v['x'][:, 1:]-v['x'][:, :-1])
#                         )

#                     # now build the linear interpolation
#                     ynew = sel('y') + sel('slopes')*(
#                                             v['xnew'] - sel('x'))

#                     if reshaped_xnew:
#                         ynew = ynew.view(original_xnew_shape)

#                 ctx.save_for_backward(ynew, *saved_inputs)
#                 return ynew

#             @staticmethod
#             def backward(ctx, grad_out):
#                 inputs = ctx.saved_tensors[1:]
#                 gradients = torch.autograd.grad(
#                                 ctx.saved_tensors[0],
#                                 [i for i in inputs if i is not None],
#                                 grad_out, retain_graph=True)
#                 result = [None, ] * 5
#                 pos = 0
#                 for index in range(len(inputs)):
#                     if inputs[index] is not None:
#                         result[index] = gradients[pos]
#                         pos += 1
#                 return (*result,)
        

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