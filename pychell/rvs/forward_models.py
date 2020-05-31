# Python built in modules
import copy
import glob # File searching
import os # Making directories
import importlib.util # importing other modules from files
import warnings # ignore warnings
import time # Time the code
import pickle
import inspect
import multiprocessing as mp # parallelization on a single node
import sys # sys utils
from sys import platform # plotting backend
from pdb import set_trace as stop # debugging

# Graphics
import matplotlib # to set the backend
import matplotlib.pyplot as plt # Plotting
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")
from matplotlib import cm

# Multiprocessing
from joblib import Parallel, delayed

# Science/math
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np # Math, Arrays
import scipy.interpolate # Cubic interpolation, Akima interpolation

# llvm
from numba import njit, jit, prange

# User defined
import pychell.maths as pcmath
import pychell.rvs.template_augmenter as pcaugmenter
import pychell.rvs.model_components as pcmodelcomponents
import pychell.rvs.data1d as pcdata
import pychell.rvs.target_functions as pctargetfuns
import pychell.utils as pcutils
import pychell.rvs.rvcalc as pcrvcalc

# Optimization
import optimparameters.parameters as OptimParameters
from neldermead.neldermead import NelderMead


# Stores all forward model objects useful wrapper to store all the forward model objects.
class ForwardModels(list):
        
    def __init__(self, forward_model_settings, model_blueprints, order_num):
        
        # Initiate the actual list
        super().__init__()
        
        # Auto-populate
        for key in forward_model_settings:
            setattr(self, key, copy.deepcopy(forward_model_settings[key]))
            
        # Overwrite the target function with the actual function to optimize the model
        self.target_function = getattr(pctargetfuns, self.target_function)
        
        # Overwrite the template augment function with the actual function to augment the template
        self.template_augmenter = getattr(pcaugmenter, self.template_augmenter)

        # The order number
        self.order_num = order_num

        # Initiate the data, models, and outputs
        
        self.init(forward_model_settings, model_blueprints)

        # Remaining instance members

        # The proper tag
        self.tag = self.star_name + '_' + self.tag

        # The number of iterations for rvs and template fits
        self.n_iters_rvs = self.n_template_fits
        self.n_iters_fits = self.n_iters_rvs + int(not self[0].models_dict['star'].from_synthetic)

        # Create output directories
        self.create_output_dirs(output_path_root=self.output_path_root)
        
        # Save the global parameters dictionary to the output directory
        with open(self.run_output_path + os.sep + 'global_parameters_dictionary.pkl', 'wb') as f:
            pickle.dump(forward_model_settings, f)

        # Print summary
        self.print_init_summary()


    # Updates spectral models according to best fit parameters
    def update_models(self, iter_num):
        
        # k1 = index for forward model array access    
        # k2 = Plot names for forward model objects
        # k3 = index for RV array access
        # k4 = RV plot names
        k1, k2, k3, k4 = self[0].iteration_indices(iter_num)

        for ispec in range(self.n_spec):

            # Pass the previous iterations best pars as starting points
            self[ispec].set_parameters(copy.deepcopy(self[ispec].best_fit_pars[k1]))
            
            # Ensure the same templates dict is shared amongst the models (just a pointer to a single instance)
            for fwm in self:
                fwm.templates_dict = self.templates_dict
            
            # Update other models
            for model in self[ispec].models_dict.keys():
                self[ispec].models_dict[model].update(self[ispec], iter_num)
                
                
    def init(self, forward_model_settings, model_blueprints):
        
        print('Loading in data and constructing forward model objects for order ' + str(self.order_num) + ' ...')
        
        # The input files
        input_files = [self.input_path + f for f in np.genfromtxt(self.input_path + self.flist_file, dtype='<U100').tolist()]
        
        self.n_spec = len(input_files)
        
        # The inidividual forward model object init
        fwm_class_init = eval(self.spectrograph + 'ForwardModel')

        # Init inidividual forward models
        for ispec in range(self.n_spec):
            self.append(fwm_class_init(input_files[ispec], forward_model_settings, model_blueprints, self.order_num, spec_num=ispec + 1))
            
        # Load templates
        self.load_templates()
        
        # Init the parameters
        self.init_parameters()
        
        # The number of spectra (may overwrite)
        self.n_spec = len(input_files)
        
        # Initiate the RV dicts
        self.init_rvs()
            
            
    def init_rvs(self):
        
        # The bary-center information
        if hasattr(self, 'bary_corr_file') and self.bary_corr_file is not None:
            self.BJDS, self.bc_vels = np.loadtxt(self.input_path + self.bary_corr_file, delimiter=',', unpack=True)
            for ispec in range(self.n_spec):
                self[ispec].data.set_bc_info(self.BJDS[ispec], self.bc_vels[ispec])
        else:
            bc_computer = self[0].data.__class__.calculate_bc_info_all
            self.BJDS, self.bc_vels = bc_computer(self, obs_name=self.observatory, star_name=self.star_name)
        
        # Compute the nightly BJDs and n obs per night
        self.BJDS_nightly, self.n_obs_nights = pcrvcalc.get_nightly_jds(self.BJDS)
        
        # Sort by BJD
        self.sort()
        
        # The number of nights
        self.n_nights = len(self.BJDS_nightly)
            
        # Storage array for RVs
        self.rvs_dict = {}
        
        # Nelder-Mead RVs
        self.rvs_dict['rvs'] = np.empty(shape=(self.n_spec, self.n_template_fits), dtype=np.float64)
        self.rvs_dict['rvs_nightly'] = np.empty(shape=(self.n_nights, self.n_template_fits), dtype=np.float64)
        self.rvs_dict['unc_nightly'] = np.empty(shape=(self.n_nights, self.n_template_fits), dtype=np.float64)
        
        # X Corr RVs
        if self.do_xcorr:
            
            self.rvs_dict['rvs_xcorr'] = np.empty(shape=(self.n_spec, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['rvs_xcorr_unc'] = np.empty(shape=(self.n_spec, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['rvs_xcorr_nightly'] = np.empty(shape=(self.n_nights, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['unc_xcorr_nightly'] = np.empty(shape=(self.n_nights, self.n_template_fits), dtype=np.float64)
            
            # Cross correlation velocity resolution
            self.n_xcorr_vels = int(2 * self.xcorr_range / self.xcorr_step)
            
            self.rvs_dict['xcorrs'] = np.empty(shape=(self.n_xcorr_vels, self.n_spec, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['xcorr_vels'] = np.empty(shape=(self.n_xcorr_vels, self.n_spec, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['line_bisectors'] = np.empty(shape=(self.n_bs, self.n_spec, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['bisector_spans'] = np.empty(shape=(self.n_spec, self.n_template_fits), dtype=np.float64)
        
    def sort(self):
        # Sort by BJD
        sorting_inds = np.argsort(self.BJDS)
        self.BJDS = self.BJDS[sorting_inds]
        self.bc_vels = self.bc_vels[sorting_inds]
        
        # Also sort the list
        for ispec in range(self.n_spec):
            self[ispec] = self[sorting_inds[ispec]]
    
    
    def init_parameters(self):
        for ispec in range(self.n_spec):
            self[ispec].init_parameters()
    
    # Optimize the initial guess parameters before a first iteration
    # Performs a crude x corr to estimate the stellar RV
    # Sanity checks remaining parameters
    def opt_init_params(self):

        # Handle the star in parallel.
        if self[0].models_dict['star'].from_synthetic:
            pcrvcalc.cross_correlate_all(self, 0)
            for ispec in range(self.n_spec):
                self[ispec].models_dict['star'].update(self[ispec], 0)
                
        # Lock any parameters with min_bound = max_bound
        for ispec in range(self.n_spec):
            self[ispec].initial_parameters.sanity_lock()
            self[ispec].crude = False

    # Stores the forward model outputs in .npz files for all iterations
    # Stores the RVs in a single .npz
    def save_results(self):
        
        # Save the RVs
        self.save_rvs()
        
        # For each spectrum, save the forward model results
        for ispec in range(self.n_spec):
            self[ispec].save_results(self.run_output_path_opt_results, self.run_output_path_spectral_fits)
        
    # Wrapper to fit all spectra
    def fit_spectra(self, iter_num):

        # Timer
        stopwatch = pcutils.StopWatch()
        
        # Get the solver wrapper, instance dependent
        solver_wrapper = self.solver_wrapper

        # Parallel fitting
        if self.n_cores > 1:

            # Construct the arguments
            args_pass = []
            kwargs_pass = []
            for spec_num in range(self.n_spec):
                args_pass.append((self[spec_num], iter_num, self.n_spec))
                kwargs_pass.append({'output_path_plot': self.run_output_path_spectral_fits, 'verbose_print': self.verbose_print, 'verbose_plot': self.verbose_plot})
            
            # Call the parallel job via joblib.
            self[:] = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(solver_wrapper)(*args_pass[ispec], **kwargs_pass[ispec]) for ispec in range(self.n_spec))

        else:
            # Fit one at a time
            for ispec in range(self.n_spec):
                print('    Performing Nelder-Mead Fit For Spectrum '  + str(ispec+1) + ' of ' + str(self.n_spec), flush=True)
                self[ispec] = solver_wrapper(self[ispec], iter_num, self.n_spec, output_path_plot=self.run_output_path_spectral_fits, verbose_print=self.verbose_print, verbose_plot=self.verbose_plot)
            
        # Fit in Parallel
        print('Fitting Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)
    
    # Outputs RVs and cross corr analyses for all iterations for a given order.
    def save_rvs(self):
        
        # Full filename
        fname = self.run_output_path_rvs + self.tag + '_rvs_ord' + str(self[0].order_num) + '.npz'
        
        # Save in a .npz file for easy access later
        np.savez(fname, BJDS=self.BJDS, BJDS_nightly=self.BJDS_nightly, n_obs_nights=self.n_obs_nights, **self.rvs_dict)

    # Loads the templates dictionary and stores in a dictionary.
    # A pointer to the templates dictionary is stored in each forward model class
    # It can be accessed via forward_models.templates_dict or forward_models[ispec].templates_dict
    def load_templates(self):
        self.templates_dict = self[0].load_templates()
        for fwm in self:
            fwm.templates_dict = self.templates_dict

    # Overwrite the getattr function
    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except:
            return getattr(self[0], key)
            

    # Create output directories
    # output_dir_root is the root output directory.
    def create_output_dirs(self, output_path_root):
        
        # Order folder
        o_folder = 'Order' + str(self.order_num) + os.sep
            
        # Output path for this run
        self.run_output_path = output_path_root + self.tag + os.sep
        
        # Output paths for this order
        self.run_output_path_rvs = self.run_output_path + o_folder + 'RVs' + os.sep
        self.run_output_path_spectral_fits = self.run_output_path + o_folder + 'Fits' + os.sep
        self.run_output_path_opt_results = self.run_output_path + o_folder + 'Opt' + os.sep
        self.run_output_path_stellar_templates = self.run_output_path + o_folder + 'Stellar_Templates' + os.sep
        
        # Create directories for this order
        os.makedirs(self.run_output_path_rvs, exist_ok=True)
        os.makedirs(self.run_output_path_spectral_fits, exist_ok=True)
        os.makedirs(self.run_output_path_opt_results, exist_ok=True)
        os.makedirs(self.run_output_path_stellar_templates, exist_ok=True)

    # Post init summary
    def print_init_summary(self):
        
        # Print summary
        print('***************************************', flush=True)
        print('** Target: ' + self.star_name, flush=True)
        print('** Spectrograph: ' + self.observatory + ' / ' + self.spectrograph, flush=True)
        print('** Observations: ' + str(self.n_spec) + ' spectra, ' + str(self.n_nights) + ' nights', flush=True)
        print('** Echelle Order: ' + str(self.order_num), flush=True)
        print('** TAG: ' + self.tag, flush=True)
        print('** N Iterations: ' + str(self.n_template_fits), flush=True)
        print('***************************************', flush=True)


    # Wrapper for parallel processing. Solves and plots the forward model results. Also does xcorr if set.
    @staticmethod
    def solver_wrapper(forward_model, iter_num, n_spec_tot, output_path_plot=None, verbose_print=False, verbose_plot=False):

        # Start the timer
        stopwatch = pcutils.StopWatch()
        
        # Construct the extra arguments to pass to the target function
        args_to_pass = (forward_model, iter_num)
        
        solver = NelderMead(forward_model.target_function, forward_model.initial_parameters, no_improve_break=3, args_to_pass=args_to_pass)
        opt_result = solver.solve()
            
        # k1 = index for forward model array access
        # k2 = Plot names for forward model objects
        # k3 = index for RV array access
        # k4 = RV plot names
        k1, k2, k3, k4 = forward_model.iteration_indices(iter_num)

        forward_model.best_fit_pars.append(opt_result[0])
        forward_model.opt.append(opt_result[1:])

        # Build the best fit forward model
        wave_grid_data, best_model = forward_model.build_full(forward_model.best_fit_pars[-1], iter_num)

        forward_model.wavelength_solutions.append(wave_grid_data)
        forward_model.models.append(best_model)

        # Compute the residuals between the data and model, don't flag bad pixels here. Cropped may still be nan.
        forward_model.residuals.append(forward_model.data.flux - best_model)

        # Print diagnostics if set
        if verbose_print:
            print('RMS = %' + str(round(100*opt_result[1], 5)))
            print('Function Calls = ' + str(opt_result[2]))
            forward_model.pretty_print()

        # Do a cross correlation analysis if set
        if forward_model.do_xcorr and forward_model.models_dict['star'].enabled:
            forward_model = pcrvcalc.cc_wrapper(forward_model, n_spec_tot, iter_num)
        
        print('    Fit Spectrum ' + str(forward_model.spec_num) + ' of ' + str(n_spec_tot) + ' in ' + str(round((stopwatch.time_since())/60, 2)) + ' min', flush=True)

        # Output a plot
        if output_path_plot is not None:
            forward_model.plot_model(iter_num, output_path=output_path_plot)

        # Return new forward model object since we possibly fit in parallel
        return forward_model

class NightlyForwardModels(ForwardModels):


    def fit_spectra(self, iter_num):
        
        # Timer
        stopwatch = pcutils.StopWatch()
        
        # Get the solver wrapper, instance dependent
        solver_wrapper = self.solver_wrapper

        # Parallel fitting
        if self.n_cores > 1:

            # Construct the arguments
            args_pass = []
            kwargs_pass = []
            for inight in range(self.n_nights):
                inds = self[0].get_all_spec_indices_from_night(inight, self.n_obs_nights)
                args_pass.append(([self[i] for i in inds], iter_num, self.n_spec, self.n_nights, self.n_obs_nights))
                kwargs_pass.append({'output_path_plot': self.run_output_path_spectral_fits, 'verbose_print': self.verbose_print, 'verbose_plot': self.verbose_plot})
            
            # Call the parallel job via joblib.
            presults = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(solver_wrapper)(*args_pass[inight], **kwargs_pass[inight]) for inight in range(self.n_nights))
            for inight in range(self.n_nights):
                inds = self[0].get_all_spec_indices_from_night(inight, self.n_obs_nights)
                for i in range(len(inds)):
                    self[inds[i]] = presults[inight][i]

        else:
            # Fit one night at a time
            for inight in range(self.n_nights):
                print('    Performing Nelder-Mead Fit For Night '  + str(inight+1) + ' of ' + str(self.n_nights), flush=True)
                inds = self[0].get_all_spec_indices_from_night(inight, self.n_obs_nights)
                result = solver_wrapper([copy.deepcopy(self[i]) for i in inds], iter_num, self.n_spec, self.n_nights, self.n_obs_nights, output_path_plot=self.run_output_path_spectral_fits, verbose_print=self.verbose_print, verbose_plot=self.verbose_plot)
                for i in range(len(inds)):
                    self[inds[i]] = result[i]
            
        # Fit in Parallel
        print('Fitting Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)
        

    # Wrapper for parallel processing. Solves and plots the forward models from a given night. Also does xcorr if set.
    @staticmethod
    def solver_wrapper(nightly_forward_models, iter_num, n_spec_tot, n_nights_tot, n_obs_nights, output_path_plot=None, verbose_print=False, verbose_plot=False):

        # Start the timer
        stopwatch = pcutils.StopWatch()
        
        # Construct the extra arguments to pass to the target function
        args_to_pass = (nightly_forward_models, iter_num)
        
        initial_parameters = OptimParameters.Parameters()
        models0 = copy.deepcopy(nightly_forward_models[0].models_dict)
        pnames_match = {}
        for model in models0:
            for pname in models0[model].par_names:
                par = models0[model].initial_parameters[pname]
                if par.commonality == 'nights':
                    # Single parameter for whole night
                    initial_parameters[pname] = copy.deepcopy(par)
                    pnames_match[pname] = pname
                else:
                    # Not shared, need unique names.
                    for ispec in range(len(nightly_forward_models)):
                        pname_temp = pname + '_spec_' + str(ispec + 1)
                        initial_parameters[pname_temp] = OptimParameters.Parameter(name=pname_temp, value=par.value, minv=par.minv, maxv=par.maxv, mcmcscale=par.mcmcscale, vary=par.vary, commonality=par.commonality)
                        k = nightly_forward_models[ispec].models_dict[model].par_names.index(pname)
                        nightly_forward_models[ispec].models_dict[model].par_names[k] = pname_temp
                        pnames_match[pname_temp] = pname
        
        optimizer = NelderMead(nightly_forward_models[0].target_function, initial_parameters, no_improve_break=3, args_to_pass=args_to_pass)
        opt_result = optimizer.solve()
            
        # k1 = index for forward model array access
        # k2 = Plot names for forward model objects
        # k3 = index for RV array access
        # k4 = RV plot names
        k1, k2, k3, k4 = nightly_forward_models[0].iteration_indices(iter_num)
        
        best_global_pars = opt_result[0]
        
        best_fit_pars = [OptimParameters.Parameters() for _ in range(len(nightly_forward_models))]
            
        # Unpack
        
        # Loop over models
        for model in nightly_forward_models[0].models_dict:
            # Loop over the params for this model
            model0_global = nightly_forward_models[0].models_dict[model]
            for pname_global in model0_global.par_names:
                
                # Possibly the first (_spec_1)
                par_global_fit = best_global_pars[pname_global]
                
                # Single parameter
                if par_global_fit.commonality == 'nights':
                    pname_single = pnames_match[pname_global]
                    for ispec in range(len(nightly_forward_models)):
                        best_fit_pars[ispec][pname_single] = copy.deepcopy(par_global_fit)
                
                # Not shared, has unique names
                else:
                    pname_single = pnames_match[pname_global]
                    for ispec in range(len(nightly_forward_models)):
                        par_unique = best_global_pars[pname_single + '_spec_' + str(ispec + 1)]
                        best_fit_pars[ispec][pname_single] = OptimParameters.Parameter(name=pname_single, value=par_unique.value, minv=par_unique.minv, maxv=par_unique.maxv, vary=par_unique.vary, mcmcscale=par_unique.mcmcscale, commonality=par_unique.commonality)
                        k = nightly_forward_models[ispec].models_dict[model].par_names.index(pname_single + '_spec_' + str(ispec + 1))
                        
                        nightly_forward_models[ispec].models_dict[model].par_names[k] = pname_single


        # Build the best fit forward model
        for ispec in range(len(nightly_forward_models)):
            
            nightly_forward_models[ispec].best_fit_pars.append(best_fit_pars[ispec])
            nightly_forward_models[ispec].opt.append(opt_result[1:])
            
            wave_grid_data, best_model = nightly_forward_models[ispec].build_full(nightly_forward_models[ispec].best_fit_pars[-1], iter_num)

            nightly_forward_models[ispec].wavelength_solutions.append(wave_grid_data)
            nightly_forward_models[ispec].models.append(best_model)

            # Compute the residuals between the data and model, don't flag bad pixels here. Cropped may still be nan.
            nightly_forward_models[ispec].residuals.append(nightly_forward_models[ispec].data.flux - best_model)

        
        # Do a cross correlation analysis if set
        if nightly_forward_models[0].do_xcorr and nightly_forward_models[0].models_dict['star'].enabled:
            for ispec in range(len(nightly_forward_models)):
                nightly_forward_models[ispec] = pcrvcalc.cc_wrapper(nightly_forward_models[ispec], n_spec_tot, iter_num)
        
        print('    Fit Night ' + str(nightly_forward_models[0].get_night_index(nightly_forward_models[0].spec_index, n_obs_nights) + 1) + ' of ' + str(n_nights_tot) + ' in ' + str(round((stopwatch.time_since())/60, 2)) + ' min', flush=True)

        # Output a plot
        if output_path_plot is not None:
            for ispec in range(len(nightly_forward_models)):
                nightly_forward_models[ispec].plot_model(iter_num, output_path=output_path_plot)

        # Return new forward model objects since we possibly fit in parallel
        return nightly_forward_models

class ForwardModel:
    
    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):
        
        # The echelle order
        self.order_num = order_num
        
        # The spectral number and index
        self.spec_num = spec_num
        self.spec_index = self.spec_num - 1
        
        # Auto-populate
        for key in forward_model_settings:
            if not hasattr(self, key):
                setattr(self, key, copy.deepcopy(forward_model_settings[key]))
                
        # The proper tag
        self.tag = self.star_name + '_' + self.tag
        
        # Overwrite the target function with the actual function to optimize the model
        self.target_function = getattr(pctargetfuns, self.target_function)
        
        # Init the data
        data_class_init = getattr(pcdata, 'SpecData' + forward_model_settings['spectrograph'])
        self.data = data_class_init(input_file, order_num=self.order_num, spec_num=self.spec_num, crop_pix=self.crop_data_pix, wave_direction=self.wave_direction)
        
        # Init the models
        self.init_models(forward_model_settings, model_blueprints)
        
        # Storage arrays after each iteration
        # Stores the final RMS [0] and target function calls [1]
        self.opt = []
        
        # Stores the best fit parameters (Parameter objects)
        self.best_fit_pars = []
        
        # Stores the wavelenth solutions (may just be copies if known a priori)
        self.wavelength_solutions = []
        
        # Stores the residuals
        self.residuals = []
        
        # Stores the best fit forward models (built from best_fit_pars)
        self.models = []
        
        # Cross correlation analysis is also stored here since it's performed in parallel
        if self.do_xcorr:
            
            # Cross correlation velocity resolution
            self.n_xcorr_vels = int(2 * self.xcorr_range / self.xcorr_step)
            
            # Stores the cross correlations
            self.xcorr_vels = np.empty(shape=(self.n_xcorr_vels, self.n_template_fits), dtype=np.float64)
            self.xcorrs = np.empty(shape=(self.n_xcorr_vels, self.n_template_fits), dtype=np.float64)
            
            # Stores the xcorr rvs. Nightly Xcorr RVs are not calculated, but can be by the user after.
            self.rvs_xcorr = np.empty(self.n_template_fits, dtype=np.float64)
            self.rvs_xcorr_nightly = np.empty(self.n_template_fits, dtype=np.float64)
            self.unc_xcorr_nightly = np.empty(self.n_template_fits, dtype=np.float64)
        
            # Stores the bisector spans
            self.line_bisectors = np.empty(shape=(self.n_bs, self.n_template_fits), dtype=np.float64)
            self.bisector_spans = np.empty(self.n_template_fits, dtype=np.float64)
            

    # Must define a build_full method which returns wave, model_flux on the detector grid
    # Can also define other build methods that return modified forward models
    def build_full(self, pars, *args, **kwargs):
        raise NotImplementedError("Must implement a build_full function for this instrument")
        
        
    def init_models(self, forward_model_settings, model_blueprints):
        
        # Stores the models
        self.models_dict = {}
        
        # Data pixels
        self.n_use_data_pix = int(self.n_data_pix - self.crop_data_pix[0] - self.crop_data_pix[1])
        
        # The left and right pixels. This should roughly match the bad pix arrays
        self.pix_bounds = [self.crop_data_pix[0], self.n_data_pix - self.crop_data_pix[1]]
        self.n_model_pix = int(self.model_resolution * self.n_data_pix)

        # First generate the wavelength solution model
        model_class = getattr(pcmodelcomponents, model_blueprints['wavelength_solution']['class_name'])
        self.wave_bounds = model_class.estimate_endpoints(self.data, model_blueprints['wavelength_solution'], self.pix_bounds)
        self.models_dict['wavelength_solution'] = model_class(model_blueprints['wavelength_solution'], self.wave_bounds, self.pix_bounds, self.n_data_pix, order_num=self.order_num)
        
        # The resolution of the high res fiducial wave grid
        self.dl = ((self.wave_bounds[1] +  15) - (self.wave_bounds[0] - 15)) / self.n_model_pix
        
        # Define the LSF model if present
        if 'lsf' in model_blueprints:
            model_class_init = getattr(pcmodelcomponents, model_blueprints['lsf']['class_name'])
            self.models_dict['lsf'] = model_class_init(model_blueprints['lsf'], self.dl, self.n_model_pix, order_num=self.order_num)
        
        # Generate the remaining model components from their blueprints and load any input templates
        # All remaining model components should subtype MultComponent
        for blueprint in model_blueprints:
            
            if blueprint in self.models_dict:
                continue
            
            # Construct the model
            model_class = getattr(pcmodelcomponents, model_blueprints[blueprint]['class_name'])
            self.models_dict[blueprint] = model_class(model_blueprints[blueprint], self.wave_bounds, order_num=self.order_num, flux_logspace=self.flux_logspace)
        
        if 'star' in self.models_dict and self.models_dict['star'].from_synthetic:
            self.crude = True
        else:
            self.crude = False

    def load_templates(self):
        
        templates_dict = {}
        
        for model in self.models_dict:
            if hasattr(self.models_dict[model], 'load_template'):
                templates_dict[model] = self.models_dict[model].load_template(nx=self.n_model_pix)
                
        return templates_dict


    def init_parameters(self):
        self.initial_parameters = OptimParameters.Parameters()
        for model in self.models_dict:
            self.models_dict[model].init_parameters(self.templates_dict)
            for pname in self.models_dict[model].par_names:
                self.initial_parameters[pname] = self.models_dict[model].initial_parameters[pname]


    # Save outputs after last iteration. This method can be implemented or not and super can be called or not.
    def save_results(self, output_path_opt, output_path_datamodels):
        
        ord_spec = '_ord' + str(self.order_num) + '_spec' + str(self.spec_num)
        
        # Best fit parameters and opt array
        filename_opt = output_path_opt + self.tag + '_opt' + ord_spec + '.npz'
        np.savez(filename_opt, best_fit_pars=self.best_fit_pars, opt=self.opt)
        
        # Data flux, flux_unc, badpix, best fit forward models, and residuals
        filename_data_models = output_path_datamodels + self.tag + '_data_model' + ord_spec + '.npz'
        data_arr = np.array([self.data.flux, self.data.flux_unc, self.data.badpix]).T
        
        np.savez(filename_data_models, wavelength_solutions=self.wavelength_solutions, residuals=self.residuals, models=self.models, data=data_arr)
        
        # Also save model as a pickle
        self.save_to_pickle(output_path=output_path_datamodels)
                
    # Prints the models and corresponding parameters after each fit if verbose_print=True
    def pretty_print(self):
        # Loop over models
        for mname in self.models_dict.keys():
            # Print the model string
            print(self.models_dict[mname])
            # Sub loop over per model parameters
            for pname in self.models_dict[mname].par_names:
                print('    ', end='', flush=True)
                if len(self.best_fit_pars) == 0:
                    print(self.initial_parameters[pname], flush=True)
                else:
                    print(self.best_fit_pars[-1][pname], flush=True)
                
    def set_parameters(self, pars):
        self.initial_parameters.update(pars)
        for model in self.models_dict:
            model = self.models_dict[model]
            for pname in model.initial_parameters.keys():
                if pname in pars:
                    model.initial_parameters[pname] = pars[pname]
    
    # Plots the forward model after each iteration with other template as well if verbose_plot = True
    def plot_model(self, iter_num, output_path, save=True):
        
        wave_factors = {
            'microns': 1E-4,
            'nm' : 1E-1,
            'ang' : 1
        }
        
        # k1 = index for forward model array access
        # k2 = Plot names for forward model objects
        # k3 = index for RV array access
        # k4 = RV plot names
        k1, k2, k3, k4 = self.iteration_indices(iter_num)
        
        # Units for plotting wavelength
        wave_factor = wave_factors[self.plot_wave_unit]
        
        # Extract the low res wave grid in proper units
        wave = self.wavelength_solutions[-1] * wave_factor
        
        # The best fit forward model for this iteration
        model = self.models[-1]
        
        # The residuals for this iteration
        residuals = self.residuals[-1]
        
        # The filename for the plot
        if save:
            fname = output_path + self.tag + '_data_model_spec' + str(self.spec_num) + '_ord' + str(self.order_num) + '_iter' + str(k2) + '.png'

        # Define some helpful indices
        good = np.where(self.data.badpix == 1)[0]
        f, l = good[0], good[-1]
        bad = np.where(self.data.badpix == 0)[0]
        bad_data_locs = np.argsort(np.abs(residuals[good]))[-1*self.flag_n_worst_pixels:]
        use_pix = np.arange(good[0], good[-1]).astype(int)
        pad = 0.01 * (wave[use_pix[-1]] - wave[use_pix[0]])
        
        # Figure
        plot_width, plot_height = 2000, 720
        dpi = 200
        fig, ax = plt.subplots(1, 1, figsize=(int(plot_width / dpi), int(plot_height / dpi)), dpi=dpi)
        
        # Data
        ax.plot(wave[use_pix], self.data.flux[use_pix], color=(0, 114/255, 189/255), lw=0.8)
        
        # Model
        ax.plot(wave[use_pix], model[use_pix], color=(217/255, 83/255, 25/255), lw=0.8)
        
        # Zero line
        ax.plot(wave[use_pix], np.zeros(wave[use_pix].size), color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':')
        
        # Residuals (all bad pixels will be zero here)
        ax.plot(wave[good], residuals[good], color=(255/255, 169/255, 22/255), lw=0.8)
        
        # The worst N pixels that were flagged
        ax.plot(wave[good][bad_data_locs], residuals[good][bad_data_locs], color='darkred', marker='X', lw=0)
        
        # Plot the convolved low res templates for debugging 
        # Plots the star and tellurics by default. Plots gas cell if present.
        if self.verbose_plot:
            
            pars = self.best_fit_pars[k1]
            lsf = self.models_dict['lsf'].build(pars=pars)
            
            # Extra zero line
            plt.plot(wave[use_pix], np.zeros(wave[use_pix].size) - 0.1, color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':', alpha=0.8)
            
            # Star
            if self.models_dict['star'].enabled:
                star_flux_hr = self.models_dict['star'].build(pars, self.templates_dict['star'][:, 0], self.templates_dict['star'][:, 1], self.templates_dict['star'][:, 0])
                star_convolved = self.models_dict['lsf'].convolve_flux(star_flux_hr, lsf=lsf)
                star_flux_lr = np.interp(wave / wave_factor, self.templates_dict['star'][:, 0], star_convolved, left=np.nan, right=np.nan)
                ax.plot(wave[use_pix], star_flux_lr[use_pix] - 1.1, label='Star', lw=0.8, color='deeppink', alpha=0.8)
            
            # Tellurics
            if 'tellurics' in self.models_dict and self.models_dict['tellurics'].enabled:
                tellurics = self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], self.templates_dict['star'][:, 0])
                tellurics_convolved = self.models_dict['lsf'].convolve_flux(tellurics, lsf=lsf)
                tell_flux_lr = np.interp(wave / wave_factor, self.templates_dict['star'][:, 0], tellurics_convolved, left=np.nan, right=np.nan)
                ax.plot(wave[use_pix], tell_flux_lr[use_pix] - 1.1, label='Tellurics', lw=0.8, color='indigo', alpha=0.8)
            
            # Gas Cell
            if 'gas_cell' in self.models_dict and self.models_dict['gas_cell'].enabled:
                gas_flux_hr = self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'][:, 0], self.templates_dict['gas_cell'][:, 1], self.templates_dict['star'][:, 0])
                gas_cell_convolved = self.models_dict['lsf'].convolve_flux(gas_flux_hr, lsf=lsf)
                gas_flux_lr = np.interp(wave / wave_factor, self.templates_dict['star'][:, 0], gas_cell_convolved, left=np.nan, right=np.nan)
                ax.plot(wave[use_pix], gas_flux_lr[use_pix] - 1.1, label='Gas Cell', lw=0.8, color='green', alpha=0.8)
            ax.set_ylim(-1.1, 1.1)
            ax.legend(loc='lower right')
        else:
            if iter_num == 0 and not self.models_dict['star'].enabled:
                ax.set_ylim(-0.4, 1.1)
            else:
                ax.set_ylim(-0.1, 1.1)
            
        ax.set_xlim(wave[f] - pad, wave[l] + pad)
        ax.set_xlabel('Wavelength [' + self.plot_wave_unit + ']', fontsize=12)
        ax.set_ylabel('Data, Model, Residuals', fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig(fname)
            plt.close()
        else:
            return fig, ax
        
    
    # k1 = index for forward model array access    
    # k2 = Plot names for forward model objects
    # k3 = index for RV array access
    # k4 = RV plot names
    def iteration_indices(self, iter_num):
        if self.models_dict['star'].from_synthetic:
            k1 = iter_num
            k2 = iter_num + 1
            k3 = iter_num
            k4 = iter_num + 1
            return k1, k2, k3, k4
        else:
            # No nelder mead fits have been performed
            if iter_num == 0 and self.crude:
                k1 = 0
                k2 = 0
                k3 = None # just to make sure!
                k4 = None # just to make sure!
                return k1, k2, k3, k4
            # Zeroth iteration is complete
            elif iter_num == 0 and not self.models_dict['star'].enabled:
                k1 = iter_num
                k2 = iter_num
                k3 = iter_num
                k4 = iter_num + 1
                return k1, k2, k3, k4
            # "first" (really the second) iteration is complete
            else:
                k1 = iter_num + 1
                k2 = iter_num + 1
                k3 = iter_num
                k4 = iter_num + 1
                return k1, k2, k3, k4


    # Save the forward model object to a pickle
    def save_to_pickle(self, output_path):
        fname = output_path + self.tag + '_forward_model_ord' + str(self.order_num) + '_spec' + str(self.spec_num) + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
            
    
    # Gets the night which corresponds to the spec index
    def get_thisnight_index(self, n_obs_nights):
        return get_night_index(self.spec_index, n_obs_nights)
    
    # Gets the night which corresponds to the spec index
    @staticmethod
    def get_night_index(spec_index, n_obs_nights):
        
        running_spec_index = n_obs_nights[0]
        n_nights = len(n_obs_nights)
        for inight in range(n_nights):
            if spec_index < running_spec_index:
                return inight
            running_spec_index += n_obs_nights[inight+1]
    
    
    # Gets the indices of spectra for a certain night. (zero based)
    def get_all_spec_indices_from_thisnight(self, n_obs_nights):
        night_index = self.get_thisnight_index(n_obs_nights)
        return get_all_spec_indices_from_night(night_index, n_obs_nights)
    
    # Gets the indices of spectra for a certain night. (zero based)
    @staticmethod
    def get_all_spec_indices_from_night(night_index, n_obs_nights):
            
        if night_index == 0:
            f = 0
            l = f + n_obs_nights[0]
        else:
            f = np.sum(n_obs_nights[0:night_index])
            l = f + n_obs_nights[night_index]

        return np.arange(f, l).astype(int).tolist()
    
    
    # Gets the actual index of a spectrum given the night and nightly index
    @staticmethod
    def night_to_full_spec_index(night_index, sub_spec_index, n_obs_nights):
            
        if night_index == 0:
            return spec_index
        else:
            f = np.sum(n_obs_nights[0:night_index])
            return f + sub_spec_index
        
            
class iSHELLForwardModel(ForwardModel):

    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):

        super().__init__(input_file, forward_model_settings, model_blueprints, order_num, spec_num=spec_num)

    def build_full(self, pars, iter_num):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]

        # Star
        model = self.models_dict['star'].build(pars, self.templates_dict['star'][:, 0], self.templates_dict['star'][:, 1], final_hr_wave_grid)
        
        # Gas Cell
        model *= self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'][:, 0], self.templates_dict['gas_cell'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Fringing from who knows what
        model *= self.models_dict['fringing'].build(pars, final_hr_wave_grid)
        
        # Blaze Model
        model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Convolve Model with LSF
        model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])

        return wavelength_solution, model_lr
                    
    # Returns the high res model on the fiducial grid with no stellar template and the low res wavelength solution
    def build_hr_nostar(self, pars, iter_num):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        # Gas Cell
        model = self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'][:, 0], self.templates_dict['gas_cell'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Fringing from who knows what
        model *= self.models_dict['fringing'].build(pars, final_hr_wave_grid)
        
        # Blaze Model
        model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])

        return wavelength_solution, model
    
class CHIRONForwardModel(ForwardModel):
    
    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):

        super().__init__(input_file, forward_model_settings, model_blueprints, order_num, spec_num=spec_num)

    def build_full(self, pars, iter_num):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]

        # Star
        model = self.models_dict['star'].build(pars, self.templates_dict['star'][:, 0], self.templates_dict['star'][:, 1], final_hr_wave_grid)
        
        # Gas Cell
        model *= self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'][:, 0], self.templates_dict['gas_cell'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Convolve Model with LSF
        model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])
        
        return wavelength_solution, model_lr


class PARVIForwardModel(ForwardModel):
    
    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):

        super().__init__(input_file, forward_model_settings, model_blueprints, order_num, spec_num=spec_index)

    def build_full(self, pars, iter_num):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]

        # Star
        model = self.models_dict['star'].build(pars, self.templates_dict['star'][:, 0], self.templates_dict['star'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Total flux
        #raw_flux_pre_conv = blaze * tell * star

        # Convolve Model with LSF
        model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])
        
        return wavelength_solution, model_lr


class MinervaAustralisForwardModel(ForwardModel):

    def __init__(self, spec_num, order_num, models_dict, data, initial_parameters, gpars):

        super().__init__(spec_num, order_num, models_dict, data, initial_parameters, gpars)

    def build_full(self, pars, templates_dict, iter_num, gpars):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = templates_dict['star'][:, 0]

        # Star
        star = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], final_hr_wave_grid)
        
        # Gas Cell
        # EVENTUALLY IODINE GAS CELL
        ##gas = self.models_dict['gas_cell'].build(pars, templates_dict['gas_cell'][:, 0], templates_dict['gas_cell'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        tell = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        blaze = self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Total flux
        raw_flux_pre_conv = blaze * tell * star

        # Convolve Model with LSF
        final_hr_flux = self.models_dict['lsf'].convolve_flux(raw_flux_pre_conv, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, final_hr_flux, left=final_hr_flux[0], right=final_hr_flux[-1])

        return wavelength_solution, model_lr