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
from robustneldermead.neldermead import NelderMead


# Stores all forward model objects useful wrapper to store all the forward model objects.
class ForwardModels(list):
    """Contains individual forward models in a list, and other helpful attributes, primarily for RVs.
    """
        
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
        
        # The proper tag
        self.tag = self.star_name + '_' + self.spectrograph.lower() + '_' + self.tag
        
        # Create output directories
        self.create_output_dirs()

        # Initiate the data, models, and outputs
        self.init(forward_model_settings, model_blueprints)

        # Remaining instance members

        # The number of iterations for rvs and template fits
        self.n_iters_rvs = self.n_template_fits
        self.n_iters_fits = self.n_iters_rvs + int(not self[0].models_dict['star'].from_synthetic)
        
        # Save the global parameters dictionary to the output directory
        with open(self.run_output_path + os.sep + 'global_parameters_dictionary.pkl', 'wb') as f:
            pickle.dump(forward_model_settings, f)

        # Print summary
        self.print_init_summary()
        
    def generate_nightly_rvs(self, iter_index):
        """Genreates individual and nightly (co-added) RVs after forward modeling all spectra and stores them in the ForwardModels object. If do_xcorr is True, nightly cross-correlation RVs are also computed.

        Args:
            iter_index (int): The iteration to generate RVs from.
        """

        # The nightly RVs and error bars.
        rvs_nightly = np.full(self.n_nights, fill_value=np.nan)
        unc_nightly = np.full(self.n_nights, fill_value=np.nan)

        # The best fit stellar RVs, remove the barycenter velocity
        rvs = np.array([self[ispec].best_fit_pars[-1][self[ispec].models_dict['star'].par_names[0]].value + self[ispec].data.bc_vel for ispec in range(self.n_spec)], dtype=np.float64)
        
        # The RMS from the forward model fit
        rms = np.array([self[ispec].opt[-1][0] for ispec in range(self.n_spec)], dtype=np.float64)
        weights = 1 / rms**2
        
        # The NM RVs
        rvs_nightly, unc_nightly = pcrvcalc.compute_nightly_rvs_single_order(rvs, weights, self.n_obs_nights, flag_outliers=False)
        self.rvs_dict['rvs'][:, iter_index] = rvs
        self.rvs_dict['rvs_nightly'][:, iter_index] = rvs_nightly
        self.rvs_dict['unc_nightly'][:, iter_index] = unc_nightly
        
        # The xcorr RVs
        if self.do_xcorr:
            rvs_xcorr_nightly = np.full(self.n_nights, fill_value=np.nan)
            unc_xcorr_nightly = np.full(self.n_nights, fill_value=np.nan)
            rvsx = self.rvs_dict['rvs_xcorr'][:, iter_index]
            rvsx_nightly, uncx_nightly = pcrvcalc.compute_nightly_rvs_single_order(rvsx, weights, self.n_obs_nights, flag_outliers=False)
            self.rvs_dict['rvs_nightly'][:, iter_index] = rvsx_nightly
            self.rvs_dict['unc_nightly'][:, iter_index] = uncx_nightly
        
        
    # Updates spectral models according to best fit parameters, and run the update method for each iteration.
    def update_models(self, iter_index):

        for ispec in range(self.n_spec):

            # Pass the previous iterations best pars as starting points
            self[ispec].set_parameters(copy.deepcopy(self[ispec].best_fit_pars[-1]))
            
            # Ensure the same templates dict is shared amongst the models (just a pointer to a single instance)
            for fwm in self:
                fwm.templates_dict = self.templates_dict
            
            # Update other models
            for model in self[ispec].models_dict.keys():
                self[ispec].models_dict[model].update(self[ispec], iter_index)
                
                
    def init(self, forward_model_settings, model_blueprints):
        
        print('Loading in data and constructing forward model objects for order ' + str(self.order_num) + ' ...')
        
        # The input files
        input_files = [self.input_path + f for f in np.atleast_1d(np.genfromtxt(self.input_path + self.flist_file, dtype='<U100', comments='#').tolist())]
        self.n_spec = len(input_files)
        
        # The inidividual forward model object init
        fwm_class_init = eval(self.spectrograph + 'ForwardModel')

        # Init inidividual forward models
        for ispec in range(self.n_spec):
            self.append(fwm_class_init(input_files[ispec], forward_model_settings, model_blueprints, self.order_num, spec_num=ispec + 1))
            
        # Load templates
        self.load_templates()
        
        # Initiate the RV dicts
        self.init_rvs()
        
        # Init the parameters
        self.init_parameters()
        
        # The number of spectra (may overwrite)
        self.n_spec = len(input_files)
            
    def init_rvs(self):
        
        # The bary-center information
        if hasattr(self, 'bary_corr_file') and self.bary_corr_file is not None:
            self.BJDS, self.bc_vels = np.loadtxt(self.input_path + self.bary_corr_file, delimiter=',', unpack=True)
            for ispec in range(self.n_spec):
                self[ispec].data.set_bc_info(self.BJDS[ispec], self.bc_vels[ispec])
        else:
            bc_computer = self[0].data.__class__.calculate_bc_info_all
            self.BJDS, self.bc_vels = bc_computer(self, obs_name=self.observatory, star_name=self.star_name)
            

        if self.compute_bc_only:
            np.savetxt(self.run_output_path + 'bary_corrs_' + self.star_name + '.txt', np.array([self.BJDS, self.bc_vels]).T, delimiter=',')
            sys.exit("Compute BC info only is set!")
        
        # Compute the nightly BJDs and n obs per night
        self.BJDS_nightly, self.n_obs_nights = pcrvcalc.get_nightly_jds(self.BJDS)
        
        # Sort everything by BJD
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
        if self.xcorr_options['method'] is not None:
            
            # Do x corr
            self.do_xcorr = True
            
            # Number of velocities to try in the brute force or ccf
            self.xcorr_options['n_vels'] = int(2 * self.xcorr_options['range'] / self.xcorr_options['step'])
            
            # Initiate arrays for xcorr rvs.
            self.rvs_dict['rvs_xcorr'] = np.empty(shape=(self.n_spec, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['rvs_xcorr_nightly'] = np.empty(shape=(self.n_nights, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['unc_xcorr_nightly'] = np.empty(shape=(self.n_nights, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['xcorrs'] = np.empty(shape=(self.xcorr_options['n_vels'], 2*self.n_spec, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['line_bisectors'] = np.empty(shape=(self.xcorr_options['n_bs'], self.n_spec, self.n_template_fits), dtype=np.float64)
            self.rvs_dict['bisector_spans'] = np.empty(shape=(self.n_spec, self.n_template_fits), dtype=np.float64)
            
        else:
            self.do_xcorr = False
        
        
    def sort(self):
        """Sorts the objects by BJD, in place.
        """
        # Sort by BJD
        sorting_inds = np.argsort(self.BJDS)
        self.BJDS = self.BJDS[sorting_inds]
        self.bc_vels = self.bc_vels[sorting_inds]
        
        # Also sort the list
        for ispec in range(self.n_spec):
            self[ispec] = self[sorting_inds[ispec]]
            self[ispec].spec_index = ispec
            self[ispec].spec_num = ispec + 1
            self[ispec].data.spec_index = ispec
            self[ispec].data.spec_num = ispec + 1
    
    
    def init_parameters(self):
        """Initializes the parameters for each forward model
        """
        for ispec in range(self.n_spec):
            self[ispec].init_parameters()

    # Stores the forward model outputs in .npz files for all iterations
    # Stores the RVs in a single .npz
    def save_results(self):
        """Saves the forward model results and RVs.
        """
        
        # Save the RVs
        self.save_rvs()
        
        # For each spectrum, save the forward model results
        for ispec in range(self.n_spec):
            self[ispec].save_results(self.run_output_path_opt_results, self.run_output_path_spectral_fits)
        
    # Wrapper to fit all spectra
    def fit_spectra(self, iter_index):
        """Forward models all spectra and performs xcorr if set.
        
        Args:
            iter_index (int): The iteration index.
        """
        # Timer
        stopwatch = pcutils.StopWatch()

        # Parallel fitting
        if self.n_cores > 1:

            # Construct the arguments
            args_pass = []
            for spec_num in range(self.n_spec):
                args_pass.append((self[spec_num], iter_index, self.n_spec))
            
            # Call the parallel job via joblib.
            self[:] = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(self.solver_wrapper)(*args_pass[ispec]) for ispec in range(self.n_spec))

        else:
            # Fit one at a time
            for ispec in range(self.n_spec):
                print('    Performing Nelder-Mead Fit For Spectrum '  + str(ispec+1) + ' of ' + str(self.n_spec), flush=True)
                self[ispec] = self.solver_wrapper(self[ispec], iter_index, self.n_spec)
        
        # Cross correlate if set
        if self.do_xcorr and self.n_template_fits > 0 and self.models_dict['star'].enabled:
            self.cross_correlate_spectra(iter_index)
            
        # Fit in Parallel
        print('Fitting Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)
    
    # Outputs RVs and cross corr analyses for all iterations for a given order.
    def save_rvs(self):
        """Saves the forward model results and RVs.
        """
        # Full filename
        fname = self.run_output_path_rvs + self.tag + '_rvs_ord' + str(self[0].order_num) + '.npz'
        
        # The bc velocities
        bc_vels = np.array([fwm.data.bc_vel for fwm in self], dtype=float)
        
        # Save in a .npz file for easy access later
        np.savez(fname, BJDS=self.BJDS, BJDS_nightly=self.BJDS_nightly, bc_vels=bc_vels, n_obs_nights=self.n_obs_nights, **self.rvs_dict)

    # Loads the templates dictionary and stores in a dictionary.
    # A pointer to the templates dictionary is stored in each forward model class
    # It can be accessed via forward_models.templates_dict or forward_models[ispec].
    def load_templates(self):
        """Load the initial templates and store in both.
        """
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
    def create_output_dirs(self):
        """Creates output dirs and filenames for outputs.
        """
        # Order folder
        o_folder = 'Order' + str(self.order_num) + os.sep
        
        # Output path for this run
        self.run_output_path = self.output_path_root + self.tag + os.sep
        
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
        """Print a summary for this run.
        """
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
    def solver_wrapper(forward_model, iter_index, n_spec_tot):
        """A wrapper for forward modeling and cross-correlating a single spectrum.

        Args:
            forward_model (ForwardModel): The forward model object
            iter_index (int): The iteration index.
            n_spec_tot (int): The total number of spectra for printing purposes.
            output_path_plot (str, optional): output path for plots. Defaults to None and uses object default.
            verbose_print (bool, optional): Whether or not to print optimization results. Defaults to False.
            verbose_plot (bool, optional): Whether or not to plot templates with the forward model. Defaults to False.

        Returns:
            forward_model (ForwardModel): The updated forward model since we possibly fit in parallel.
        """
        # Start the timer
        stopwatch = pcutils.StopWatch()
        
        # Construct the extra arguments to pass to the target function
        args_to_pass = (forward_model, iter_index)
        
        # Construct the Nelder Mead Solver and run
        solver = NelderMead(forward_model.target_function, forward_model.initial_parameters, no_improve_break=3, args_to_pass=args_to_pass, ftol=1E-5)
        opt_result = solver.solve()

        # Pass best fit parameters and optimization result to forward model
        forward_model.best_fit_pars.append(opt_result[0])
        forward_model.opt.append(opt_result[1:])

        # Build the best fit forward model
        best_wave, best_model = forward_model.build_full(forward_model.best_fit_pars[-1], iter_index)

        # Pass wavelength solution and best model to forward model
        forward_model.wavelength_solutions.append(best_wave)
        forward_model.models.append(best_model)

        # Compute the residuals between the data and model, don't flag bad pixels here. Cropped may still be nan.
        forward_model.residuals.append(forward_model.data.flux - best_model)

        # Print diagnostics if set
        if forward_model.verbose_print:
            print('RMS = %' + str(round(100*opt_result[1], 5)))
            print('Function Calls = ' + str(opt_result[2]))
            forward_model.pretty_print()
        
        print('    Fit Spectrum ' + str(forward_model.spec_num) + ' of ' + str(n_spec_tot) + ' in ' + str(round((stopwatch.time_since())/60, 2)) + ' min', flush=True)

        # Output a plot
        forward_model.plot_model(iter_index)

        # Return new forward model object since we possibly fit in parallel
        return forward_model


    def cross_correlate_spectra(self, iter_index=None):
        """Cross correlation wrapper for all spectra.

        Args:
            iter_index (int or None): The iteration to use. If None, then it's assumed to be a crude first guess.
        """
        # Fit in Parallel
        stopwatch = pcutils.StopWatch()
        print('Cross Correlating Spectra ... ', flush=True)
        
        if iter_index is None:
            ccf_method = getattr(pcrvcalc, 'crude_brute_force')
        else:
            ccf_method = getattr(pcrvcalc, self.xcorr_options['method'])

        # Perform xcorr in series or parallel
        if self.n_cores > 1:

            # Construct the arguments
            iter_pass = []
            for ispec in range(self.n_spec):
                iter_pass.append((self[ispec], iter_index))

            # Cross Correlate in Parallel
            ccf_results = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(ccf_method)(*iter_pass[ispec]) for ispec in range(self.n_spec))
            
        else:
            ccf_results = [ccf_method(self[ispec], iter_index) for ispec in range(self.n_spec)]
        
        # Pass to arrays
        if iter_index is None:
            for ispec in range(self.n_spec):
                self[ispec].initial_parameters[self[ispec].models_dict['star'].par_names[0]].setv(value=ccf_results[ispec])
        else:
            for ispec in range(self.n_spec):
                self.rvs_dict['xcorrs'][:, 2*ispec:2*ispec+2, iter_index] = np.array([ccf_results[ispec][0], ccf_results[ispec][1]]).T
                self.rvs_dict['rvs_xcorr'][ispec, iter_index] = ccf_results[ispec][2]
                self.rvs_dict['bisector_spans'][ispec, iter_index] = ccf_results[ispec][3]
                
        print('Cross Correlation Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)

    def plot_rvs(self, iter_index):
        """Plots all RVs and cross-correlation analysis after forward modeling all spectra.

        Args:
            forward_models (ForwardModels): The list of forward model objects.
            iter_index (int): The iteration to use.
        """
        
        # Plot the rvs, nightly rvs, xcorr rvs, xcorr nightly rvs
        plot_width, plot_height = 1800, 600
        dpi = 200
        plt.figure(num=1, figsize=(int(plot_width/dpi), int(plot_height/dpi)), dpi=200)
        
        # Alias
        rvs = self.rvs_dict
        
        # Individual rvs from nelder mead fitting
        plt.plot(self.BJDS - self.BJDS_nightly[0],
                rvs['rvs'][:, iter_index] - np.nanmedian(rvs['rvs_nightly'][:, iter_index]),
                marker='.', linewidth=0, alpha=0.7, color=(0.1, 0.8, 0.1))
        
        # Nightly rvs from nelder mead fitting
        plt.errorbar(self.BJDS_nightly - self.BJDS_nightly[0],
                        rvs['rvs_nightly'][:, iter_index] - np.nanmedian(rvs['rvs_nightly'][:, iter_index]),
                        yerr=rvs['unc_nightly'][:, iter_index], marker='o', linewidth=0, elinewidth=1, label='Nelder Mead', color=(0, 114/255, 189/255))

        # Individual and nightly xcorr rvs
        if self.do_xcorr:
            plt.plot(self.BJDS - self.BJDS_nightly[0],
                        rvs['rvs_xcorr'][:, iter_index] - np.nanmedian(rvs['rvs_xcorr'][:, iter_index]),
                        marker='.', linewidth=0, color='black', alpha=0.6)
            plt.errorbar(self.BJDS_nightly - self.BJDS_nightly[0],
                            rvs['rvs_xcorr_nightly'][:, iter_index] - np.nanmedian(rvs['rvs_xcorr_nightly'][:, iter_index]),
                            yerr=rvs['unc_xcorr_nightly'][:, iter_index], marker='X', linewidth=0, alpha=0.8, label='X Corr', color='darkorange')
        
        plt.title(self[0].star_name + ' RVs Order ' + str(self.order_num) + ', Iteration ' + str(iter_index + 1), fontweight='bold')
        plt.xlabel('BJD - BJD$_{0}$', fontweight='bold')
        plt.ylabel('RV [m/s]', fontweight='bold')
        plt.legend(loc='upper right')
        plt.tight_layout()
        fname = self.run_output_path_rvs + self.tag + '_rvs_ord' + str(self.order_num) + '_iter' + str(iter_index + 1) + '.png'
        plt.savefig(fname)
        plt.close()
        
        if self.do_xcorr:
            # Plot the Bisector stuff
            plt.figure(1, figsize=(12, 7), dpi=200)
            for ispec in range(self.n_spec):
                v0 = rvs['rvs_xcorr'][ispec, iter_index]
                depths = np.linspace(0, 1, num=self.xcorr_options['n_bs'])
                ccf_ = rvs['xcorrs'][:, 2*ispec+1, iter_index] - np.nanmin(rvs['xcorrs'][:, 2*ispec+1, iter_index])
                ccf_ = ccf_ / np.nanmax(ccf_)
                plt.plot(rvs['xcorrs'][:, 2*ispec, iter_index] - v0, ccf_)
                plt.plot(rvs['line_bisectors'][:, ispec, iter_index], depths)
            
            plt.title(self.star_name + ' CCFs Order ' + str(self.order_num) + ', Iteration ' + str(iter_index + 1), fontweight='bold')
            plt.xlabel('RV$_{\star}$ [m/s]', fontweight='bold')
            plt.ylabel('CCF (RMS surface)', fontweight='bold')
            plt.xlim(-10000, 10000)
            plt.tight_layout()
            fname = self.run_output_path_rvs + self.tag + '_ccfs_ord' + str(self.order_num) + '_iter' + str(iter_index + 1) + '.png'
            plt.savefig(fname)
            plt.close()
        
            # Plot the Bisector stuff
            plt.figure(1, figsize=(12, 7), dpi=200)
            plt.plot(rvs['rvs_xcorr'][:, iter_index], rvs['bisector_spans'][:, iter_index], marker='o', linewidth=0)
            plt.title(self[0].star_name + ' CCF Bisector Spans Order ' + str(self.order_num) + ', Iteration ' + str(iter_index + 1), fontweight='bold')
            plt.xlabel('X Corr RV [m/s]', fontweight='bold')
            plt.ylabel('Bisector Span [m/s]', fontweight='bold')
            plt.tight_layout()
            fname = self.run_output_path_rvs + self.tag + '_bisectorspans_ord' + str(self.order_num) + '_iter' + str(iter_index + 1) + '.png'
            plt.savefig(fname)
            plt.close()
        
        
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
        self.tag = self.star_name + '_' + self.spectrograph.lower() + '_' + self.tag
        
        # Overwrite the target function with the actual function to optimize the model
        self.target_function = getattr(pctargetfuns, self.target_function)
        
        # Initialize the data
        data_class_init = getattr(pcdata, 'SpecData' + forward_model_settings['spectrograph'])
        self.data = data_class_init(input_file, self)
        self.n_data_pix = self.data.flux.size
        
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
        
        # Output path for this run
        self.run_output_path = self.output_path_root + self.tag + os.sep
        
        # Output paths for this order
        o_folder = 'Order' + str(self.order_num) + os.sep
        self.run_output_path_rvs = self.run_output_path + o_folder + 'RVs' + os.sep
        self.run_output_path_spectral_fits = self.run_output_path + o_folder + 'Fits' + os.sep
        self.run_output_path_opt_results = self.run_output_path + o_folder + 'Opt' + os.sep
        self.run_output_path_stellar_templates = self.run_output_path + o_folder + 'Stellar_Templates' + os.sep
        
        self.do_xcorr = True if self.xcorr_options['method'] is not None else False
        
        if self.do_xcorr:
            # Number of velocities to try in the brute force or ccf
            self.xcorr_options['n_vels'] = int(2 * self.xcorr_options['range'] / self.xcorr_options['step'])
            

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
        self.wave_bounds = model_class.estimate_bounds(self, model_blueprints['wavelength_solution'])
        self.models_dict['wavelength_solution'] = model_class(self, model_blueprints['wavelength_solution'])
        
        # The spacing of the high res fiducial wave grid
        self.dl = ((self.wave_bounds[1] +  15) - (self.wave_bounds[0] - 15)) / self.n_model_pix
        
        # Define the LSF model if present
        if 'lsf' in model_blueprints:
            model_class_init = getattr(pcmodelcomponents, model_blueprints['lsf']['class_name'])
            self.models_dict['lsf'] = model_class_init(self, model_blueprints['lsf'])
        
        # Generate the remaining model components from their blueprints and load any input templates
        # All remaining model components should subtype MultComponent
        for blueprint in model_blueprints:
            
            if blueprint in self.models_dict:
                continue
            
            # Construct the model
            model_class = getattr(pcmodelcomponents, model_blueprints[blueprint]['class_name'])
            self.models_dict[blueprint] = model_class(self, model_blueprints[blueprint])


    def load_templates(self):
        
        templates_dict = {}
        
        for model in self.models_dict:
            if hasattr(self.models_dict[model], 'load_template'):
                templates_dict[model] = self.models_dict[model].load_template(self)
                
        return templates_dict


    def init_parameters(self):
        self.initial_parameters = OptimParameters.Parameters()
        for model in self.models_dict:
            self.models_dict[model].init_parameters(self)
        self.initial_parameters.sanity_lock()


    # Save outputs after last iteration. This method can be implemented or not and super can be called or not.
    def save_results(self, output_path_opt, output_path_datamodels):
        
        ord_spec = '_ord' + str(self.order_num) + '_spec' + str(self.spec_num)
        
        # Best fit parameters and opt array
        #filename_opt = output_path_opt + self.tag + '_opt' + ord_spec + '.npz'
        #np.savez(filename_opt, best_fit_pars=self.best_fit_pars, opt=self.opt)
        
        # Data flux, flux_unc, badpix, best fit forward models, and residuals
        #filename_data_models = output_path_datamodels + self.tag + '_data_model' + ord_spec + '.npz'
        #data_arr = np.array([self.data.flux, self.data.flux_unc, self.data.badpix]).T
        
        #np.savez(filename_data_models, wavelength_solutions=self.wavelength_solutions, residuals=self.residuals, models=self.models, data=data_arr)
        
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
    
    # Plots the forward model after each iteration with other template as well if verbose_plot = True
    def plot_model(self, iter_index, save=True):
        
        wave_factors = {
            'microns': 1E-4,
            'nm' : 1E-1,
            'ang' : 1
        }
        
        # Units for plotting wavelength
        wave_factor = wave_factors[self.plot_wave_unit]
        
        # Extract the low res wave grid in proper units
        wave = self.wavelength_solutions[-1] * wave_factor
        
        # The best fit forward model for this iteration
        model = self.models[-1]
        
        # The residuals for this iteration
        residuals = self.residuals[-1]
        
        # The filename for the plot
        if self.models_dict['star'].enabled:
            fname = self.run_output_path_spectral_fits + self.tag + '_data_model_spec' + str(self.spec_num) + '_ord' + str(self.order_num) + '_iter' + str(iter_index + 1) + '.png'
        else:
            fname = self.run_output_path_spectral_fits + self.tag + '_data_model_spec' + str(self.spec_num) + '_ord' + str(self.order_num) + '_iter0.png'

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
        
        # "Zero" line
        ax.plot(wave[use_pix], np.zeros(use_pix.size), color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':')
        
        # Residuals (all bad pixels will be zero here)
        ax.plot(wave[good], residuals[good], color=(255/255, 169/255, 22/255), lw=0.8)
        
        # The worst N pixels that were flagged
        ax.plot(wave[good][bad_data_locs], residuals[good][bad_data_locs], color='darkred', marker='X', lw=0)
        
        # Plot the convolved low res templates for debugging 
        # Plots the star and tellurics by default. Plots gas cell if present.
        if self.verbose_plot:
            
            pars = self.best_fit_pars[-1]
            lsf = self.models_dict['lsf'].build(pars=pars)
            
            # Extra zero line
            ax.plot(wave[use_pix], np.zeros(wave[use_pix].size) - 0.1, color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':', alpha=0.8)
            
            # Star
            if self.models_dict['star'].enabled:
                star_flux_hr = self.models_dict['star'].build(pars, self.templates_dict['star'], self.templates_dict['star'][:, 0])
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
                gas_flux_hr = self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'], self.templates_dict['star'][:, 0])
                gas_cell_convolved = self.models_dict['lsf'].convolve_flux(gas_flux_hr, lsf=lsf)
                gas_flux_lr = np.interp(wave / wave_factor, self.templates_dict['star'][:, 0], gas_cell_convolved, left=np.nan, right=np.nan)
                ax.plot(wave[use_pix], gas_flux_lr[use_pix] - 1.1, label='Gas Cell', lw=0.8, color='green', alpha=0.8)
            ax.set_ylim(-1.1, 1.1)
            ax.legend(loc='lower right')
            
            # Residual lab flux
            if 'residual_lab' in self.templates_dict:
                res_hr = self.templates_dict['residual_lab'][:, 1]
                res_lr = np.interp(wave / wave_factor, self.templates_dict['star'][:, 0], res_hr, left=np.nan, right=np.nan)
                ax.plot(wave[use_pix], res_lr[use_pix] - 0.1, label='Lab Frame Coherence', lw=0.8, color='darkred', alpha=0.8)
        else:
            ax.set_ylim(-0.1, 1.1)
            
        ax.set_xlim(wave[f] - pad, wave[l] + pad)
        ax.set_xlabel('Wavelength [' + self.plot_wave_unit + ']', fontsize=12)
        ax.set_ylabel('Data, Model, Residuals', fontsize=12)
        
        if save:
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
        else:
            return fig

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

    def build_full(self, pars, iter_index):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        model = np.ones_like(final_hr_wave_grid)

        # Star
        if self.models_dict['star'].enabled:
            model *= self.models_dict['star'].build(pars, self.templates_dict['star'], final_hr_wave_grid)
        
        # Gas Cell
        if self.models_dict['gas_cell'].enabled:
            model *= self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'], final_hr_wave_grid)
            
        # All tellurics
        if self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Fringing from who knows what
        if self.models_dict['fringing'].enabled:
            model *= self.models_dict['fringing'].build(pars, final_hr_wave_grid)
        
        # Blaze Model
        if self.models_dict['blaze'].enabled:
            model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)
        
        # Residual lab flux
        if 'residual_lab' in self.templates_dict:
            model += self.templates_dict['residual_lab'][:, 1]

        # Convolve Model with LSF
        model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])
        
        if self.debug:
            stop()

        return wavelength_solution, model_lr
                    
    # Returns the high res model on the fiducial grid with no stellar template and the low res wavelength solution
    def build_hr_nostar(self, pars, iter_index):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        model = np.ones_like(final_hr_wave_grid)
        
        # Gas Cell
        if self.models_dict['gas_cell'].enabled:
            model *= self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'], final_hr_wave_grid)
        
        # All tellurics
        if self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Fringing from who knows what
        if self.models_dict['fringing'].enabled:
            model *= self.models_dict['fringing'].build(pars, final_hr_wave_grid)
        
        # Blaze Model
        if self.models_dict['blaze'].enabled:
            model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)
        
        # Residual lab flux
        if 'residual_lab' in self.templates_dict:
            model += self.templates_dict['residual_lab'][:, 1]

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)
        
        if self.debug:
            stop()

        return wavelength_solution, model
    
class CHIRONForwardModel(ForwardModel):
    
    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):

        super().__init__(input_file, forward_model_settings, model_blueprints, order_num, spec_num=spec_num)

    def build_full(self, pars, iter_index):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        model = np.ones_like(final_hr_wave_grid)

        # Star
        if self.models_dict['star'].enabled:
            model *= self.models_dict['star'].build(pars, self.templates_dict['star'], final_hr_wave_grid)
        
        # Gas Cell
        if self.models_dict['gas_cell'].enabled:
            model *= self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'], final_hr_wave_grid)
        
        # All tellurics
        if self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        if self.models_dict['blaze'].enabled:
            model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Convolve Model with LSF
        if self.models_dict['lsf'].enabled:
            model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])
        
        if self.debug:
            stop()
        
        return wavelength_solution, model_lr


class PARVIForwardModel(ForwardModel):
    
    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):

        super().__init__(input_file, forward_model_settings, model_blueprints, order_num, spec_num=spec_num)

    def build_full(self, pars, iter_index):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        model = np.ones_like(final_hr_wave_grid)

        # Star
        if self.models_dict['star'].enabled:
            model *= self.models_dict['star'].build(pars, self.templates_dict['star'], final_hr_wave_grid)
        
        # All tellurics
        if self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        if self.models_dict['blaze'].enabled:
            model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Convolve Model with LSF
        if self.models_dict['lsf'].enabled:
            model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])
        
        if self.debug:
            stop()
        
        return wavelength_solution, model_lr


class MinervaAustralisForwardModel(ForwardModel):

    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):

        super().__init__(input_file, forward_model_settings, model_blueprints, order_num, spec_num=spec_num)

    def build_full(self, pars, iter_index):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        model = np.ones_like(final_hr_wave_grid)

        # Star
        if self.models_dict['star'].enabled:
            model *= self.models_dict['star'].build(pars, self.templates_dict['star'], final_hr_wave_grid)
        
        # All tellurics
        if self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        if self.models_dict['blaze'].enabled:
            model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Convolve Model with LSF
        if self.models_dict['lsf'].enabled:
            model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])
        
        if self.debug:
            stop()

        return wavelength_solution, model_lr
    
    
class MinervaNorthForwardModel(ForwardModel):
    
    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):

        super().__init__(input_file, forward_model_settings, model_blueprints, order_num, spec_num=spec_num)

    def build_full(self, pars, iter_index):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        model = np.ones_like(final_hr_wave_grid)

        # Star
        if self.models_dict['star'].enabled:
            model *= self.models_dict['star'].build(pars, self.templates_dict['star'], final_hr_wave_grid)
        
        # Gas Cell
        if self.models_dict['gas_cell'].enabled:
            model *= self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'], final_hr_wave_grid)
        
        # All tellurics
        if self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        if self.models_dict['blaze'].enabled:
            model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Convolve Model with LSF
        if self.models_dict['lsf'].enabled:
            model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])
        
        if self.debug:
            stop()
        
        return wavelength_solution, model_lr
    
    
class NIRSPECForwardModel(ForwardModel):

    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):
        
        super().__init__(input_file, forward_model_settings, model_blueprints, order_num, spec_num=spec_num)

    def build_full(self, pars, iter_index):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        model = np.ones_like(final_hr_wave_grid)

        # Star
        if self.models_dict['star'].enabled:
            model *= self.models_dict['star'].build(pars, self.templates_dict['star'], final_hr_wave_grid)
            
        # All tellurics
        if self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Fringing
        if self.models_dict['fringing'].enabled:
            model *= self.models_dict['fringing'].build(pars, final_hr_wave_grid)
        
        # Blaze Model
        if self.models_dict['blaze'].enabled:
            model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)
        
        # Residual lab flux
        if 'residual_lab' in self.templates_dict:
            model += self.templates_dict['residual_lab'][:, 1]

        # Convolve Model with LSF
        if self.models_dict['lsf'].enabled:
            model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])
        
        if self.debug:
            stop()

        return wavelength_solution, model_lr
    
    
class PARVI_devForwardModel(ForwardModel):
    
    def __init__(self, input_file, forward_model_settings, model_blueprints, order_num, spec_num=None):

        super().__init__(input_file, forward_model_settings, model_blueprints, order_num, spec_num=spec_num)

    def build_full(self, pars, iter_index):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        model = np.ones_like(final_hr_wave_grid)

        # Star
        if self.models_dict['star'].enabled:
            model *= self.models_dict['star'].build(pars, self.templates_dict['star'][:, 0], self.templates_dict['star'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        if self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        if self.models_dict['blaze'].enabled:
            model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)
            
        # Residual lab flux
        if 'residual_lab' in self.templates_dict:
            model += self.templates_dict['residual_lab'][:, 1]

        # Convolve Model with LSF
        if self.models_dict['lsf'].enabled:
            model[:] = self.models_dict['lsf'].convolve_flux(model, pars=pars)

        # Generate the wavelength solution of the data
        #print("JWD final_hr_wave_grid.shape:",final_hr_wave_grid.shape)
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, model, left=model[0], right=model[-1])
        
        if self.debug:
            stop()
        
        return wavelength_solution, model_lr
    
    def build_hr_nostar(self, pars, iter_index):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = self.templates_dict['star'][:, 0]
        
        model = np.ones_like(final_hr_wave_grid)
        
        # Gas Cell
        if self.models_dict['gas_cell'].enabled:
            model *= self.models_dict['gas_cell'].build(pars, self.templates_dict['gas_cell'], final_hr_wave_grid)
        
        # All tellurics
        if self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, self.templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        if self.models_dict['blaze'].enabled:
            model *= self.models_dict['blaze'].build(pars, final_hr_wave_grid)
        
        # Residual lab flux
        if 'residual_lab' in self.templates_dict:
            model += self.templates_dict['residual_lab'][:, 1]

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)
        
        if self.debug:
            stop()

        return wavelength_solution, model