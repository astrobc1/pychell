# Python built in modules
import copy
import os # Making directories
import time # Time the code
import pickle

# Graphics
import matplotlib # to set the backend
import matplotlib.pyplot as plt # Plotting
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Multiprocessing
from joblib import Parallel, delayed
import tqdm

# Science/math
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np # Math, Arrays

# llvm
from numba import njit, jit, prange

# User defined
import pychell.maths as pcmath
from pychell.maths import cspline_interp, lin_interp
import pychell.rvs.template_augmenter as pcaugmenter
import pychell.rvs.model_components as pcmodels
import pychell.data as pcdata
import pychell.rvs.target_functions as pctargetfuns
import pychell.utils as pcutils
import pychell.rvs.rvcalc as pcrvcalc

# Optimization
import optimparameters.parameters as OptimParameters
from robustneldermead.neldermead import NelderMead

class ForwardModels(list):
    
    def __init__(self, config, model_blueprints, parser, order_num):
        
        # Init the list
        super().__init__()
        
        # Order number
        self.order_num = order_num
        
        # The parser
        self.parser = parser
        
        # Set config
        for key in config:
            setattr(self, key, copy.deepcopy(config[key]))
            
        # The proper tag
        self.tag = self.spectrograph.lower() + '_' + self.tag
        
        self.template_augmenter = getattr(pcaugmenter, self.template_augmenter)
            
        # The output directories
        self.create_output_paths()
            
        # grab the input files
        input_files = self.load_filelist()
        
        # Number of observations
        self.n_spec = len(input_files)
        
        # The sub class
        fwm_class = eval(self.spectrograph + 'ForwardModel')
        
        # Init the sub classes for each chunk
        print('Initializing Forward Models for Each Observation', flush=True)
        for ispec in range(self.n_spec):
            self.append(fwm_class(input_files[ispec], config, model_blueprints, parser, order_num, ispec + 1))
            
        # Sort by bjd
        self.sort()
            
        # Init RVs dictionary
        self.init_rvs()
        
        # Init parameters
        for fwm in self:
            fwm.init_parameters()
        
        # Load templates
        self.load_templates()
            
        self.print_init_summary()
        
        # Initial optimization to guess some parameters
        self.init_optimize()
        
    def load_filelist(self):
        return self.parser.load_filelist(self)
    
    @property
    def n_nights(self):
        return self.rvs_dict['n_nights']
    
    @property
    def n_obs_nights(self):
        return self.rvs_dict['n_obs_nights']
    
    @property
    def bjds(self):
        return self.rvs_dict['bjds']
    
    @property
    def bc_vels(self):
        return self.rvs_dict['bc_vels']
    
    def update_templates(self, iter_index):
        for model in self[0].models_dict:
            if hasattr(self[0].models_dict[model], 'augmenter'):
                self[0].models_dict[model].augmenter(self, iter_index)
    
    def sort(self):
        jds = np.array([fwm.data.parser.parse_time(fwm.data).jd for fwm in self], dtype=float)
        ss = np.argsort(jds)
        __tempself__ = copy.deepcopy(self)
        for ispec in range(self.n_spec):
            self[ispec] = copy.deepcopy(__tempself__[ss[ispec]])
        del __tempself__
    
    def update_models(self, iter_index):
        for fwm in self:
            for model in fwm.models_dict:
                fwm.models_dict[model].update(fwm, iter_index)
    
    def print_init_summary(self):
        """Print a summary for this run.
        """
        # Print summary
        print('***************************************', flush=True)
        print('** Target: ' + self.star_name, flush=True)
        print('** Spectrograph: ' + self.observatory['name'] + ' / ' + self.spectrograph, flush=True)
        print('** Observations: ' + str(self.n_spec) + ' spectra, ' + str(self.n_nights) + ' nights', flush=True)
        print('** Echelle Order: ' + str(self.order_num), flush=True)
        print('** TAG: ' + self.tag, flush=True)
        print('** N Iterations: ' + str(self.n_template_fits), flush=True)
        print('***************************************', flush=True)
    
    def init_rvs(self):
        
        # Init the shared rv dictionary
        self.rvs_dict = {}
        
        # Bc info
        self.parser.load_barycenter_corrections(self)
        
        self.rvs_dict['bjds'] = [fwm.data.bjd for fwm in self]
        self.rvs_dict['bc_vels'] = [fwm.data.bc_vel for fwm in self]
        
        # Compute the nightly BJDs and n obs per night
        self.rvs_dict['bjds_nightly'], self.rvs_dict['n_obs_nights'] = pcrvcalc.get_nightly_jds(self.rvs_dict['bjds'])
        
        # The number of nights
        self.rvs_dict['n_nights'] = len(self.rvs_dict['bjds_nightly'])
        
        # Nelder-Mead RVs
        self.rvs_dict['rvsfwm'] = np.full(shape=(self.n_spec, self.n_chunks, self.n_template_fits), fill_value=np.nan)
        self.rvs_dict['rvsfwm_nightly'] = np.full(shape=(self.n_nights, self.n_template_fits), fill_value=np.nan)
        self.rvs_dict['uncfwm_nightly'] = np.full(shape=(self.n_nights, self.n_template_fits), fill_value=np.nan)
        
        # X Corr RVs
        self.rvs_dict["xcorr_options"] = self.xcorr_options
        if self.rvs_dict['xcorr_options']['method'] is not None:
            
            # Do x corr or not
            self.rvs_dict['do_xcorr'] = True
            
            # Number of velocities to try in the brute force or ccf
            self.rvs_dict["xcorr_options"]['n_vels'] = int(2 * self.xcorr_options['range'] / self.xcorr_options['step'])
            
            # Initiate arrays for xcorr rvs.
            self.rvs_dict['rvsxc'] = np.full(shape=(self.n_spec, self.n_chunks, self.n_template_fits), dtype=np.float64, fill_value=np.nan)
            self.rvs_dict['uncxc'] = np.full(shape=(self.n_spec, self.n_chunks, self.n_template_fits), dtype=np.float64, fill_value=np.nan)
            self.rvs_dict['rvsxc_nightly'] = np.full(shape=(self.n_nights, self.n_template_fits), dtype=np.float64, fill_value=np.nan)
            self.rvs_dict['uncxc_nightly'] = np.full(shape=(self.n_nights, self.n_template_fits), dtype=np.float64, fill_value=np.nan)
            self.rvs_dict['xcorrs'] = np.full(shape=(self.rvs_dict['xcorr_options']['n_vels'], self.n_spec, self.n_chunks, self.n_template_fits, 2), dtype=np.float64, fill_value=np.nan)
            self.rvs_dict['line_bisectors'] = np.full(shape=(self.xcorr_options['n_bs'], self.n_spec, self.n_chunks, self.n_template_fits), dtype=np.float64, fill_value=np.nan)
            self.rvs_dict['bis'] = np.full(shape=(self.n_spec, self.n_chunks, self.n_template_fits), dtype=np.float64, fill_value=np.nan)
            
        else:
            self.rvs_dict['do_xcorr'] = False
            
    def load_templates(self):
        self.templates_dict = {}
        for model in self[0].models_dict:
            if hasattr(self[0].models_dict[model], 'load_template'):
                self.templates_dict[model] = self[0].models_dict[model].load_template(self[0])
            
    def create_output_paths(self):
        self.run_output_path = self.output_path + self.tag + os.sep
        self.o_folder = 'Order' + str(self.order_num) + os.sep
        os.makedirs(self.run_output_path, exist_ok=True)
        os.makedirs(self.run_output_path + self.o_folder, exist_ok=True)
        os.makedirs(self.run_output_path + self.o_folder + 'ForwardModels', exist_ok=True)
        os.makedirs(self.run_output_path + self.o_folder + 'RVs', exist_ok=True)
        os.makedirs(self.run_output_path + self.o_folder + 'Templates', exist_ok=True)
    
    def init_optimize(self):
        if self[0].models_dict['star'].from_synthetic:
            self.get_init_star_vel_ccf()
        for fwm in self:
            fwm.init_optimize(self.templates_dict)

    def save_to_pickle(self):
        fname = self.run_output_path + self.o_folder + 'ForwardModels' + os.sep + self.tag + '_forward_models_ord' + str(self.order_num) + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
            
    def save_results(self):
        self.save_to_pickle()
        
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
            for ispec in range(self.n_spec):
                args_pass.append((self[ispec], self.templates_dict, iter_index))
            
            # Call the parallel job via joblib.
            self[:] = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(self[0].fit_all_regions_wrapper)(*args_pass[ispec]) for ispec in range(self.n_spec))

        else:
            # Fit one at a time
            for ispec in range(self.n_spec):
                print('    Performing Nelder-Mead Fit For Spectrum '  + str(ispec+1) + ' of ' + str(self.n_spec), flush=True)
                self[ispec] = self[0].fit_all_regions_wrapper(self[ispec], self.templates_dict, iter_index)
        
        # Cross correlate if set
        if self.rvs_dict['do_xcorr'] and self.n_template_fits > 0 and self[0].models_dict['star'].enabled:
            self.cross_correlate_spectra(iter_index)
            
        # Fit in Parallel
        print('Fitting Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)
        
    def cross_correlate_spectra(self, iter_index):
        """Cross correlation wrapper for all spectra.

        Args:
            iter_index (int or None): The iteration to use.
        """
        
        stopwatch = pcutils.StopWatch()
        
        print('Cross Correlating Spectra ... ', flush=True)
        
        # The method
        ccf_method = getattr(pcrvcalc, self.xcorr_options['method'])

        # Perform xcorr in series or parallel
        if self.n_cores > 1:

            # Construct the arguments
            iter_pass = []
            for ispec in range(self.n_spec):
                iter_pass.append((self[ispec], self.templates_dict, iter_index, self.rvs_dict['xcorr_options']))

            # Cross Correlate in Parallel
            #ccf_results = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(self[0].cross_correlate_all_regions_wrapper)(*iter_pass[ispec]) for ispec in tqdm.tqdm(range(self.n_spec)))
            ccf_results = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(self[0].cross_correlate_all_regions_wrapper)(*iter_pass[ispec]) for ispec in range(self.n_spec))
            
        else:
            ccf_results = [self[0].cross_correlate_all_regions_wrapper(self[ispec], self.templates_dict, iter_index, self.rvs_dict['xcorr_options']) for ispec in range(self.n_spec)]
            
        for ispec in range(self.n_spec):
            for ichunk in range(self.n_chunks):
                self.rvs_dict['rvsxc'][ispec, ichunk, iter_index] = ccf_results[ispec][ichunk]['rv']
                self.rvs_dict['uncxc'][ispec, ichunk, iter_index] = ccf_results[ispec][ichunk]['rv_unc']
                self.rvs_dict['bis'][ispec, ichunk, iter_index] = ccf_results[ispec][ichunk]['bis']
                self.rvs_dict['xcorrs'][:, ispec, ichunk, iter_index, 0] = ccf_results[ispec][ichunk]['vels']
                self.rvs_dict['xcorrs'][:, ispec, ichunk, iter_index, 1] = ccf_results[ispec][ichunk]['ccf']
                
        print('Cross Correlation Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)
        
    def compute_nightly_rvs(self, iter_index):
        """Genreates individual and nightly (co-added) RVs after forward modeling all spectra and stores them in the ForwardModels object. If do_xcorr is True, nightly cross-correlation RVs are also computed.

        Args:
            iter_index (int): The iteration to generate RVs from.
        """

        # The best fit stellar RVs, remove the barycenter velocity
        rvsfwm = np.full((self.n_spec, self.n_chunks), np.nan)
        for ispec in range(self.n_spec):
            for ichunk in range(self.n_chunks):
                rvsfwm[ispec, ichunk] = self[ispec].opt_results[-1][ichunk]['xbest'][self[ispec].models_dict['star'].par_names[0]].value + self[ispec].data.bc_vel
        
        # The RMS from the forward model fit
        fit_metric = np.full((self.n_spec, self.n_chunks), np.nan)
        for ispec in range(self.n_spec):
            for ichunk in range(self.n_chunks):
                fit_metric[ispec, ichunk] = self[ispec].opt_results[-1][ichunk]['fbest']
        weights = 1 / fit_metric**2
        
        # The NM RVs
        rvsfwm_nightly, uncfwm_nightly = pcrvcalc.compute_nightly_rvs_single_order(rvsfwm, weights, self.rvs_dict['n_obs_nights'], flag_outliers=True)
        self.rvs_dict['rvsfwm'][:, :, iter_index] = rvsfwm
        self.rvs_dict['rvsfwm_nightly'][:, iter_index] = rvsfwm_nightly
        self.rvs_dict['uncfwm_nightly'][:, iter_index] = uncfwm_nightly
        
        # The xcorr RVs
        if self.rvs_dict['do_xcorr']:
            rvsx_nightly, uncx_nightly = pcrvcalc.compute_nightly_rvs_single_order(self.rvs_dict['rvsxc'][:, :, iter_index], weights, self.rvs_dict['n_obs_nights'], flag_outliers=True)
            self.rvs_dict['rvsxc_nightly'][:, iter_index] = rvsx_nightly
            self.rvs_dict['uncxc_nightly'][:, iter_index] = uncx_nightly
        
    def save_rvs(self):
        """Saves the forward model results and RVs.
        """
        fname = self.run_output_path + self.o_folder + 'RVs' + os.sep + self.tag + '_rvs_ord' + str(self[0].order_num) + '.npz'
        #bc_vels = np.array([fwm.data.bc_vel for fwm in self], dtype=float)
        
        # Save in a .npz file for easy access later
        np.savez(fname, **self.rvs_dict)
    
    def plot_rvs(self, iter_index):
        """Plots all RVs and cross-correlation analysis after forward modeling all spectra.

        Args:
            iter_index (int): The iteration to use.
        """
        
        # Plot the rvs, nightly rvs, xcorr rvs, xcorr nightly rvs
        plot_width, plot_height = 1800, 600
        dpi = 200
        plt.figure(num=1, figsize=(int(plot_width/dpi), int(plot_height/dpi)), dpi=200)
        
        # Alias
        rvs_dict = self.rvs_dict
        
        # Plot the individual rvs for each chunk (n_chunks * n_spec)
        for ichunk in range(self.n_chunks):
        
            # Individual rvs from nelder mead fitting
            plt.plot(rvs_dict['bjds'] - rvs_dict['bjds_nightly'][0],
                    rvs_dict['rvsfwm'][:, ichunk, iter_index] - np.nanmedian(rvs_dict['rvsfwm'][:, ichunk, iter_index]),
                    marker='.', linewidth=0, alpha=0.7, color=(0.1, 0.8, 0.1))

            # Individual and nightly xcorr rvs
            if rvs_dict['do_xcorr']:
                plt.errorbar(rvs_dict['bjds'] - rvs_dict['bjds_nightly'][0],
                            rvs_dict['rvsxc'][:, ichunk, iter_index] - np.nanmedian(rvs_dict['rvsxc'][:, ichunk, iter_index]),
                            yerr=rvs_dict['uncxc'][:, ichunk, iter_index],
                            marker='.', linewidth=0, color='black', alpha=0.6, elinewidth=0.8)
        
        
        # Nightly RVs from nelder mead fitting
        plt.errorbar(rvs_dict['bjds_nightly'] - rvs_dict['bjds_nightly'][0],
                        rvs_dict['rvsfwm_nightly'][:, iter_index] - np.nanmedian(rvs_dict['rvsfwm_nightly'][:, iter_index]),
                        yerr=rvs_dict['uncfwm_nightly'][:, iter_index], marker='o', linewidth=0, elinewidth=1, label='Nelder Mead', color=(0, 114/255, 189/255))
        
        # Nightly RVs from xc
        if rvs_dict['do_xcorr']:
            plt.errorbar(rvs_dict['bjds_nightly'] - rvs_dict['bjds_nightly'][0],
                                    rvs_dict['rvsxc_nightly'][:, iter_index] - np.nanmedian(rvs_dict['rvsxc_nightly'][:, iter_index]),
                                    yerr=rvs_dict['uncxc_nightly'][:, iter_index], marker='X', linewidth=0, alpha=0.8, label='X Corr', color='darkorange', elinewidth=1)
        
        plt.title(self[0].star_name + ' RVs Order ' + str(self.order_num) + ', Iteration ' + str(iter_index + 1), fontweight='bold')
        plt.xlabel('BJD - BJD$_{0}$', fontweight='bold')
        plt.ylabel('RV [m/s]', fontweight='bold')
        plt.legend(loc='upper right')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fname = self.run_output_path + self.o_folder + 'RVs' + os.sep + self.tag + '_rvs_ord' + str(self.order_num) + '_iter' + str(iter_index + 1) + '.png'
        plt.savefig(fname)
        plt.close()
        
        if rvs_dict['do_xcorr']:
            plt.figure(1, figsize=(12, 7), dpi=200)
            for ichunk in range(self.n_chunks):
                for ispec in range(self.n_spec):
                    v0 = rvs_dict['rvsxc'][ispec, ichunk, iter_index]
                    depths = np.linspace(0, 1, num=rvs_dict['xcorr_options']['n_bs'])
                    ccf_ = rvs_dict['xcorrs'][:, ispec, ichunk, iter_index, 1] - np.nanmin(rvs_dict['xcorrs'][:, ispec, ichunk, iter_index, 1])
                    ccf_ = ccf_ / np.nanmax(ccf_)
                    plt.plot(rvs_dict['xcorrs'][:, ispec, ichunk, iter_index, 0] - v0, ccf_)
                    
            plt.title(self.star_name + ' CCFs Order ' + str(self.order_num) + ', Iteration ' + str(iter_index + 1), fontweight='bold')
            plt.xlabel('RV$_{\star}$ [m/s]', fontweight='bold')
            plt.ylabel('CCF (RMS surface)', fontweight='bold')
            plt.xlim(-10000, 10000)
            plt.tight_layout()
            fname = self.run_output_path + self.o_folder + 'RVs' + os.sep + self.tag + '_ccfs_ord' + str(self.order_num) + '_iter' + str(iter_index + 1) + '.png'
            plt.savefig(fname)
            plt.close()
        
            # Plot the bis
            plt.figure(1, figsize=(12, 7), dpi=200)
            for ichunk in range(self.n_chunks):
                plt.plot(rvs_dict['rvsxc'][:, ichunk, iter_index], rvs_dict['bis'][:, ichunk, iter_index], marker='o', linewidth=0)
                plt.title(self[0].star_name + ' CCF Bisector Spans Order ' + str(self.order_num) + ', Iteration ' + str(iter_index + 1), fontweight='bold')
                plt.xlabel('X Corr RV [m/s]', fontweight='bold')
                plt.ylabel('Bisector Span [m/s]', fontweight='bold')
            plt.tight_layout()
            fname = self.run_output_path + self.o_folder + 'RVs' + os.sep + self.tag + '_bisectorspans_ord' + str(self.order_num) + '_iter' + str(iter_index + 1) + '.png'
            plt.savefig(fname)
            plt.close()
        
    def get_init_star_vel_ccf(self):
        """Cross correlation wrapper for all spectra.

        Args:
            iter_index (int or None): The iteration to use. If None, then it's assumed to be a crude first guess.
        """
        stopwatch = pcutils.StopWatch()
        
        print('Cross Correlating Spectra To Determine Crude RV ... ', flush=True)
        
        # The method
        ccf_method = pcrvcalc.crude_brute_force

        # Perform xcorr in series or parallel
        if self.n_cores > 1:

            # Construct the arguments
            iter_pass = []
            for ispec in range(self.n_spec):
                iter_pass.append((self[ispec], self.templates_dict, self[ispec].sregion_order))

            # Cross Correlate in Parallel
            #ccf_results = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(ccf_method)(*iter_pass[ispec]) for ispec in tqdm.tqdm(range(self.n_spec)))
            ccf_results = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(ccf_method)(*iter_pass[ispec]) for ispec in range(self.n_spec))
        else:
            #ccf_results = [ccf_method(fwm, self.templates_dict, fwm.sregion_order) for fwm in tqdm.tqdm(self)]
            ccf_results = [ccf_method(fwm, self.templates_dict, fwm.sregion_order) for fwm in self]
            
        for ispec, fwm in enumerate(self):
            if ccf_results[ispec] != 0:
                fwm.initial_parameters[fwm.models_dict['star'].par_names[0]].value = ccf_results[ispec]
            else:
                fwm.initial_parameters[fwm.models_dict['star'].par_names[0]].value = ccf_results[ispec] + 10
                
        print('Cross Correlation Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)


class ForwardModel:
    
    def __init__(self, input_file, config, model_blueprints, parser, order_num, spec_num):
        
        # The echelle order
        self.order_num = order_num
        
        # The parser
        self.parser = parser
        
        # The spectral number and index
        self.spec_num = spec_num
        self.spec_index = self.spec_num - 1
        
        # Auto-populate
        for key in config:
            setattr(self, key, copy.deepcopy(config[key]))

        # The proper tag
        self.tag = self.spectrograph.lower() + '_' + self.tag
        self.run_output_path = self.output_path + self.tag + os.sep
        self.o_folder = 'Order' + str(self.order_num) + os.sep
        
        # Overwrite the target function with the actual function to optimize the model
        self.target_function = getattr(pctargetfuns, self.target_function)
        
        # Initialize the data
        self.data = pcdata.SpecData1d.from_forward_model(input_file, self)
        
        # Init the models
        self.init_models(config, model_blueprints)
        
        # N iters
        self.n_iters_rvs = self.n_template_fits
        self.index_offset = int(not self.models_dict['star'].from_synthetic)
        self.n_iters_opt = self.n_template_fits + self.index_offset
    
        # Storage arrays after each iteration
        # Each entry is a tuple for each iteration: (best_fit_pars, RMS, FCALLS)
        self.opt_results = []

    def init_chunks(self, model_blueprints):
        good = np.where(self.data.mask)[0]
        order_pixmin, order_pixmax = good[0], good[-1]
        wave_class = getattr(pcmodels, model_blueprints['wavelength_solution']['class'])
        self.chunk_regions = []
        stitch_points_pix = np.linspace(order_pixmin, order_pixmax, num=self.n_chunks + 1).astype(int)
        wave_estimate = wave_class.estimate_order_wave(self, model_blueprints["wavelength_solution"])
        self.sregion_order = pcutils.SpectralRegion(order_pixmin, order_pixmax, wave_estimate[order_pixmin], wave_estimate[order_pixmax], label="order")
        self.dl = (1 / self.sregion_order.pix_per_wave()) / self.model_resolution
        for ichunk in range(self.n_chunks):
            pixmin, pixmax = stitch_points_pix[ichunk], stitch_points_pix[ichunk + 1]
            wavemin, wavemax = wave_estimate[pixmin], wave_estimate[pixmax]
            self.chunk_regions.append(pcutils.SpectralRegion(pixmin, pixmax, wavemin, wavemax, label=ichunk))

    def init_models(self, config, model_blueprints):
        
        # A dictionary to store model components
        self.models_dict = {}

        # First generate the wavelength solution model
        model_class = getattr(pcmodels, model_blueprints['wavelength_solution']['class'])
        
        # Init the chunks
        self.init_chunks(model_blueprints)
        
        self.models_dict['wavelength_solution'] = model_class(self, model_blueprints['wavelength_solution'])
        
        # Define the LSF model if present
        if 'lsf' in model_blueprints:
            model_class_init = getattr(pcmodels, model_blueprints['lsf']['class'])
            self.models_dict['lsf'] = model_class_init(self, model_blueprints['lsf'])
        
        # Generate the remaining model components from their blueprints and load any input templates
        # All remaining model components should subtype MultComponent
        for blueprint in model_blueprints:
            
            if blueprint in self.models_dict:
                continue
            
            # Construct the model
            model_class = getattr(pcmodels, model_blueprints[blueprint]['class'])
            self.models_dict[blueprint] = model_class(self, model_blueprints[blueprint])

    def init_parameters(self, sregion=None):
        self.initial_parameters = OptimParameters.Parameters()
        for model in self.models_dict:
            self.models_dict[model].init_parameters(self)
        self.initial_parameters.sanity_lock()

    def init_optimize(self, templates_dict):
        templates_dict_chunked = self.init_chunk(templates_dict, self.sregion_order)
        for model in self.models_dict:
            self.models_dict[model].init_optimize(self, templates_dict_chunked)

    # Prints the models and corresponding parameters after each fit if verbose_print=True
    def pretty_print(self):
        # Loop over models
        for mname in self.models_dict.keys():
            if not self.models_dict[mname].enabled:
                continue
            # Print the model string
            print(self.models_dict[mname], flush=True)
            # Sub loop over per model parameters
            for pname in self.models_dict[mname].par_names:
                print('    ', end='', flush=True)
                if len(self.opt_results) == 0:
                    print(self.initial_parameters[pname], flush=True)
                else:
                    print(self.opt_results[-1][-1]['xbest'][pname], flush=True)

    def set_parameters(self, pars):
        self.initial_parameters.update(pars)
    
    # Plots the forward model after each iteration with other template as well if verbose_plot = True
    def plot_data_model(self, templates_dict, iter_index):
        
        # The filename
        if self.models_dict['star'].enabled:
            fname = self.run_output_path + self.o_folder + 'ForwardModels' + os.sep + self.tag + '_data_model_spec' + str(self.spec_num) + '_ord' + str(self.order_num) + '_iter' + str(iter_index + 1) + '.png'
        else:
            fname = self.run_output_path + self.o_folder + 'ForwardModels' + os.sep + self.tag + '_data_model_spec' + str(self.spec_num) + '_ord' + str(self.order_num) + '_iter0.png'
            
        # Figure
        plot_width, plot_height = 2000, 720
        dpi = 200
        fig, axarr = plt.subplots(self.n_chunks, 1, figsize=(int(plot_width / dpi), int(self.n_chunks * plot_height / dpi)), dpi=dpi, constrained_layout=True)
        axarr = np.atleast_1d(axarr)
        
        for ichunk, sregion in enumerate(self.chunk_regions):
            
            # The best fit parameters
            pars = self.opt_results[-1][sregion.label]['xbest']
            
            # Init chunk
            templates_dict_chunked = self.init_chunk(templates_dict, sregion)
        
            # Build the model
            wave_data, model_lr = self.build_full(pars, templates_dict)
        
            # The residuals for this iteration
            residuals = self.data.flux_chunk  - model_lr

            # Define some helpful indices
            good = np.where(self.data.mask_chunk == 1)[0]
            bad = np.where(self.data.mask_chunk == 0)[0]
            if self.flag_n_worst_pixels > 0:
                bad_data_locs = np.argsort(np.abs(residuals[good]))[-1*self.flag_n_worst_pixels:]
        
            # Left and right padding
            pad = 0.01 * sregion.wave_len()
            
            # Data
            axarr[ichunk].plot(wave_data / 10, self.data.flux_chunk, color=(0, 114/255, 189/255), lw=0.8)
            
            # Model
            axarr[ichunk].plot(wave_data / 10, model_lr, color=(217/255, 83/255, 25/255), lw=0.8)
            
            # Zero line
            axarr[ichunk].plot(wave_data / 10, np.zeros(sregion.pix_len()), color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':')
            
            # Residuals
            axarr[ichunk].plot(wave_data[good] / 10, residuals[good], color=(255/255, 169/255, 22/255), lw=0.8)
            
            # The worst N pixels that were flagged
            if self.flag_n_worst_pixels > 0:
                axarr[ichunk].plot(wave_data[good][bad_data_locs] / 10, residuals[good][bad_data_locs], color='darkred', marker='X', lw=0)
            
            # Plot the convolved low res templates for debugging 
            # Plots the star and tellurics by default. Plots gas cell if present.
            if self.verbose_plot:
                
                lsf = self.models_dict['lsf'].build(pars=pars)
                
                # Extra zero line
                axarr[ichunk].plot(wave_data / 10, np.zeros(sregion.pix_len()) - 0.1, color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':', alpha=0.8)
                
                # Star
                if self.models_dict['star'].enabled:
                    star_flux_hr = self.models_dict['star'].build(pars, templates_dict_chunked['star'], self.model_wave)
                    star_convolved = self.models_dict['lsf'].convolve_flux(star_flux_hr, lsf=lsf)
                    star_flux_lr = cspline_interp(self.model_wave, star_convolved, wave_data)
                    axarr[ichunk].plot(wave_data / 10, star_flux_lr - 1.1, label='Star', lw=0.8, color='deeppink', alpha=0.8)
                
                # Tellurics
                if 'tellurics' in self.models_dict and self.models_dict['tellurics'].enabled:
                    tellurics = self.models_dict['tellurics'].build(pars, templates_dict_chunked['tellurics'], self.model_wave)
                    tellurics_convolved = self.models_dict['lsf'].convolve_flux(tellurics, lsf=lsf)
                    tell_flux_lr = cspline_interp(self.model_wave, tellurics_convolved, wave_data)
                    axarr[ichunk].plot(wave_data / 10, tell_flux_lr - 1.1, label='Tellurics', lw=0.8, color='indigo', alpha=0.8)
                
                # Gas Cell
                if 'gas_cell' in self.models_dict and self.models_dict['gas_cell'].enabled:
                    gas_flux_hr = self.models_dict['gas_cell'].build(pars, templates_dict_chunked['gas_cell'], self.model_wave)
                    gas_cell_convolved = self.models_dict['lsf'].convolve_flux(gas_flux_hr, lsf=lsf)
                    gas_flux_lr = cspline_interp(self.model_wave, gas_cell_convolved, wave_data)
                    axarr[ichunk].plot(wave_data / 10, gas_flux_lr - 1.1, label='Gas Cell', lw=0.8, color='green', alpha=0.8)
                axarr[ichunk].set_ylim(-1.1, 1.1)
                
                # Residual lab flux
                if 'residual_lab' in templates_dict:
                    res_hr = templates_dict['residual_lab'][:, 1]
                    res_lr = cspline_interp(self.model_wave, res_hr, wave_data)
                    ax.plot(wave_data / 10, res_lr - 0.1, label='Lab Frame Coherence', lw=0.8, color='darkred', alpha=0.8)
                    
                axarr[ichunk].legend(prop={'size': 8}, loc='lower right')
            else:
                axarr[ichunk].set_ylim(-0.1, 1.1)
                
            axarr[ichunk].tick_params(axis='both', labelsize=10)
            axarr[ichunk].set_xlim((sregion.wavemin - pad) / 10, (sregion.wavemax + pad) / 10)
            fig.text(0.5, 0.015, 'Wavelength [nm]', fontsize=10, horizontalalignment='center', verticalalignment='center')
            fig.text(0.015, 0.5, 'Data, Model, Residuals', fontsize=10, rotation=90, verticalalignment='center', horizontalalignment='center')
        
        # Save
        #plt.subplots_adjust(left=0.05, bottom=0.05, right=None, top=0.95, wspace=None, hspace=None)
        plt.savefig(fname)
        plt.close()

    def init_chunk(self, templates_dict, sregion=None):
        
        if sregion is None:
            sregion = self.sregion_order
            
        templates_dict_chunked = copy.deepcopy(templates_dict)
        
        self.model_wave = np.arange(sregion.wavemin - 5, sregion.wavemax + 5, self.dl)
        
        # Init the models
        for model in self.models_dict:
            self.models_dict[model].init_chunk(self, templates_dict_chunked, sregion)

        try:
            p0_copy = copy.deepcopy(self.initial_parameters)
            self.initial_parameters = self.opt_results[-2][sregion.label]['xbest']
            for pname in self.initial_parameters:
                self.initial_parameters[pname].vary = p0_copy[pname].vary
        except:
            pass
        
        self.data.flux_chunk = self.data.flux[sregion.data_inds] / pcmath.weighted_median(self.data.flux[sregion.data_inds], percentile=0.98)
        self.data.flux_unc_chunk = self.data.flux_unc[sregion.data_inds]
        self.data.mask_chunk = self.data.mask[sregion.data_inds]

        return templates_dict_chunked

    # Save the forward model object to a pickle
    def save_to_pickle(self):
        fname = self.run_output_path + self.o_folder + 'ForwardModels' + os.sep + self.tag + '_forward_model_ord' + str(self.order_num) + '_spec' + str(self.spec_num) + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    # Gets the night which corresponds to the spec index
    def get_thisnight_index(self, n_obs_nights):
        return self.get_night_index(self.spec_index, n_obs_nights)

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
        return self.get_all_spec_indices_from_night(night_index, n_obs_nights)
    
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


    # Wrapper for parallel processing. Solves and plots the forward model results. Also does xcorr if set.
    @staticmethod
    def fit_region_wrapper(forward_model, templates_dict, iter_index, sregion):
        """A wrapper for forward modeling and cross-correlating a single spectrum.

        Args:
            forward_model (ForwardModel): The forward model object
            iter_index (int): The iteration index.
            output_path_plot (str, optional): output path for plots. Defaults to None and uses object default.
            verbose_print (bool, optional): Whether or not to print optimization results. Defaults to False.
            verbose_plot (bool, optional): Whether or not to plot templates with the forward model. Defaults to False.

        Returns:
            forward_model (ForwardModel): The updated forward model since we possibly fit in parallel.
        """
        
        # Init this region
        templates_dict_chunked = forward_model.init_chunk(templates_dict, sregion)
        
        # Construct the extra arguments to pass to the target function
        args_to_pass = (forward_model, templates_dict_chunked, sregion)
    
        # Construct the Nelder Mead Solver and run
        solver = NelderMead(forward_model.target_function, forward_model.initial_parameters, no_improve_break=3, args_to_pass=args_to_pass, ftol=1E-6, xtol=1E-6)
        opt_result = solver.solve()
        
        # Pass best fit parameters and optimization result to forward model
        forward_model.opt_results[-1].append({'xbest': opt_result['xmin'], 'fbest': opt_result['fmin'], 'fcalls': opt_result['fcalls']})

        # Return new forward model object since we possibly fit in parallel
        return forward_model
    
    # Wrapper for parallel processing. Solves and plots the forward model results. Also does xcorr if set.
    @staticmethod
    def fit_all_regions_wrapper(forward_model, templates_dict, iter_index):
        """A wrapper for forward modeling and cross-correlating a single spectrum.

        Args:
            forward_model (ForwardModel): The forward model object
            iter_index (int): The iteration index.
            output_path_plot (str, optional): output path for plots. Defaults to None and uses object default.
            verbose_print (bool, optional): Whether or not to print optimization results. Defaults to False.
            verbose_plot (bool, optional): Whether or not to plot templates with the forward model. Defaults to False.

        Returns:
            forward_model (ForwardModel): The updated forward model since we possibly fit in parallel.
        """
        
        # Start the timer
        stopwatch = pcutils.StopWatch()
        
        # Add an iteration to the opt_results list
        forward_model.opt_results.append([])
        
        for ichunk, sregion in enumerate(forward_model.chunk_regions):
            
            # Fit chunk
            forward_model = forward_model.fit_region_wrapper(forward_model, templates_dict, iter_index, sregion)

            # Print diagnostics if set
            if forward_model.verbose_print:
                print('Function Val = ' + str(round(forward_model.opt_results[-1][-1]['fbest'], 5)), flush=True)
                print('Function Calls = ' + str(forward_model.opt_results[-1][-1]['fcalls']), flush=True)
                forward_model.pretty_print()
        
        print('Fit Spectrum ' + str(forward_model.spec_num) + ' in ' + str(round((stopwatch.time_since())/60, 2)) + ' min', flush=True)

        # Output a plot
        if forward_model.gen_fwm_plots:
            forward_model.plot_data_model(templates_dict, iter_index)

        # Return new forward model object since we possibly fit in parallel
        return forward_model
    
    @staticmethod
    def cross_correlate_region_wrapper(forward_model, templates_dict, iter_index, sregion, xcorr_options):
        """A wrapper for forward modeling and cross-correlating a single spectrum.

        Args:
            forward_model (ForwardModel): The forward model object
            iter_index (int): The iteration index.
            output_path_plot (str, optional): output path for plots. Defaults to None and uses object default.
            verbose_print (bool, optional): Whether or not to print optimization results. Defaults to False.
            verbose_plot (bool, optional): Whether or not to plot templates with the forward model. Defaults to False.

        Returns:
            forward_model (ForwardModel): The updated forward model since we possibly fit in parallel.
        """
        
        # Init this region
        templates_dict_chunked = forward_model.init_chunk(templates_dict, sregion)
        
        ccf_method = getattr(pcrvcalc, xcorr_options["method"])
        
        # Run the CCF
        ccf_result = ccf_method(forward_model, templates_dict_chunked, iter_index, sregion, xcorr_options)

        # Return new forward model object since we possibly fit in parallel
        return ccf_result
    
    # Wrapper for parallel processing. Solves and plots the forward model results. Also does xcorr if set.
    @staticmethod
    def cross_correlate_all_regions_wrapper(forward_model, templates_dict, iter_index, xcorr_options):
        """A wrapper for forward modeling and cross-correlating a single spectrum.

        Args:
            forward_model (ForwardModel): The forward model object
            iter_index (int): The iteration index.
            output_path_plot (str, optional): output path for plots. Defaults to None and uses object default.
            verbose_print (bool, optional): Whether or not to print optimization results. Defaults to False.
            verbose_plot (bool, optional): Whether or not to plot templates with the forward model. Defaults to False.

        Returns:
            forward_model (ForwardModel): The updated forward model since we possibly fit in parallel.
        """
        
        # Start the timer
        stopwatch = pcutils.StopWatch()
        
        ccf_results = []
        
        for ichunk, sregion in enumerate(forward_model.chunk_regions):
            
            # CCF chunk
             ccf_results.append(forward_model.cross_correlate_region_wrapper(forward_model, templates_dict, iter_index, sregion, xcorr_options))

        return ccf_results
    
    def build_full(self, pars, templates_dict):
        
        # Init a model
        model = np.ones_like(self.model_wave)

        # Star
        if 'star' in self.models_dict and self.models_dict['star'].enabled:
            model *= self.models_dict['star'].build(pars, templates_dict['star'], self.model_wave)
        
        # Gas Cell
        if 'gas_cell' in self.models_dict and self.models_dict['gas_cell'].enabled:
            model *= self.models_dict['gas_cell'].build(pars, templates_dict['gas_cell'], self.model_wave)
            
        # All tellurics
        if 'tellurics' in self.models_dict and self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], self.model_wave)
        
        # Fringing from who knows what
        if 'fringing' in self.models_dict and self.models_dict['fringing'].enabled:
            model *= self.models_dict['fringing'].build(pars, self.model_wave)
            
        # Convolve
        if 'lsf' in self.models_dict and self.models_dict['lsf'].enabled:
            model[:] = self.models_dict['lsf'].convolve_flux(model, pars)
            
            # Renormalize model to remove degeneracy between blaze and lsf
            model /= np.nanmax(model)
            
        # Continuum
        if 'continuum' in self.models_dict and self.models_dict['continuum'].enabled:
            model *= self.models_dict['continuum'].build(pars, self.model_wave)
        
        # Residual lab flux
        if 'residual_lab' in templates_dict:
            model += templates_dict['residual_lab'][:, 1]

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = cspline_interp(self.model_wave, model, wavelength_solution)
        
        if self.debug:
            breakpoint()

        return wavelength_solution, model_lr
    
    # Returns the high res model on the fiducial grid with no stellar template and the low res wavelength solution
    def build_hr_nostar(self, pars, templates_dict):
        
        # Init a model
        model = np.ones_like(self.model_wave)
        
        # Gas Cell
        if 'gas_cell' in self.models_dict and self.models_dict['gas_cell'].enabled:
            model *= self.models_dict['gas_cell'].build(pars, templates_dict['gas_cell'], self.model_wave)
            
        # All tellurics
        if 'tellurics' in self.models_dict and self.models_dict['tellurics'].enabled:
            model *= self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], self.model_wave)
        
        # Fringing from who knows what
        if 'fringing' in self.models_dict and self.models_dict['fringing'].enabled:
            model *= self.models_dict['fringing'].build(pars, self.model_wave)
            
        # Continuum
        if 'continuum' in self.models_dict and self.models_dict['continuum'].enabled:
            model *= self.models_dict['continuum'].build(pars, self.model_wave)
        
        # Residual lab flux
        if 'residual_lab' in templates_dict:
            model += templates_dict['residual_lab'][:, 1]

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = cspline_interp(self.model_wave, model, wavelength_solution)
        
        if self.debug:
            breakpoint()

        return wavelength_solution, model_lr
  
class iSHELLForwardModel(ForwardModel):
    pass
        
class CHIRONForwardModel(ForwardModel):
    pass

class PARVIForwardModel(ForwardModel):
    pass

class IRDForwardModel(ForwardModel):
    pass

class MinervaAustralisForwardModel(ForwardModel):
    pass
    
class MinervaNorthForwardModel(ForwardModel):
    pass
    
class NIRSPECForwardModel(ForwardModel):
    pass
    
class SimulatedForwardModel(ForwardModel):
    pass
    