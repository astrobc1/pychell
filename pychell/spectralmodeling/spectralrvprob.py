# Base Python
import importlib
import copy
import pickle
import os

# Maths
import numpy as np

# Multiprocessing
from joblib import Parallel, delayed

# pychell
import pychell
import pychell.maths as pcmath
import pychell.spectralmodeling.rvcalc as pcrvcalc
import pychell.utils as pcutils
from pychell.data.spectraldata import SpecData1d
from pychell.spectralmodeling.spectralmodels import IterativeSpectralForwardModel

# Plots
import matplotlib.pyplot as plt
try:
    plt.style.use(f"{os.path.dirname(pychell.__file__) + os.sep}gadfly_stylesheet.mplstyle")
except:
    print("Could not locate gadfly stylesheet, using default matplotlib stylesheet.")

# Optimize
from optimize.frameworks import OptProblem

######################
#### PRIMARY TYPE ####
######################

class IterativeSpectralRVProb(OptProblem):
    """The primary container for a spectral forward model problem where the goal is to provide precise RVs.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, spectrograph,
                 data_input_path, filelist,
                 spectral_model,
                 augmenter,
                 tag, output_path, target_dict,
                 bc_corrs=None,
                 optimizer=None, obj=None,
                 n_cores=1, verbose=True):
        """Initiate the top level iterative spectral rv problem object.

        Args:
            spectrograph (str): The name of the spectrograph.
            data_input_path (str): The full path to the data folder.
            filelist (str): A text file listing the observations (filenames) within data_input_path to use.
            spectral_model (IterativeSpectralForwardModel): The spectral model obejct. For now only IterativeSpectralForwardModel is supported.
            augmenter (TemnplateAugmenter): The template augmenter object.
            tag (str): A tag to uniquely identify this run in the outputs. The full tag will be spectrograph_tag.
            output_path (str): The output path. All outputs wioll be stored within a single sub folder within output_path, which will also contain multiple sub folders.
            target_dict (dict): The information for this target. For now, only the name key is used to generate the barycenter corrections (BJDs and barycenter velocity corrections) using Simbad to obtain the necessary inormation.
            bc_corrs (np.ndarray, optional): The barycenter corrections may be passed manually as a two column numpy array; shape=(n_observations, 2). Defaults to None and the barycenter correcitons are computed with barycorrpy from information pulled form Simbad.
            optimizer (Optimizer, optional): The optimizer to use. Defaults to None.
            obj (SpectralObjective, optional): The objective function to ultimiately extremize. Defaults to None.
            n_cores (int, optional): The number of cores to use. Defaults to 1.
            verbose (bool, optional): Whether or not to print additional diagnostics ater each fit. This should be False for long runs. Defaults to True.
        """
        
        # The number of cores
        self.n_cores = n_cores
        
        # Verbose
        self.verbose = verbose
        
        # Input path
        self.data_input_path = data_input_path
        self.filelist = filelist
        
        # The base output path
        self.output_path = output_path
        
        # The spectral model
        self.spectral_model = spectral_model

        # The spectrogaph
        self.spectrograph = spectrograph
        self._init_spectrograph()
        
        # Full tag is spectrograph_ + tag
        self.tag = f"{self.spectrograph.lower()}_{tag}"
        
        # The actual output path
        self.output_path += self.tag + os.sep
        self.create_output_paths()
        
        # Initialize the data
        self._init_data()
        
        # Optimize results
        self.opt_results = np.empty(shape=(self.n_spec, self.n_iterations), dtype=dict)
        self.stellar_templates = np.empty(self.n_iterations, dtype=np.ndarray)
        
        # Initialize the spectral model
        self._init_spectral_model()
        self.p0cp = copy.deepcopy(self.p0)
        
        # The target dictionary
        self.target_dict = target_dict
        
        # The template augmenter
        self.augmenter = augmenter
        
        # The objective function
        self.obj = obj
        
        # The optimizer
        self.optimizer = optimizer
        
        # Init RVs
        self._init_rvs(bc_corrs=bc_corrs)
        
        # Print summary
        self._print_init_summary()
            
    def _init_data(self):
        
        # List of input files
        input_files = [self.data_input_path + f for f in np.atleast_1d(np.genfromtxt(self.data_input_path + self.filelist, dtype='<U100', comments='#').tolist())]
        
        # Load in each observation for this order
        self.data = [SpecData1d(fname, self.order_num, ispec + 1, self.parser, self.crop_pix) for ispec, fname in enumerate(input_files)]
            
        # Estimate the wavelength bounds for this order
        wave_grid = self.parser.estimate_wavelength_solution(self.data[0])
        good = np.where(self.data[0].mask == 1)[0]
        pixmin, pixmax = np.max([good[0] - 5, 0]), np.min([good[-1] + 5, len(self.data[0].mask) - 1])
        wavemin, wavemax = wave_grid[pixmin], wave_grid[pixmax]

    def _init_spectrograph(self):
        
        # Load the spectrograph module
        spec_module = self.spec_module
        
        # Construct the data parser
        parser_class = getattr(spec_module, f"{self.spectrograph}Parser")
        self.parser = parser_class(self.data_input_path, self.output_path)

    def _init_spectral_model(self):
        self.spectral_model._init_templates(self.data)
        if self.spectral_model.star is not None and not self.spectral_model.star.from_flat:
            self.stellar_templates[0] = np.copy(self.spectral_model.templates_dict["star"])
        self.spectral_model._init_parameters(self.data)

    def _init_rvs(self, bc_corrs=None):
        
        # Get the spectrograph observatory
        spec_module = self.spec_module
        observatory = spec_module.observatory
        
        # Store all rv info in a dict
        self.rvs_dict = {}
        
        # Individual and per-night BJD
        if bc_corrs is None:
            self.rvs_dict["bjds"] = np.full(self.n_spec, np.nan)
            self.rvs_dict["bc_vels"] = np.full(self.n_spec, np.nan)
            for i in range(self.n_spec):
                self.rvs_dict["bjds"][i], self.rvs_dict["bc_vels"][i] = self.parser.compute_barycenter_corrections(self.data[i], observatory, self.target_dict)
        else:
            bc_corrs = np.atleast_2d(bc_corrs)
            for i in range(self.n_spec):
                self.data[i].bjd = bc_corrs[i, 0]
                self.data[i].bc_vel = bc_corrs[i, 1]
            self.rvs_dict["bjds"] = bc_corrs[:, 0]
            self.rvs_dict["bc_vels"] = bc_corrs[:, 1]
        
        # Get the nightly jds
        self.rvs_dict["bjds_nightly"], self.rvs_dict["n_obs_nights"] = pcrvcalc.gen_nightly_jds(self.rvs_dict["bjds"])
        
        # Individual and per-night Forward Modeled RVs
        self.rvs_dict["rvsfwm"] = np.full((self.n_spec, self.n_iterations), np.nan)
        self.rvs_dict["rvsfwm_nightly"] = np.full((self.n_nights, self.n_iterations), np.nan)
        self.rvs_dict["uncfwm_nightly"] = np.full((self.n_nights, self.n_iterations), np.nan)
        
        # Individual and per-night XC RVs
        self.rvs_dict["rvsxc"] = np.full((self.n_spec, self.n_iterations), np.nan)
        self.rvs_dict['uncxc'] = np.full((self.n_spec, self.n_iterations), np.nan)
        self.rvs_dict["rvsxc_nightly"] = np.full((self.n_nights, self.n_iterations), np.nan)
        self.rvs_dict["uncxc_nightly"] = np.full((self.n_nights, self.n_iterations), np.nan)
        
        # BIS info
        self.rvs_dict['bis'] = np.full((self.n_spec, self.n_iterations), np.nan)
        self.rvs_dict['bis_nightly'] = np.full((self.n_nights, self.n_iterations), np.nan)
        self.rvs_dict['uncbis_nightly'] = np.full((self.n_nights, self.n_iterations), np.nan)
        
        # xc grid info
        self.rvs_dict['xcorrs'] = np.empty(shape=(self.n_spec, self.n_iterations), dtype=np.ndarray)
        
    def _print_init_summary(self):
        print("***************************************", flush=True)
        print(f"** Target: {self.target_dict['name'].replace('_', ' ')}", flush=True)
        print(f"** Spectrograph: {self.spec_module.observatory['name']} / {self.spectrograph}", flush=True)
        print(f"** Observations: {self.n_spec} spectra, {self.n_nights} nights", flush=True)
        print(f"** Image Order: {self.order_num}", flush=True)
        print(f"** Tag: {self.tag}", flush=True)
        print(f"** Iterations: {self.n_iterations}", flush=True)
        print(f"** N Cores: {self.n_cores}", flush=True)
        print("***************************************", flush=True)
        
    ##################
    #### OPTIMIZE ####
    ##################
    
    def compute_rvs_for_target(self):
        """The main function to run for a given target to compute the RVs for iterative spectral rv problems.
        """

        # Start the main clock!
        stopwatch = pcutils.StopWatch()
        stopwatch.lap(name='ti_main')
                
        # Iterate over remaining stellar template generations
        for iter_index in range(self.n_iterations):
            
            if iter_index == 0 and hasattr(self.spectral_model, "star") and self.spectral_model.star is not None and self.spectral_model.star.from_flat:
                
                print(f"Starting Iteration {iter_index + 1} of {self.n_iterations} (flat template, no RVs) ...", flush=True)
                stopwatch.lap(name='ti_iter')
                
                # Fit all observations
                self.optimize_all_observations(0)
                
                print(f"Finished Iteration {iter_index + 1} in {round(stopwatch.time_since(name='ti_iter')/3600, 2)} hours", flush=True)
                
                # Augment the template
                if iter_index < self.n_iterations - 1:
                    self.augment_templates(iter_index)
            
            else:
                
                print(f"Starting Iteration {iter_index + 1} of {self.n_iterations} ...", flush=True)
                stopwatch.lap(name='ti_iter')

                # Run the fit for all spectra and do a cross correlation analysis as well.
                self.optimize_all_observations(iter_index)
            
                # Run the ccf for all spectra
                self.cross_correlate_spectra(iter_index)
            
                # Generate the rvs for each observation
                self.gen_nightly_rvs(iter_index)
            
                # Plot the rvs
                self.plot_rvs(iter_index)
            
                # Save the rvs each iteration
                self.save_rvs()
                
                print(f"Finished Iteration {iter_index + 1} in {round(stopwatch.time_since(name='ti_iter')/3600, 2)} hours", flush=True)

                # Print RV Diagnostics
                if self.n_spec >= 1:
                    rvs_std = np.nanstd(self.rvs_dict['rvsfwm'][:, iter_index])
                    print(f"  Stddev of all fwm RVs: {round(rvs_std, 4)} m/s", flush=True)
                    rvs_std = np.nanstd(self.rvs_dict['rvsxc'][:, iter_index])
                    print(f"  Stddev of all xc RVs: {round(rvs_std, 4)} m/s", flush=True)
                if self.n_nights > 1:
                    rvs_std = np.nanstd(self.rvs_dict['rvsfwm_nightly'][:, iter_index])
                    print(f"  Stddev of all fwm nightly RVs: {round(rvs_std, 4)} m/s", flush=True)
                    rvs_std = np.nanstd(self.rvs_dict['rvsxc_nightly'][:, iter_index])
                    print(f"  Stddev of all xc nightly RVs: {round(rvs_std, 4)} m/s", flush=True)
                    
                # Augment the template
                if iter_index < self.n_iterations - 1:
                    self.augment_templates(iter_index)
                

        # Save forward model outputs
        print("Saving results ... ", flush=True)
        self.save_to_pickle()
        
        # End the clock!
        print(f"Completed order {self.order_num} Runtime: {round(stopwatch.time_since(name='ti_main') / 3600, 2)} hours", flush=True)
        
    def optimize_all_observations(self, iter_index):
            
        # Timer
        stopwatch = pcutils.StopWatch()

        # Parallel fitting
        if self.n_cores > 1:
            
            if iter_index == 0:
                p0s = [self.p0] * self.n_spec
            else:
                p0s = []
                for ispec in range(self.n_spec):
                    p0s.append(self.opt_results[ispec, iter_index - 1]["pbest"])
            
            # Call the parallel job via joblib.
            self.opt_results[:, iter_index] = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(self.optimize_and_plot_observation)(p0s[ispec], self.data[ispec], self.spectral_model, self.obj, self.optimizer, iter_index, self.output_path, self.tag, self.target_dict["name"], self.verbose) for ispec in range(self.n_spec))

        else:

            # Fit one observation at a time
            for ispec in range(self.n_spec):
                
                # Get the initial parameters for this spectrum and chunk, further modified later on before optimizing
                if iter_index == 0:
                    p0 = self.p0
                else:
                    p0 = self.opt_results[ispec, iter_index - 1]["pbest"]

                # Optimize and plot all chunks, store results
                self.opt_results[ispec, iter_index] = self.optimize_and_plot_observation(p0, self.data[ispec], self.spectral_model,
                                                                                         self.obj, self.optimizer, iter_index,
                                                                                         self.output_path,
                                                                                         self.tag, self.target_dict["name"], self.verbose)
        
        # Store rvs
        rvsfwm = np.full(self.n_spec, np.nan)
        for ispec in range(self.n_spec):
            pbest = self.opt_results[ispec, iter_index]["pbest"]
            true_star_vel_tdb = pbest[self.spectral_model.star.par_names[0]].value + self.data[ispec].bc_vel
            rvsfwm[ispec] = true_star_vel_tdb
        self.rvs_dict["rvsfwm"][:, iter_index] = rvsfwm
        
        # Print finished
        print(f"Fitting Finished in {round((stopwatch.time_since())/60, 3)} min ", flush=True)
            
    @staticmethod
    def optimize_and_plot_observation(p0, data, spectral_model, obj, optimizer, iter_index, output_path, tag, star_name, verbose):
        
        if data.is_good:
            
            # Time the fit
            stopwatch = pcutils.StopWatch()
        
            # Sanity lock parameters
            p0.sanity_lock()
        
            # Initialize model
            spectral_model.initialize(p0, data, iter_index)
        
            # Store the objective with the model
            spectral_model.obj = obj
        
            # Initialize objective
            obj.initialize(spectral_model)
            
            # Initialize optimizer
            optimizer.initialize(obj)
            
            # Fit the observation
            opt_result = optimizer.optimize()
            
            # Print diagnostics
            print(f"Fit spectrum {data.spec_num} in {round((stopwatch.time_since())/60, 2)} min", flush=True)
            if verbose:
                print(f" RMS = {round(opt_result['fbest'], 3)}", flush=True)
                print(f" Best Fit Parameters:\n{spectral_model.summary(opt_result['pbest'])}", flush=True)

            # Plot
            IterativeSpectralRVProb.plot_spectral_model(opt_result["pbest"], data, spectral_model, iter_index, output_path, tag, star_name)
            
        else:
            opt_result = dict(pbest=p0.gen_nan_pars(), fbest=np.nan, fcalls=np.nan)
        
        # Return result
        return opt_result
        
    ###############
    #### PLOTS ####
    ###############
    
    @staticmethod
    def plot_spectral_model(pars, data, spectral_model, iter_index, output_path, tag, star_name):
        
        # Figure dims for 1 chunk
        fig_width, fig_height = 2000, 720
        dpi = 200
        figsize = int(fig_width / dpi), int(fig_height / dpi)
        
        # Create subplot
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Build the model
        wave_data, model_lr = spectral_model.build(pars)
        wave_data_nm = wave_data / 10
        
        # The residuals for this iteration
        residuals = spectral_model.data.flux  - model_lr

        # Get the mask
        mask = np.copy(spectral_model.data.mask)
        
        # Ensure known bad pixels are nans in the residuals
        residuals[mask == 0] = np.nan
        
        # Change edges to nans
        if spectral_model.obj.remove_edges > 0:
            good = np.where(mask == 1)[0]
            mask[good[0:spectral_model.obj.remove_edges]] = 0
            mask[good[-spectral_model.obj.remove_edges:]] = 0
            residuals[mask == 0] = np.nan
            
        # Now check which bad pixels were also flagged
        if spectral_model.obj.flag_n_worst_pixels > 1:
            ss = np.argsort(np.abs(residuals))
            k = np.max(np.where(np.isfinite(residuals[ss]))[0])
            flagged_inds = ss[k-1*spectral_model.obj.flag_n_worst_pixels - 1:k]
    
        # Left and right padding
        good = np.where(mask == 1)[0]
        pad = 0.01 * (wave_data_nm[good][-1] - wave_data_nm[good][0])
        
        # Data
        plt.plot(wave_data_nm, spectral_model.data.flux, color=(0, 114/255, 189/255), lw=0.8, label="Data")
        
        # Model
        plt.plot(wave_data_nm, model_lr, color=(217/255, 83/255, 25/255), lw=0.8, label="Model")
        
        # Zero line and -0.1 line
        plt.plot(wave_data_nm, np.zeros_like(wave_data_nm), color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':')
        plt.plot(wave_data_nm, np.zeros_like(wave_data_nm) - 0.2, color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':')
        
        # Residuals and worst pixels which were flagged
        plt.plot(wave_data_nm, residuals, color=(255/255, 169/255, 22/255), lw=0.8, label="Residuals")
        plt.plot(wave_data_nm[flagged_inds], residuals[flagged_inds], color="maroon", alpha=0.8, marker='X', markersize=4, lw=0)
        
        # LSF
        lsf = spectral_model.lsf.build(pars=pars)
        
        # Star
        if spectral_model.star is not None:
            
            # Initial star
            if not spectral_model.star.from_flat and iter_index != 0:
                star_wave = spectral_model.star.initial_template[:, 0]
                star_flux = spectral_model.star.initial_template[:, 1]
                star_flux = pcmath.doppler_shift(star_wave, pars[spectral_model.star.par_names[0]].value, wave_out=spectral_model.model_wave, flux=star_flux, interp="cspline", kind="exp")
                star_flux = spectral_model.lsf.convolve_flux(star_flux, lsf=lsf)
                star_flux = pcmath.cspline_interp(spectral_model.model_wave, star_flux, wave_data)
                plt.plot(wave_data_nm, star_flux - 1.2, label='Initial Star', lw=0.8, color='aqua', alpha=0.5)
            
            # Current star
            star_flux = spectral_model.star.build(pars, spectral_model.templates_dict['star'], spectral_model.model_wave)
            star_flux = spectral_model.lsf.convolve_flux(star_flux, lsf=lsf)
            star_flux = pcmath.cspline_interp(spectral_model.model_wave, star_flux, wave_data)
            plt.plot(wave_data_nm, star_flux - 1.2, label='Current Star', lw=0.8, color='deeppink', alpha=0.8)
        
        # Tellurics
        if spectral_model.tellurics is not None:
            tell_flux = spectral_model.tellurics.build(pars, spectral_model.templates_dict['tellurics'], spectral_model.model_wave)
            tell_flux = spectral_model.lsf.convolve_flux(tell_flux, lsf=lsf)
            tell_flux = pcmath.cspline_interp(spectral_model.model_wave, tell_flux, wave_data)
            plt.plot(wave_data_nm, tell_flux - 1.2, label='Tellurics', lw=0.8, color='indigo', alpha=0.2)
        
        # Gas Cell
        if spectral_model.gas_cell is not None:
            gas_flux = spectral_model.gas_cell.build(pars, spectral_model.templates_dict['gas_cell'], spectral_model.model_wave)
            gas_flux = spectral_model.lsf.convolve_flux(gas_flux, lsf=lsf)
            gas_flux = pcmath.cspline_interp(spectral_model.model_wave, gas_flux, wave_data)
            plt.plot(wave_data_nm, gas_flux - 1.2, label='Gas Cell', lw=0.8, color='green', alpha=0.2)
        
        # X and Y limits
        plt.xlim(spectral_model.sregion.wavemin / 10 - pad, spectral_model.sregion.wavemax / 10 + pad)
        plt.ylim(-1.2, 1.1)
            
        # The legend for each chunk
        plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1.0, 0.5))
            
        # X and Y tick parameters
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize=10)
        
        # X and Y axis labels
        plt.xlabel("Wavelength [nm]", fontsize=10)
        plt.ylabel("Norm. flux", fontsize=10)
        
        # The title of each chunk
        plt.title(f"{star_name.replace('_', ' ')}, Order {data.order_num}, Iteration {iter_index + 1}", fontsize=10)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        fname = f"{output_path}Order{data.order_num}{os.sep}ForwardModels{os.sep}{tag}_data_model_spec{data.spec_num}_ord{data.order_num}_iter{iter_index + 1}.png"
        fig.savefig(fname)
        plt.close()
            
        return fig
    
    ###########################
    #### Radial Velocities ####
    ###########################
    
    def cross_correlate_spectra(self, iter_index):
        """Cross correlation wrapper for all spectra.
        
        Args:
            iter_index (int or None): The iteration to use.
        """
        
        stopwatch = pcutils.StopWatch()
        
        print("Cross Correlating Spectra ... ", flush=True)

        # Perform xcorr in series or parallel
        if self.n_cores > 1:
            
            p0s = []
            for ispec in range(self.n_spec):
                p0s.append(self.opt_results[ispec, iter_index]["pbest"])

            # Run in parallel
            ccf_results = Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(self.cross_correlate_observation)(p0s[ispec], self.data[ispec], self.spectral_model, iter_index) for ispec in range(self.n_spec))
            
            for ispec in range(self.n_spec):
                if np.isfinite(ccf_results[ispec][0]):
                    self.rvs_dict['rvsxc'][ispec, iter_index] = ccf_results[ispec][0]
                    self.rvs_dict['uncxc'][ispec, iter_index] = ccf_results[ispec][1]
                    self.rvs_dict['bis'][ispec, iter_index] = ccf_results[ispec][2]
                    self.rvs_dict['xcorrs'][ispec, iter_index] = np.array([ccf_results[ispec][3], ccf_results[ispec][4]]).T
                else:
                    self.data[ispec].is_good = False
            
        else:
            
            # Run in series
            for ispec in range(self.n_spec):
                
                p0 = self.opt_results[ispec, iter_index]["pbest"]
                    
                ccf_results = self.cross_correlate_observation(p0, self.data[ispec], self.spectral_model, iter_index)
                if np.isfinite(ccf_results[0]):
                    self.rvs_dict['rvsxc'][ispec, iter_index] = ccf_results[0]
                    self.rvs_dict['uncxc'][ispec, iter_index] = ccf_results[1]
                    self.rvs_dict['bis'][ispec, iter_index] = ccf_results[2]
                    self.rvs_dict['xcorrs'][ispec, iter_index] = np.array([ccf_results[3], ccf_results[4]]).T
                else:
                    self.data[ispec].is_good = False
                
        print('Cross Correlation Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)
    
    def gen_nightly_rvs(self, iter_index):
        
        # The RMS from the forward model fit
        fit_metrics = np.full(self.n_spec, np.nan)
        for ispec in range(self.n_spec):
            fit_metrics[ispec] = self.opt_results[ispec, iter_index]['fbest']
                
        # Weights are the inverse of the fwm fit
        weights = 1 / fit_metrics**2
        
        # The FwM RVs
        rvsfwm_nightly, uncfwm_nightly = pcrvcalc.compute_nightly_rvs_single_order(self.rvs_dict["rvsfwm"][:, iter_index], weights, self.rvs_dict['n_obs_nights'])
        self.rvs_dict['rvsfwm_nightly'][:, iter_index] = rvsfwm_nightly
        self.rvs_dict['uncfwm_nightly'][:, iter_index] = uncfwm_nightly
        
        # The XC RVs
        rvsxc_nightly, uncxc_nightly = pcrvcalc.compute_nightly_rvs_single_order(self.rvs_dict['rvsxc'][:, iter_index], weights, self.rvs_dict['n_obs_nights'])
        self.rvs_dict['rvsxc_nightly'][:, iter_index] = rvsxc_nightly
        self.rvs_dict['uncxc_nightly'][:, iter_index] = uncxc_nightly
        
        # The XC BIS
        bis_nightly, uncbis_nightly = pcrvcalc.compute_nightly_rvs_single_order(self.rvs_dict['bis'][:, iter_index], weights, self.rvs_dict['n_obs_nights'])
        self.rvs_dict['bis_nightly'][:, iter_index] = bis_nightly
        self.rvs_dict['uncbis_nightly'][:, iter_index] = uncbis_nightly

    def plot_rvs(self, iter_index, time_offset=2450000):
        """Plots all RVs and cross-correlation analysis after forward modeling all spectra.
        """
        
        # Plot the rvs, nightly rvs, xcorr rvs, xcorr nightly rvs
        plot_width, plot_height = 1800, 600
        dpi = 200
        plt.figure(num=1, figsize=(int(plot_width / dpi), int(plot_height / dpi)), dpi=200)
        
        # Aliases
        rvs_dict = self.rvs_dict
        bjds = rvs_dict["bjds"]
        bjdsn = rvs_dict["bjds_nightly"]
        
        # Individual Forward Model
        plt.plot(bjds - time_offset,
                    rvs_dict['rvsfwm'][:, iter_index] - np.nanmedian(rvs_dict['rvsfwm'][:, iter_index]),
                    marker='.', linewidth=0, alpha=0.7, color=(0.1, 0.8, 0.1), label="FwM [indiv]")

        # Individual XC
        plt.plot(bjds - time_offset,
                    rvs_dict['rvsxc'][:, iter_index] - np.nanmedian(rvs_dict['rvsxc'][:, iter_index]),
                    marker='.', linewidth=0, color='black', alpha=0.6, label="XC [indiv]")
        
        
        # Nightly Forward Model
        plt.errorbar(bjdsn - time_offset,
                     rvs_dict['rvsfwm_nightly'][:, iter_index] - np.nanmedian(rvs_dict['rvsfwm_nightly'][:, iter_index]),
                     yerr=rvs_dict['uncfwm_nightly'][:, iter_index],
                     marker='o', linewidth=0, elinewidth=1, label='FwM [nightly]', color=(0, 114/255, 189/255))
        
        # Nightly XC
        plt.errorbar(bjdsn - time_offset,
                     rvs_dict['rvsxc_nightly'][:, iter_index] - np.nanmedian(rvs_dict['rvsxc_nightly'][:, iter_index]),
                     yerr=rvs_dict['uncxc_nightly'][:, iter_index],
                     marker='X', linewidth=0, alpha=0.8, label='XC [nightly]', color='darkorange', elinewidth=1)
        
        # Plot labels
        plt.title(f"{self.target_dict['name'].replace('_', ' ')}, Order {self.order_num}, Iteration {iter_index + 1}")
        ax = plt.gca()
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.xlabel(f"BJD - {time_offset}")
        plt.ylabel('RV [m/s]')
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        fname = f"{self.output_path}Order{self.order_num}{os.sep}RVs{os.sep}{self.tag}_rvs_ord{self.order_num}_iter{iter_index + 1}.png"
        plt.savefig(fname)
        plt.close()
        
        # Plot the BIS vs. XC RV
        plt.figure(1, figsize=(12, 7), dpi=200)
        
        plt.errorbar(rvs_dict['rvsxc_nightly'][:, iter_index], rvs_dict['bis_nightly'][:, iter_index],
                     xerr=rvs_dict['uncxc_nightly'][:, iter_index], yerr=rvs_dict['uncbis_nightly'][:, iter_index],
                     marker='o', lw=0, elinewidth=1)
        
        # Annotate
        plt.title(f"{self.target_dict['name'].replace('_', ' ')} XC BIS Correlation, Order {self.order_num}, Iteration {iter_index + 1}")
        plt.xlabel('XC RV [m/s]')
        plt.ylabel('BIS [m/s]')
        plt.tight_layout()
        fname = f"{self.output_path}Order{self.order_num}{os.sep}RVs{os.sep}{self.tag}_BIS_ord{self.order_num}_iter{iter_index + 1}.png"
        plt.savefig(fname)
        plt.close()

    @staticmethod
    def cross_correlate_observation(p0, data, spectral_model, iter_index):
        
        if data.is_good:
        
            # Initialize
            spectral_model.initialize(p0, data, iter_index)
        
            # Run the CCF
            ccf_result = pcrvcalc.brute_force_ccf(p0, spectral_model, iter_index)
            
        else:
            
            ccf_result = np.nan, np.nan, np.nan, np.full(400, np.nan), np.full(400, np.nan)
            
        return ccf_result
    

    ###############################
    #### TEMPLATE AUGMENTATION ####
    ###############################
    
    def augment_templates(self, iter_index):
        
        # Augment the templates
        self.augmenter.augment_templates(self, iter_index)
        
        # Stellar template
        self.stellar_templates[iter_index + 1] = np.copy(self.spectral_model.templates_dict["star"])
    

    ###############
    #### MISC. ####
    ###############
    
    def create_output_paths(self):
        o_folder = f"Order{self.order_num}{os.sep}"
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_path + o_folder, exist_ok=True)
        os.makedirs(self.output_path + o_folder + "ForwardModels", exist_ok=True)
        os.makedirs(self.output_path + o_folder + "RVs", exist_ok=True)
        os.makedirs(self.output_path + o_folder + "Templates", exist_ok=True)
    
    @property
    def spec_module(self):
        return importlib.import_module(f"pychell.data.{self.spectrograph.lower()}")
    
    @property
    def n_spec(self):
        return len(self.data)

    @property
    def n_nights(self):
        return len(self.bjdsn)

    @property
    def bjds(self):
        return self.rvs_dict["bjds"]
    
    @property
    def bjdsn(self):
        return self.rvs_dict["bjds_nightly"]

    @property
    def p0(self):
        return self.spectral_model.p0
    
    @property
    def n_iterations(self):
        return self.spectral_model.n_iterations
    
    @property
    def model_resolution(self):
        return self.spectral_model.model_resolution
    
    @property
    def crop_pix(self):
        return self.spectral_model.crop_pix
    
    @property
    def order_num(self):
        return self.spectral_model.order_num
        
    def __repr__(self):
        s = repr(self.spectral_model)
        s += f"{repr(self.optimizer)}\n"
        s += f"{repr(self.obj)}\n"
        return s
        

    ##############
    #### SAVE ####
    ##############
    
    def save_rvs(self):
        """Saves the RVs to an npz file since each value in the dictionary is a numpy array.
        """
        
        # Filename
        fname = f"{self.output_path}Order{self.order_num}{os.sep}RVs{os.sep}{self.tag}_rvs_ord{self.order_num}.npz"
        
        # Save in a .npz file for easy access later
        np.savez(fname, **self.rvs_dict)
    
    def save_to_pickle(self):
        fname = f"{self.output_path}Order{self.order_num}{os.sep}{self.tag}_spectralrvprob_ord{self.order_num}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
    