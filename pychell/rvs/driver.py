# Python built in modules
import copy
import glob # File searching
import os # Making directories
import importlib.util # importing other modules from files
import warnings # ignore warnings
import sys # sys utils
import pickle
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
# Numpy warnings
np.seterr(divide='ignore')
warnings.filterwarnings('ignore')
import scipy.interpolate # Cubic interpolation, Akima interpolation

# llvm
from numba import njit, jit, prange

# Pychell modules
import pychell.config as pcconfig
import pychell.rvs.template_augmenter as pcaugmenter
import pychell.maths as pcmath # mathy equations
import pychell.rvs.forward_models as pcforwardmodels # the various forward model implementations
import pychell.rvs.data1d as pcdata # the data objects
import pychell.rvs.model_components as pcmodelcomponents # the data objects
import pychell.utils as pcutils # random helpful functions
import pychell.rvs.rvcalc as pcrvcalc

# Main function
def fit_target(user_forward_model_settings, user_model_blueprints):

    # Start the main clock!
    stopwatch = pcutils.StopWatch()
    stopwatch.lap(name='ti_main')

    # Set things up and create a dictionary forward_model_settings used throughout the code
    forward_model_settings, model_blueprints = init(user_forward_model_settings, user_model_blueprints)
    
    # Main loop over orders
    # order_num = 0, ..., n_orders-1
    for order_num in forward_model_settings['do_orders']:
        
        # Construct the forward models object
        # This will construct the individual forward model objects (single spectrum)
        forward_models = pcforwardmodels.ForwardModels(forward_model_settings, model_blueprints, order_num) # basically a fancy list
        
        # Get better estimation for star (eg xcorr for star)
        if forward_models[0].models_dict['star'].from_synthetic:
            forward_models.cross_correlate_spectra()
        
        # Stores the stellar templates over iterations.
        stellar_templates = np.empty(shape=(forward_models[0].n_model_pix, forward_models.n_template_fits + 1), dtype=np.float64)
        stellar_templates[:, 0] = forward_models.templates_dict['star'][:, 0]
        
        # Zeroth Iteration - No doppler shift if using a flat template.
        # We could also flag the stellar lines, but this has minimal impact on the RVs
        if not forward_models[0].models_dict['star'].from_synthetic:
            
            print('Iteration: 0 (flat stellar template, no RVs)', flush=True)

            # Check if any parameters are enabled
            if not np.any([forward_models[0].initial_parameters[pname].vary for pname in forward_models[0].initial_parameters]):
                print('No parameters to optimize, moving on', flush=True)
                for ispec in range(forward_models.n_spec):
                    fwm = forward_models[ispec]
                    start_wave, start_flux = fwm.build_full(fwm.initial_parameters, None)
                    fwm.best_fit_pars.append(fwm.initial_parameters)
                    fwm.wavelength_solutions.append(start_wave)
                    fwm.models.append(start_flux)
                    fwm.opt.append([np.nan, np.nan])
                    fwm.residuals.append(fwm.data.flux - start_flux)
                forward_models.template_augmenter(forward_models, iter_index=0, nights_for_template=forward_models.nights_for_template, templates_to_optimize=forward_models.templates_to_optimize)
                forward_models.update_models(0)
            else:
                
                forward_models.fit_spectra(0)

                if forward_model_settings['n_template_fits'] == 0:
                    forward_models.save_results()
                    continue
                else:
                    forward_models.template_augmenter(forward_models, iter_index=0, nights_for_template=forward_models.nights_for_template, templates_to_optimize=forward_models.templates_to_optimize)
                    forward_models.update_models(0)
                
        stellar_templates[:, 1] = np.copy(forward_models.templates_dict['star'][:, 1])

        # Iterate over remaining stellar template generations
        for iter_index in range(forward_model_settings['n_template_fits']):

            print('Starting Iteration: ' + str(iter_index+1) + ' of ' + str(forward_models.n_template_fits), flush=True)
            stopwatch.lap(name='ti_iter')

            # Run the fit for all spectra and do a cross correlation analysis as well.
            forward_models.fit_spectra(iter_index)
            
            print('Finished Iteration ' + str(iter_index + 1) + ' in ' + str(round(stopwatch.time_since(name='ti_iter')/3600, 2)) + ' hours', flush=True)
            
            # Compute the RVs and output after each iteration for diagnostic purposes (same file is overwritten)
            forward_models.generate_nightly_rvs(iter_index)
            forward_models.plot_rvs(iter_index)
            forward_models.save_rvs()

            # Print RV Diagnostics
            if forward_models.n_nights > 1:
                rvscd_std = np.nanstd(forward_models.rvs_dict['rvs_nightly'][:, iter_index])
                print('  Stddev of all nightly RVs: ' + str(round(rvscd_std, 4)) + ' m/s', flush=True)
            elif forward_models.n_spec >= 1:
                rvs_std = np.nanstd(forward_models.rvs_dict['rvs'][:, iter_index])
                print('  Stddev of all RVs: ' + str(round(rvs_std, 4)) + ' m/s', flush=True)

            # Compute the new stellar template, update parameters.
            if iter_index + 1 < forward_models.n_template_fits:
                
                # Template Augmentation
                if hasattr(forward_models, 'templates_to_optimize') and len(forward_models.templates_to_optimize) > 0:
                    pcaugmenter.global_fit(forward_models, iter_index=iter_index, nights_for_template=forward_models.nights_for_template, templates_to_optimize=forward_models.templates_to_optimize)
                else:
                    forward_models.template_augmenter(forward_models, iter_index=iter_index, nights_for_template=forward_models.nights_for_template, templates_to_optimize=forward_models.templates_to_optimize)

                # Update the forward model initial_parameters.
                forward_models.update_models(iter_index)
                
                # Pass to stellar template array
                stellar_templates[:, iter_index+2] = np.copy(forward_models.templates_dict['star'][:, 1])
                

        # Save forward model outputs
        print('Saving Final Outputs ... ', flush=True)
        forward_models.save_results()

        # Save Stellar Template Outputs
        np.savez(forward_models.run_output_path_stellar_templates + os.sep + forward_models.tag + '_stellar_templates_ord' + str(order_num) + '.npz', stellar_templates=stellar_templates)
        
        if 'lab_coherence' in forward_models.templates_dict:
            np.savez(forward_models.run_output_path_stellar_templates + os.sep + forward_models.tag + '_lab_coherence_ord' + str(order_num) + '.npz', lab_coherence=forward_models.templates_dict['lab_coherence'])

    # End the clock!
    print('ALL DONE! Runtime: ' + str(round(stopwatch.time_since(name='ti_main') / 3600, 2)) + ' hours', flush=True)
                    

# Initialize the pipeline based on input_options file
def init(user_forward_model_settings, user_model_blueprints):

    # Dictionaries to store settings (and later forward model blueprints)
    forward_model_settings = {}

    # Pipeline Defaults
    init_defaults(forward_model_settings)
    
    # Instrument
    init_spectrograph(forward_model_settings, user_forward_model_settings['spectrograph'])
    
    # Init user
    init_user(forward_model_settings, user_forward_model_settings)
    
    # The model blueprints
    model_blueprints = init_blueprints(forward_model_settings, user_model_blueprints=user_model_blueprints)
    
    # Download templates if need be
    init_templates(forward_model_settings)
    
    # Matplotlib backend
    if forward_model_settings['n_cores'] > 1 or platform != 'darwin':
        matplotlib.use("AGG")
    else:
        matplotlib.use("MacOSX")

    return forward_model_settings, model_blueprints


def init_spectrograph(forward_model_settings, spectrograph=None):
    
    if spectrograph is None:
        spectrograph = forward_model_settings['spectrograph']
        
    # Load in the default instrument settings and add to dict.
    spec_module = importlib.import_module('pychell.spectrographs.' + spectrograph.lower())
    forward_model_settings.update(spec_module.forward_model_settings)
        
def init_defaults(forward_model_settings):
    
    # Load the config file and add to dict.
    forward_model_settings.update(pcconfig.general_settings)
    forward_model_settings.update(pcconfig.forward_model_settings)
    
def init_user(forward_model_settings, user_forward_model_settings):
    
    # Add to dictionary
    forward_model_settings.update(user_forward_model_settings)
    
    # Ensure the number of echelle orders is 1-d
    forward_model_settings['do_orders'] = np.atleast_1d(forward_model_settings['do_orders'])
    
def init_blueprints(forward_model_settings, user_model_blueprints=None, spectrograph=None):
    
    if spectrograph is None:
       spectrograph = forward_model_settings['spectrograph']
    
    model_blueprints = {}
    
    spec_mod = importlib.import_module('pychell.spectrographs.' + spectrograph.lower())
    model_blueprints = spec_mod.forward_model_blueprints

    for user_key in user_model_blueprints:
        if user_key in model_blueprints: # Key is common, update sub keys only
            model_blueprints[user_key].update(user_model_blueprints[user_key])
        else: # Key is new, just add
            model_blueprints[user_key] = user_model_blueprints[user_key]
            
    return model_blueprints


def init_templates(forward_model_settings):
    
    if forward_model_settings['force_download_templates']:
        pcutils.download_templates(forward_model_settings['templates_path'])


################################################################################################################
