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
        
        # Get better estimations for some parameters (eg xcorr for star)
        forward_models.opt_init_params()
        
        # Stores the stellar templates over iterations. The plus 1 is for the wave grid
        stellar_templates = np.empty(shape=(forward_models[0].n_model_pix, forward_models.n_template_fits + 1), dtype=np.float64)

        # Zeroth Iteration - No doppler shift if using a flat template.
        # We could also flag the stellar lines, but this has minimal impact on the RVs
        if not forward_models[0].models_dict['star'].from_synthetic:

            print('Iteration: 0 (flat stellar template, no RVs)', flush=True)

            forward_models.fit_spectra(0)

            if forward_model_settings['n_template_fits'] == 0:
                forward_models.save_final_outputs(forward_model_settings)
                continue
            else:
                forward_models.template_augmenter(forward_models, iter_num=0, nights_for_template=forward_model_settings['nights_for_template'])
                
                stellar_templates[:, 1] = np.copy(forward_models.templates_dict['star'][:, 1])
                forward_models.update_models(0)

        # Iterate over remaining stellar template generations
        for iter_num in range(forward_model_settings['n_template_fits']):

            print('Starting Iteration: ' + str(iter_num+1) + ' of ' + str(forward_models.n_template_fits), flush=True)
            stopwatch.lap(name='ti_iter')

            # Run the fit for all spectra and do a cross correlation analysis as well.
            forward_models.fit_spectra(iter_num)
            
            print('Finished Iteration ' + str(iter_num+1) + ' in ' + str(round(stopwatch.time_since(name='ti_iter')/3600, 2)) + ' hours', flush=True)
            
            # Compute the RVs and output after each iteration (same file is overwritten)
            pcrvcalc.generate_rvs(forward_models, iter_num)
            pcrvcalc.plot_rvs(forward_models, iter_num)
            forward_models.save_rvs()

            # Print RV Diagnostics
            if forward_models.n_nights > 1:
                rvscd_std = np.nanstd(forward_models.rvs_dict['rvs_nightly'][:, iter_num])
                print('  Stddev of all nightly RVs: ' + str(round(rvscd_std, 4)) + ' m/s', flush=True)
            elif forward_models.n_spec >= 1:
                rvs_std = np.nanstd(forward_models.rvs_dict['rvs'][:, iter_num])
                print('  Stddev of all RVs: ' + str(round(rvs_std, 4)) + ' m/s', flush=True)

            # Compute the new stellar template, update parameters.
            if iter_num + 1 < forward_models.n_template_fits:
                
                # Template Augmentation
                forward_models.template_augmenter(forward_models, iter_num=iter_num, nights_for_template=forward_models.nights_for_template)

                # Update the forward model initial_parameters.
                forward_models.update_models(iter_num)
                
                stellar_templates[:, iter_num+1] = np.copy(forward_models.templates_dict['star'][:, 1])
                

        # Save forward model outputs
        print('Saving Final Outputs ... ', flush=True)
        forward_models.save_results()

        # Save Stellar Template Outputs
        np.savez(forward_models.run_output_path_stellar_templates + os.sep + forward_models.tag + '_stellar_templates_ord' + str(order_num+1) + '.npz', stellar_templates=stellar_templates)

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
    forward_model_settings.update(spec_module.general_settings)
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


################################################################################################################
