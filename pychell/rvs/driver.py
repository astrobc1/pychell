# Python built in modules
import copy
import os
import importlib.util
import warnings
from sys import platform # plotting backend

# Graphics
import matplotlib
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Science/math
import numpy as np # Math, Arrays
np.seterr(divide='ignore')
warnings.filterwarnings('ignore')

# pychell
import pychell.config as pcconfig
import pychell.rvs.forward_models as pcforwardmodels
import pychell.data.parsers as pcparsers
import pychell.utils as pcutils

# Main function
def compute_rvs(user_forward_model_settings, user_model_blueprints):
    """The main function to run for a given target to compute the RVs.

    Args:
        user_forward_model_settings (dict): A dictionary containing the user settings.
        user_model_blueprints (dict): A dictionary containing the user blueprints.
    """

    # Start the main clock!
    stopwatch = pcutils.StopWatch()
    stopwatch.lap(name='ti_main')

    # Set things up and create a dictionary forward_model_settings used throughout the code
    config, model_blueprints = init(user_forward_model_settings, user_model_blueprints)
    
    # The data parser
    if hasattr(pcparsers, config['spectrograph'] + 'Parser'):
        data_parser_class = getattr(pcparsers, config['spectrograph'] + 'Parser')
        parser = data_parser_class(config)
    else:
        raise NotImplementedError("Must use a supported instrument for now, or implement a new instrument.")
    
    # Main loop over orders
    # order_num = 0, ..., n_orders-1
    for order_num in config['do_orders']:
        
        # Construct the forward models object
        # This will construct the individual forward model objects (single spectrum)
        forward_models = pcforwardmodels.ForwardModels(config, model_blueprints, parser=parser, order_num=order_num) # basically a fancy list
        
        # Stores the stellar templates over iterations.
        stellar_templates = []
        
        # Zeroth Iteration - No doppler shift if using a flat template.
        # We could also flag the stellar lines, but this has minimal impact on the RVs
        if not forward_models[0].models_dict['star'].from_synthetic:
            
            print('Iteration: 0 (flat stellar template, no RVs)', flush=True)

            # Check if any parameters are enabled
            if len(forward_models[0].initial_parameters.get_varied()) == 0:
                print('No parameters to optimize, moving on', flush=True)
                for fwm in forward_models:
                    fwm.opt_results.append([])
                    for _ in range(forward_models.n_chunks):
                        fwm.opt_results[-1].append({'xbest': fwm.initial_parameters, 'fbest': np.nan, 'fcalls': np.nan})
                forward_models.template_augmenter(forward_models, iter_index=0)
                forward_models.update_models(0)
            else:
                
                forward_models.fit_spectra(0)

                if forward_models.n_template_fits == 0:
                    forward_models.save_results()
                    continue
                else:
                    forward_models.template_augmenter(forward_models, iter_index=0)
                    forward_models.update_models(0)
                
        stellar_templates.append(np.copy(forward_models.templates_dict['star'][:, 1]))

        # Iterate over remaining stellar template generations
        for iter_index in range(config['n_template_fits']):

            print('Starting Iteration: ' + str(iter_index+1) + ' of ' + str(forward_models.n_template_fits), flush=True)
            stopwatch.lap(name='ti_iter')

            # Run the fit for all spectra and do a cross correlation analysis as well.
            forward_models.fit_spectra(iter_index)
            
            print('Finished Iteration ' + str(iter_index + 1) + ' in ' + str(round(stopwatch.time_since(name='ti_iter')/3600, 2)) + ' hours', flush=True)
            
            # Compute the RVs and output after each iteration for diagnostic purposes (same file is overwritten)
            forward_models.compute_nightly_rvs(iter_index)
            forward_models.plot_rvs(iter_index)
            forward_models.save_rvs()

            # Print RV Diagnostics
            if forward_models.n_nights > 1:
                rvscd_std = np.nanstd(forward_models.rvs_dict['rvsfwm_nightly'][:, iter_index])
                print('  Stddev of all nightly RVs: ' + str(round(rvscd_std, 4)) + ' m/s', flush=True)
            elif forward_models.n_spec >= 1:
                rvs_std = np.nanstd(forward_models.rvs_dict['rvsfwm'][:, :, iter_index])
                print('  Stddev of all RVs: ' + str(round(rvs_std, 4)) + ' m/s', flush=True)

            # Compute the new stellar template, update parameters.
            if iter_index + 1 < forward_models.n_template_fits:
                
                # Template Augmentation
                forward_models.update_templates(iter_index=iter_index)

                # Update the forward model initial_parameters.
                forward_models.update_models(iter_index)
                
                # Pass to stellar template array
                stellar_templates.append(np.copy(forward_models.templates_dict['star'][:, 1]))
                

        # Save forward model outputs
        print('Saving Final Outputs ... ', flush=True)
        forward_models.save_results()

        # Save Stellar Template Outputs
        np.savez(forward_models.run_output_path + forward_models.o_folder + 'Templates' + os.sep + forward_models.tag + '_stellar_templates_ord' + str(order_num) + '.npz', stellar_templates=stellar_templates)

    # End the clock!
    print('ALL DONE! Runtime: ' + str(round(stopwatch.time_since(name='ti_main') / 3600, 2)) + ' hours', flush=True)

# Init methods below to merge user passed settings with default settings.
def init(user_forward_model_settings, user_model_blueprints):

    # Dictionaries
    config = {}
    model_blueprints = {}

    # Pipeline Defaults
    init_config(config, user_forward_model_settings)
    
    # The model blueprints
    init_blueprints(config, model_blueprints, user_model_blueprints=user_model_blueprints)
    
    # Download templates if need be
    init_templates(config)
    
    # Matplotlib backend
    if config['n_cores'] > 1 or platform != 'darwin':
        matplotlib.use("AGG")
    else:
        matplotlib.use("MacOSX")
    plt.ioff()

    return config, model_blueprints
        
def init_config(config, user_forward_model_settings):
    
    # Load the config file and add to dict.
    config.update(pcconfig.general_settings)
    config.update(pcconfig.forward_model_settings)
    
    # Ensure the number of echelle orders is 1-d
    user_forward_model_settings['do_orders'] = np.atleast_1d(user_forward_model_settings['do_orders'])
    
    # Load in the default instrument settings and add to dict.
    spec_module = importlib.import_module('pychell.data.' + user_forward_model_settings["spectrograph"].lower())
    config.update(spec_module.forward_model_settings)
    
    # Add user settings to dictionary
    config.update(user_forward_model_settings)
    
def init_blueprints(config, model_blueprints, user_model_blueprints=None):
    
    # Default no user blueprints, use defaults
    if user_model_blueprints is None:
        user_model_blueprints = {}
    
    spec_mod = importlib.import_module('pychell.data.' + config["spectrograph"].lower())
    model_blueprints.update(spec_mod.forward_model_blueprints)

    for user_key in user_model_blueprints:
        if user_key in model_blueprints: # Key is common, update sub keys only
            model_blueprints[user_key].update(user_model_blueprints[user_key])
        else: # Key is new, just add
            model_blueprints[user_key] = user_model_blueprints[user_key]

def create_output_dirs(config):
    
    # Output path for this run
    config["run_output_path"] = config["output_path"] + config["spectrograph"].lower() + "_" + config["tag"] + os.sep
    
    # Create the output dir for this run
    os.makedirs(config["run_output_path"], exist_ok=True)
    
    for order_num in config["do_orders"]:
        o_folder = 'Order' + str(order_num) + os.sep
        os.makedirs(config["run_output_path"] + o_folder + 'RVs', exist_ok=True)
        os.makedirs(config["run_output_path"] + o_folder + 'ForwardModels', exist_ok=True)
        os.makedirs(config["run_output_path"] + o_folder + 'Templates', exist_ok=True)

def init_templates(config):
    
    if config['force_download_templates']:
        pcutils.download_templates(config['templates_path'])
