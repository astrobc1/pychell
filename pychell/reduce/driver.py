# Default Python modules
import os
import glob
import importlib
import sys
import pdb
import pickle
import copy
import time
import json
import warnings

# Science/math
import numpy as np
np.seterr(invalid='ignore', divide='ignore')
warnings.filterwarnings('ignore')
from astropy.io import fits
from astropy.time import Time

# Graphics
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Parallelization
from joblib import Parallel, delayed

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
import pychell.data as pcdata
import pychell.data.parsers as pcparsers
import pychell.reduce.extract as pcextract
import pychell.reduce.calib as pccalib
import pychell.reduce.order_map as pcomap
import pychell.config as pcconfig

def reduce_night(user_redux_settings):
    """The main function to reduce a night of data.

    Args:
        user_redux_settings (dict): A dictionary with settings to reduce this night.
    """

    # Start the main clock
    stopwatch = pcutils.StopWatch()

    # Parse the input directories, return data (images not loaded into memory)
    config, data = init_night(user_redux_settings)
    
    #####################
    #### Calibration ####
    #####################
    
    # Create Master Bias (only one)
    if config['bias_subtraction']:
        
        print('Creating Master Bias ...', flush=True)
        
        # Create master bias image and save
        master_bias_image = pccalib.generate_master_bias(data['master_bias'].individuals)
        data['master_bias'].save_master_image(master_bias_image)
    
    # Create Master Darks for each unique exposure time
    if config['dark_subtraction']:
        
        print('Creating Master Dark(s) ...', flush=True)
        
        # Create master dark image and save
        for master_dark in data['master_darks']:
            master_dark_image = pccalib.generate_master_dark(master_dark.individuals)
            master_dark.save(master_dark_image)
            
    # Create Master Flats for each flats group
    if config['flat_division']:
        
        print('Creating Master Flat(s) ...', flush=True)
        
        # Create master flat image and save
        for master_flat in data['master_flats']:
            master_flat_image = pccalib.generate_master_flat(master_flat.individuals, bias_subtraction=config['bias_subtraction'], dark_subtraction=config['dark_subtraction'], norm=config['flatfield_percentile'])
            master_flat.save(master_flat_image)

    #######################
    #### Order Tracing ####
    #######################
    
    # Order images and order dictionaries containing the name, height, and polynomial coeffs for each order
    if config['order_map']['source'] == 'empirical_unique':
        print('Empirically Deriving Trace For Each Image ...', flush=True)
        for sci in data['science']:
            sci.order_map.trace_orders()
        
    elif config['order_map']['source'] == 'empirical_from_flat_fields':
        print('Tracing Orders From Master Flat(s) ...', flush=True)
        for order_map in data['order_maps']:
            order_map.trace_orders(config)
        
    elif config['order_map']['source'] == 'hard_coded':
        
        data['order_maps'] = []
        
        map_init = getattr(pcdata, config['order_map']['class'])
        
        print('Using Hard Coded Order Map from Class ' + config['order_map']['class'], flush=True)
        
        data['order_maps'].append(map_init(data['science'][0]))
        
        data['science'][0].order_map = data['order_maps'][0]
            
        for d in data['science']:
            d.order_map = data['science'][0].order_map
    else:
        raise ValueError("Order Map options not recognized")
    
    # Extraction of science spectra
    if config['n_cores'] > 1:
        
        print('Extracting Spectra In Parallel Using ' + str(config['n_cores']) + ' Cores ...', flush=True)
        
        # Run in Parallel
        iter_pass = [(data['science'], i, config) for i in range(len(data['science']))]
        
        # Call in parallel
        Parallel(n_jobs=config['n_cores'], verbose=0, batch_size=1)(delayed(pcextract.extract_full_image_wrapper)(*iter_pass[i]) for i in range(len(data['science'])))
        
    else:
        # One at a time
        print('Extracting Spectra ...', flush=True)
        for i in range(len(data['science'])):
            pcextract.extract_full_image_wrapper(data['science'], i, config)

    print('TOTAL RUN TIME: ' + str(round((stopwatch.time_since()) / 3600, 3)) + ' Hours')
    print('ALL FINISHED !')
    
    
def init_night(user_redux_settings):
    
    # Start with the default config
    config = {}
    config.update(pcconfig.general_settings)
    config.update(pcconfig.redux_settings)
    
    # Update with the default instrument dictionaries
    spec_module = importlib.import_module('pychell.data.' + user_redux_settings['spectrograph'].lower())
    config.update(spec_module.redux_settings)
    
    # User settings
    config.update(user_redux_settings)

    # Get the base input directory. The reduction (output) folder will have this same name.
    base_input_path = os.path.basename(os.path.normpath(config['data_input_path']))
    config['run_output_path'] = config['output_path_root'] + base_input_path + os.sep
    
    # Make the output directories
    create_output_dirs(config)
    
    # Identify what's what.
    print('Analyzing the input files ...')
    if hasattr(pcparsers, config['spectrograph'] + 'Parser'):
        data_parser_class= getattr(pcparsers, config['spectrograph'] + 'Parser')
        data = data_parser_class(config).categorize_raw_data(config)
    else:
        raise NotImplementedError("Must use a supported instrument for now, or implement a new instrument.")
            
    return config, data
    
    
def create_output_dirs(config):
    
    # Make the root output directory for this run
    os.makedirs(config['run_output_path'], exist_ok=True)

    # Trace information (profiles, refined y positions, order maps)
    os.makedirs(config['run_output_path'] + 'trace', exist_ok=True)

    # 1-dimensional spectra in fits files and 
    os.makedirs(config['run_output_path'] + 'spectra', exist_ok=True)

    # Calibration (master bias, darks, flats, tellurics, wavecal)
    os.makedirs(config['run_output_path'] + 'calib', exist_ok=True)