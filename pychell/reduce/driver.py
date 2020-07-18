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
stop = pdb.set_trace

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
import pychell.reduce.data2d as pcdata
import pychell.reduce.extract as pcextract
import pychell.reduce.calib as pccalib
import pychell.reduce.order_map as pcomap
import pychell.config as pcconfig

def reduce_night(user_redux_settings):

    # Start the main clock
    stopwatch = pcutils.StopWatch()

    # Parse the input directories, return data (images not loaded into memory)
    redux_settings, data = init_night(user_redux_settings)
    
    #####################
    #### Calibration ####
    #####################
    
    # Create Master Bias (only one)
    if redux_settings['bias_subtraction']:
        
        print('Creating Master Bias ...', flush=True)
        
        # Simple median combine, probably
        master_bias_image = pccalib.generate_master_bias(data['master_bias'].individuals)
        
        # Save the master bias image
        data['master_bias'].save_master_image(master_bias_image)
    
    # Create Master Darks for each unique exposure time
    if redux_settings['dark_subtraction']:
        
        print('Creating Master Dark(s) ...', flush=True)
        
        for master_dark in data['master_darks']:
            
            # Simple median combine, probably
            master_dark_image = pccalib.generate_master_dark(master_dark.individuals)
            
            # Save
            master_dark.save(master_dark_image)
            
    # Create Master Flats for each flats group
    if redux_settings['flat_division']:
        
        print('Creating Master Flat(s) ...', flush=True)
        
        for master_flat in data['master_flats']:
            
            # Simple median combine, probably
            master_flat_image = pccalib.generate_master_flat(master_flat.individuals, bias_subtraction=redux_settings['bias_subtraction'], dark_subtraction=redux_settings['dark_subtraction'], flatfield_percentile=redux_settings['flatfield_percentile'])
            
            # Save
            master_flat.save(master_flat_image)

    #######################
    #### Order Tracing ####
    #######################
    
    # Order images and order dictionaries containing the name, height, and polynomial coeffs for each order
    if redux_settings['order_map']['source'] == 'empirical_unique':
        
        print('Empirically Deriving Trace For Each Image ...', flush=True)
        data['order_maps'] = []
        map_init = redux_settings['order_map']['method']
        for sci in data['science']:
            data['order_maps'].append(pcdata.ScienceOrderMap(sci, map_init))
            data['order_maps'][-1].trace_orders(redux_settings)
            sci.order_map = data['order_maps'][-1]
        
    elif redux_settings['order_map']['source'] == 'empirical_from_flat_fields':
        print('Tracing Orders From Master Flat(s) ...', flush=True)
        data['order_maps'] = []
        map_init = redux_settings['order_map']['method']
        for master_flat in data['master_flats']:
            data['order_maps'].append(pcdata.FlatFieldOrderMap(master_flat, map_init))
            data['order_maps'][-1].trace_orders(redux_settings)
            master_flat.order_map = data['order_maps'][-1]
            
            
        # Also give the order map attributes to the science images
        for d in data['science']:
            d.order_map = d.master_flat.order_map
        
    elif redux_settings['order_map']['source'] == 'hard_coded':
        
        data['order_maps'] = []
        
        map_init = getattr(pcdata, redux_settings['order_map']['class'])
        
        print('Using Hard Coded Order Map from Class ' + redux_settings['order_map']['class'], flush=True)
        
        data['order_maps'].append(map_init(data['science'][0]))
        
        data['science'][0].order_map = data['order_maps'][0]
            
        for d in data['science']:
            d.order_map = data['science'][0].order_map
    else:
        raise ValueError("Order Map options not recognized")
        
    # Correct master flats for any artifacts (fringing, remove blaze)
    #if redux_settings['correct_flat_field_artifacts']:
    
    
    #if ('correct_fringing_in_flatfield' in redux_settings and redux_settings['correct_fringing_in_flatfield']) or ('correct_blaze_function_in_flatfield' in redux_settings and redux_settings['correct_blaze_function_in_flatfield']):
        
    #    print('Correcting artifacts in master flat(s) ...', flush=True)
    #    for sci_data in raw_data['science']:
    #        sci_data.correct_flat_artifacts(output_dir=general_settings['output_path_root'], calibration_settings=calib_settings)

    # Extraction of science spectra
    if redux_settings['n_cores'] > 1:
        
        print('Extracting Spectra In Parallel Using ' + str(redux_settings['n_cores']) + ' Cores ...', flush=True)
        
        # Run in Parallel
        iter_pass = [(data['science'], i, redux_settings) for i in range(len(data['science']))]
        
        # Call in parallel
        Parallel(n_jobs=redux_settings['n_cores'], verbose=0, batch_size=1)(delayed(pcextract.extract_full_image_wrapper)(*iter_pass[i]) for i in range(len(data['science'])))
        
    else:
        # One at a time
        print('Extracting Spectra ...', flush=True)
        for i in range(len(data['science'])):
            pcextract.extract_full_image_wrapper(data['science'], i, redux_settings)

    print('TOTAL RUN TIME: ' + str(round((stopwatch.time_since()) / 3600, 3)) + ' Hours')
    print('ALL FINISHED !')
    
    
def init_night(user_redux_settings):
    
    # Start with the default config
    redux_settings = {}
    redux_settings.update(pcconfig.redux_settings)
    redux_settings.update(pcconfig.general_settings)
    
    # Update with the default instrument dictionaries
    spec_module = importlib.import_module('pychell.spectrographs.' + user_redux_settings['spectrograph'].lower() + '.settings')
    redux_settings.update(spec_module.redux_settings)
    
    # User settings
    redux_settings.update(user_redux_settings)

    # Get the base input directory. The reduction (output) folder will have this same name.
    base_input_dir = os.path.basename(os.path.normpath(redux_settings['input_path']))
    redux_settings['run_output_path'] = redux_settings['output_path_root'] + base_input_dir + os.sep
    
    # Make the output directories
    if not os.path.isdir(redux_settings['run_output_path']):

        # Make the root output directory for this run
        os.makedirs(redux_settings['run_output_path'], exist_ok=True)

        # Trace information (profiles, refined y positions, order maps)
        os.makedirs(redux_settings['run_output_path'] + 'trace', exist_ok=True)

        # 1-dimensional spectra in fits files and 
        os.makedirs(redux_settings['run_output_path'] + 'spectra', exist_ok=True)

        # Calibration (master bias, darks, flats, tellurics, wavecal)
        os.makedirs(redux_settings['run_output_path'] + 'calib', exist_ok=True)
    
    # Identify what's what.
    print('Analyzing the input files ...')
    if hasattr(pcdata, redux_settings['spectrograph'] + 'Parser'):
        data_parser_class= getattr(pcdata, redux_settings['spectrograph'] + 'Parser')
        data = data_parser_class()(redux_settings)
    else:
        raise NotImplementedError("Must use one a supported instrument for now, or implement a new instrument.")
        #data_parser = pcdata.GeneralParser()
        #data = data_parser.categorize()(redux_settings)
            
    return redux_settings, data
    