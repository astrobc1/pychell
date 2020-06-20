# Default Python modules
import os
import glob
import importlib
import sys
import pdb
import pickle
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

def reduce_night(user_general_settings, user_extraction_settings, user_calib_settings, header_keys=None):

    # Start the main clock
    stopwatch = pcutils.StopWatch()

    # Parse the input directory, return the headers and the updated global parameters
    general_settings, calib_settings, extraction_settings, raw_data, master_calib_data = init_night(user_general_settings, user_extraction_settings, user_calib_settings, header_keys=header_keys)
    
    # Create Master Bias (only one)
    if calib_settings['bias_subtraction']:
        print('Creating Master Bias ...', flush=True)
        
        # Generate the data cube
        bias_cube = pcdata.SpecImage.generate_data_cube(raw_data['bias'])
        
        # Simple median combine, probably
        master_bias_image = pccalib.generate_master_bias(bias_cube)
        
        # Save
        master_calib_data['flats'][i].save_master_image(master_bias_image)
    
    # Create Master Darks for each unique exposure time
    if calib_settings['dark_subtraction']:
        print('Creating Master Dark(s) ...', flush=True)

        # Generate the data cube
        darks_cube = pcdata.SpecImage.generate_data_cube(raw_data['darks'])
        
        for i in range(len(master_calib_data['darks'])):
            
            # Generate the data cube
            darks_cube = pcdata.SpecDataImage.generate_data_cube(master_calib_data['darks'][i].calib_images)
            
            # Simple median combine, probably
            master_dark_image = pccalib.generate_master_flat(darks_cube)
            
            # Save
            master_calib_data['darks'][i].save_master_image(master_dark_image)
            
        
    # Create Master Flats for each flats group
    if calib_settings['flat_division']:
        
        print('Creating Master Flat(s) ...', flush=True)
        
        # For loop over groups
        for i in range(len(master_calib_data['flats'])):
            
            # Generate the data cube
            flats_cube = pcdata.SpecDataImage.generate_data_cube(master_calib_data['flats'][i].calib_images)
            
            # Simple median combine, probably
            master_flat_image = pccalib.generate_master_flat(flats_cube)
            
            # Save
            master_calib_data['flats'][i].save(master_flat_image)

    # Order Tracing
    # Order images and order dictionaries containing the height and polynomial coeffs for each order
    if extraction_settings['order_map'] == 'from_flats':
        
        print('Tracing Orders From Master Flat(s) ...', flush=True)
        
        for sci_data in raw_data['science']:
            sci_data.trace_orders(output_dir=general_settings['output_dir_root'] + 'calib' + os.sep, extraction_settings=extraction_settings, src=extraction_settings['order_map'])
        
        
    # Correct master flats
    if ('correct_fringing_in_flatfield' in calib_settings and calib_settings['correct_fringing_in_flatfield']) or ('correct_blaze_function_in_flatfield' in calib_settings and calib_settings['correct_blaze_function_in_flatfield']):
        
        print('Correcting artifacts in master flat(s) ...', flush=True)
        for sci_data in raw_data['science']:
            sci_data.correct_flat_artifacts(output_dir=general_settings['output_dir_root'], calibration_settings=calib_settings)

    # Extraction of science spectra
    if general_settings['n_cores'] > 1:
        
        print('Extracting Spectra In Parallel Using ' + str(general_settings['n_cores']) + ' Cores ...', flush=True)
        
        # Run in Parallel
        iter_pass = []

        for i in range(len(raw_data['science'])):
            iter_pass.append((raw_data['science'][i], general_settings, calib_settings, extraction_settings))
            
        # Call in parallel
        Parallel(n_jobs=general_settings['n_cores'], verbose=0, batch_size=1)(delayed(pcextract.extract_full_image)(*iter_pass[i]) for i in range(len(raw_data['science'])))
        
    else:
        # One at a time
        print('Extracting Spectra ...', flush=True)
        for i in range(len(raw_data['science'])):
            pcextract.extract_full_image(raw_data['science'][i], general_settings, calib_settings, extraction_settings)

    print('TOTAL RUN TIME: ' + str(round((stopwatch.time_since()) / 3600, 3)) + ' Hours')
    print('ALL FINISHED !')
    
    
def init_night(user_general_settings, user_extraction_settings, user_calib_settings, header_keys=None):
    
    # Helpful dictionaries
    general_settings = {}
    extraction_settings = {}
    calib_settings = {}
    
    # Load in the default pipeline config
    general_settings.update(pcconfig.general_settings)
    extraction_settings.update(pcconfig.extraction_settings)
    
    # Update with the default instrument dictionaries
    spec_module = importlib.import_module('pychell.spectrographs.' + user_general_settings['spectrograph'].lower())
    general_settings.update(spec_module.general_settings)
    calib_settings.update(spec_module.calibration_settings)
    extraction_settings.update(spec_module.extraction_settings)
    
    # Header keys
    if header_keys is None:
        header_keys = spec_module.header_keys
        
    # User settings
    general_settings.update(user_general_settings)
    extraction_settings.update(user_extraction_settings)
    calib_settings.update(user_calib_settings)

    # Get the base input directory. The reduction (output) folder will have this same name.
    base_input_dir = os.path.basename(os.path.normpath(general_settings['input_dir']))
    general_settings['output_dir_root'] = general_settings['output_dir'] + base_input_dir + os.sep
    
    if not os.path.isdir(general_settings['output_dir_root']):

        # Make the root output directory for this run
        os.makedirs(general_settings['output_dir_root'], exist_ok=True)

        # Trace Profiles
        os.makedirs(general_settings['output_dir_root'] + 'trace_profiles', exist_ok=True)

        # 1-dimensional spectra in text files with headers commented out (starting with #)
        os.makedirs(general_settings['output_dir_root'] + 'spectra', exist_ok=True)

        # Previews (Full order figures of the above)
        os.makedirs(general_settings['output_dir_root'] + 'previews', exist_ok=True)

        # Calibration (order map, flats, darks, bias)
        os.makedirs(general_settings['output_dir_root'] + 'calib', exist_ok=True)
        
    # Identify what's what. Only headers are stored here.
    print('Analyzing the input files ...')
    raw_data = {} # science, bias, darks, flats (individual images)
    master_calib_data = {} # bias, darks, flats (median combined images, possibly further corrected)
    
    # Science images
    sci_files = glob.glob(general_settings['input_dir'] + '*' + general_settings['sci_tag'] + '*.fits')
    raw_data['science'] = [pcdata.ScienceImage(input_file=sci_files[f], header_keys=header_keys, parse_header=True, output_dir_root=general_settings['output_dir_root'], time_offset=general_settings['time_offset'], filename_parser=general_settings['filename_parser'], img_num=f, n_tot_imgs=len(sci_files)) for f in range(len(sci_files))]
    
    # Bias. Assumes a single set of exposures to create a single master bias from
    if calib_settings['bias_subtraction']:
        
        # Parse the bias dir
        bias_files = glob.glob(general_settings['input_dir'] + '*' + general_settings['bias_tag'] + '*.fits')
        
        # Read in the bias and construct bias image objects.
        raw_data['bias'] = [pcdata.BiasImage(input_file=bias_files[f], header_keys=header_keys, parse_header=True, output_dir_root=general_settings['output_dir_root'], time_offset=general_settings['time_offset'], filename_parser=general_settings['filename_parser'], img_num=f, n_tot_imgs=len(bias_files)) for f in range(len(bias_files))]
        
        # Initialize a master bias object. No master bias is created yet.
        master_calib_data['bias'] = pcdata.MasterBiasImage(raw_data['bias'], output_dir=general_settings['output_dir_root'] + 'calib', header_keys=header_keys)
        
    # Darks, multiple exposure times
    if calib_settings['dark_subtraction']:
        
        # Parse the darks dir
        dark_files = glob.glob(general_settings['input_dir'] + '*' + general_settings['darks_tag'] + '*.fits')
        
        # Read in the darks and construct darkimage objects.
        raw_data['darks'] = [pcdata.DarkImage(input_file=dark_files[f], header_keys=header_keys, parse_header=True, output_dir_root=general_settings['output_dir_root'], time_offset=general_settings['time_offset'], filename_parser=general_settings['filename_parser'], img_num=f, n_tot_imgs=len(dark_files)) for f in range(len(dark_files))]
        
        # Initialize a master flat object. No master flat is created yet.
        master_calib_data['darks'] = pcdata.MasterDarkImage.from_all_darks(raw_data['darks'], output_dir=general_settings['output_dir_root'] + 'calib', header_keys=header_keys)
        
    # Flats, multiple per night possibly
    if calib_settings['flat_division']:
        
        # Parse the flats dir
        flat_files = glob.glob(general_settings['input_dir'] + '*' + general_settings['flats_tag'] + '*.fits')
        
        # Read in the flats and construct flatimage objects.
        raw_data['flats'] = [pcdata.FlatImage(input_file=flat_files[f], header_keys=header_keys, parse_header=True, output_dir_root=general_settings['output_dir_root'], time_offset=general_settings['time_offset'], filename_parser=general_settings['filename_parser'], img_num=f, n_tot_imgs=len(flat_files)) for f in range(len(flat_files))]
        
        # Initialize a master flat object. No master flat is created yet.
        master_calib_data['flats'] = pcdata.MasterFlatImage.from_all_flats(raw_data['flats'], output_dir=general_settings['output_dir_root'] + 'calib', header_keys=header_keys)
        
    # Go through science images and pair them with master dark and flat images
    for sci_data in raw_data['science']:
            
        if calib_settings['dark_subtraction']:
            sci_data.pair_master_dark(master_calib_data['darks'])
            
        if calib_settings['flat_division']:
            sci_data.pair_master_flat(master_calib_data['flats'])
            
            
    print_summary(raw_data)
            
    return general_settings, calib_settings, extraction_settings, raw_data, master_calib_data


def print_summary(raw_data):
    
    n_sci_tot = len(raw_data['science'])
    targets_all = np.array([raw_data['science'][i].target for i in range(n_sci_tot)], dtype='<U50')
    targets_unique = np.unique(targets_all)
    for i in range(len(targets_unique)):
        
        target = targets_unique[i]
        
        locs_this_target = np.where(targets_all == target)[0]
        
        sci_this_target = [raw_data['science'][j] for j in locs_this_target]
        
        print('Target: ' + target)
        print('    N Exposures: ' + str(locs_this_target.size))
        if hasattr(sci_this_target[0], 'master_bias'):
            print('    Master Bias File(s): ')
            print('    ' + raw_data['science'].master_bias.base_input_file)
            
        if hasattr(sci_this_target[0], 'master_dark'):
            darks_this_target_all = np.array([sci.master_dark for sci in sci_this_target], dtype=pcdata.DarkImage)
            darks_unique = np.unique(darks_this_target_all)
            print('  Master Dark File(s): ')
            for d in darks_unique:
                print('    ' + d.base_input_file)
            
        if hasattr(sci_this_target[0], 'master_flat'):
            flats_this_target_all = np.array([sci.master_flat for sci in sci_this_target], dtype=pcdata.FlatImage)
            flats_unique = np.unique(flats_this_target_all)
            print('  Master Flat File(s): ')
            for f in flats_unique:
                print('    ' + f.base_input_file)
                
        print('')
    