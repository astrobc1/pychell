# Import the reduction code driver
import pychell.reduce.driver

# OS tools
import os

# This must be defined. See README for key description.
redux_settings = {
    
    # Input and output path
    "input_path": "Vega" + os.sep, # the input directory the raw data lives in.
    "output_path_root": os.path.dirname(os.path.realpath(__file__)) + os.sep + 'outputs' + os.sep, # For current directory
    "spectrograph": "iSHELL",
    "n_cores": 1,
    
    # Calibration
    "bias_subtraction": False,
    "dark_subtraction": False,
    "flat_division": True,
    "correct_fringing_in_flatfield": False,
    "correct_blaze_function_in_flatfield": False,
    
    # Order map
    'mask_trace_edges':  5,
    'order_map': {'source': 'empirical_from_flat_fields', 'method': None},
    
    'oversample': 4,
    
    'pmassey_settings': {'bad_thresh': [20, 16, 12]}
}

pychell.reduce.driver.reduce_night(redux_settings)