# Import the reduction code driver
import pychell.reduce.driver

# OS tools
import os

# This must be defined. See README for key description.
general_settings = {
    "input_dir": "Vega" + os.sep, # the input directory the raw data lives in.
    "output_dir": os.path.dirname(os.path.realpath(__file__)) + os.sep + 'outputs' + os.sep, # For current directory
    "instrument": "iSHELL",
    "n_cores": 1
}

# This must be defined. See README for key description.
extraction_settings = {
    "order_map": 'from_flats'
}

# This must be defined. See README for key description.
calib_settings = {
    
    "bias_subtraction": False,
    "dark_subtraction": False,
    "flat_division": True
}

pychell.reduce.driver.reduce_night(general_settings, extraction_settings, calib_settings, header_keys=None)