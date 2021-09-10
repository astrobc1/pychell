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

class Reducer:
    pass

class StandardReducer(Reducer):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, spectrograph, data_input_path, output_path, pre_calib=None, tracer=None, extractor=None, post_calib=None, n_cores=1):
        
        # The spectrograph
        self.spectrograph = spectrograph
        
        # Number of cores
        self.n_cores = n_cores
        
        # The data input path
        self.data_input_path = data_input_path
        
        # The output path
        self.output_path = output_path
        self.output_path += os.path.basename(data_input_path[0:-1]) + os.sep
        
        # Create the output directories
        self._create_output_dirs()
        
        # Init the reduction steps
        self.pre_calib = pre_calib
        self.tracer = tracer
        self.extractor = extractor
        self.post_calib = post_calib
        
        # Init the spectrograph and data
        self._init_spectrograph()
        
    def _create_output_dirs(self):
    
        # Make the root output directory for this run
        os.makedirs(self.output_path, exist_ok=True)

        # Trace information (positions, profiles)
        os.makedirs(self.output_path + "trace", exist_ok=True)

        # 1-dimensional spectra in fits files
        os.makedirs(self.output_path + "spectra", exist_ok=True)

        # Calibration (master bias, darks, flats, tellurics, wavecal, etc.)
        os.makedirs(self.output_path + "calib", exist_ok=True)
        
    def _init_spectrograph(self):
        
        # Load the spectrograph module
        spec_module = self.spec_module
        
        # Construct the data parser
        parser_class = getattr(spec_module, f"{self.spectrograph}Parser")
        self.parser = parser_class(self.data_input_path, self.output_path)
        
        # Identify what's what.
        print("Categorizing Data ...", flush=True)
        self.data = self.parser.categorize_raw_data(self)


    ##################################
    #### PRIMARY REDUCE / EXTRACT ####
    ##################################
    
    def reduce(self):

        # Start the main clock
        stopwatch = pcutils.StopWatch()
        
        # Generate master bias, dark, flat
        self.generate_master_calib_images()
        
        # Trace all orders (possibly multiple traces per order)
        self.trace()
        
        # Extract all images
        self.extract()
        
        # Run Time
        print(f"COMPLETE! TOTAL TIME: {round(stopwatch.time_since() / 3600, 2)} hours")
        
    def generate_master_calib_images(self):
        self.pre_calib.generate_master_calib_images(self)
    
    def trace(self):
        for order_map in self.data["order_maps"]:
            self.tracer.trace(order_map, self.n_cores)
        
    def extract(self):
        
        if self.n_cores > 1:
            
            print(f"Extracting Science Spectra In Parallel Using {self.n_cores} Cores ...", flush=True)
            
            # Call in parallel
            Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(self.extractor.extract_image)(self, data, i + 1) for i, data in enumerate(self.data["science"]))
            
        else:
            
            # One at a time
            print("Extracting Science Spectra In Series ...", flush=True)
            for i, data in enumerate(self.data["science"]):
                self.extractor.extract_image(self, data, i + 1)
    
    ###############
    #### MISC. ####
    ###############
    
    @property
    def spec_module(self):
        return importlib.import_module(f"pychell.data.{self.spectrograph.lower()}")

