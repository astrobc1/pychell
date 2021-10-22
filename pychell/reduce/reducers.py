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
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, spectrograph, data_input_path, output_path, pre_calib=None, tracer=None, extractor=None, wave_cal=None, lsf_cal=None, n_cores=1):
        
        # The spectrograph
        self.spectrograph = spectrograph
        
        # Number of cores
        self.n_cores = n_cores
        
        # The data input path
        self.data_input_path = data_input_path
        
        # The output path
        self.output_path = output_path if output_path[-1] == os.sep else output_path + os.sep
        self.output_path += os.path.basename(data_input_path[0:-1]) + os.sep

        # Reduction steps
        self.pre_calib = pre_calib
        self.tracer = tracer
        self.extractor = extractor
        self.lsf_cal = lsf_cal

        # Create the output directories
        self.create_output_dirs()
        
        # Init the spectrograph and data
        self.init_spectrograph()
        
    def create_output_dirs(self):
    
        # Make the root output directory for this run
        os.makedirs(self.output_path, exist_ok=True)

        # Trace information (positions, profiles)
        os.makedirs(self.output_path + "trace", exist_ok=True)

        # 1-dimensional spectra in fits files
        os.makedirs(self.output_path + "spectra", exist_ok=True)

        # Calibration (master bias, darks, flats, tellurics, wavecal, etc.)
        os.makedirs(self.output_path + "calib", exist_ok=True)
        
    def init_spectrograph(self):
        
        # Load the spectrograph module
        spec_module = self.spec_module
        
        # Construct the data parser
        parser_class = getattr(spec_module, f"{self.spectrograph}Parser")
        self.parser = parser_class()
        
        # Identify what's what.
        print("Categorizing Data ...", flush=True)
        self.data = self.parser.categorize_raw_data(self)
    
    ##########################
    #### PRIMARY ROUTINES ####
    ##########################

    def reduce(self):
        """Primary method to reduce a full directory. Steps performed are:
            1. Generate and save all master calibration images (self.gen_master_calib_images).
            2. Trace orders for all images (self.trace).
            3. Extract all desired spectra (includes precalibrating (bias, dark, flat), trace positions, profile, and background scatter calculation) (self.extract).
        """

        # Start the main clock
        stopwatch = pcutils.StopWatch()
        
        # 1. Generate master bias, dark, flats
        self.gen_master_calib_images()
        
        # Trace orders for appropriate frames
        self.trace()
        
        # Extract all images
        self.extract()
        
        # Run Time
        print(f"REDUCTION COMPLETE! TOTAL TIME: {round(stopwatch.time_since() / 3600, 2)} hours")
        
    def gen_master_calib_images(self):
        if self.pre_calib is not None:
            self.pre_calib.gen_master_calib_images(self.data)
    
    def trace(self):
        for order_map in self.data["order_maps"]:
            print(f"Tracing orders for {order_map} ...", flush=True)
            self.tracer.trace(order_map)
            with open(f"{self.output_path}{os.sep}trace{os.sep}{order_map.base_input_file_noext}_order_map.pkl", 'wb') as f:
                pickle.dump(order_map.orders_list, f)

    def extract(self):
        
        if self.n_cores > 1:
            
            print(f"Extracting Spectra In Parallel Using {self.n_cores} Cores ...", flush=True)
            
            # Call in parallel
            Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(self.extract_image_wrapper)(self, data, i + 1, len(self.data["extract"])) for i, data in enumerate(self.data["extract"]))
            
        else:
            
            # One at a time
            print("Extracting Spectra In Series ...", flush=True)
            for i, data in enumerate(self.data["extract"]):
                self.extract_image_wrapper(self, data, i + 1, len(self.data["extract"]))

    @staticmethod
    def extract_image_wrapper(reducer, data, image_num, n_extract_tot):

        # Stopwatch
        stopwatch = pcutils.StopWatch()

        # Print start
        print(f"Extracting Image {image_num} of {n_extract_tot} [{data}]")

        # Load image
        data_image = data.parse_image()

        # Calibrate image
        if reducer.pre_calib is not None:
            reducer.pre_calib.pre_calibrate(data, data_image)

        # Mask
        if 'badpix_mask' in reducer.data:
            badpix_mask = reducer.data['badpix_mask']
        else:
            badpix_mask = None
        
        # Extract image
        reducer.extractor.extract_image(reducer, data, data_image, badpix_mask=badpix_mask)
        

        # Print end
        print(f"Extracted Image {image_num} of {n_extract_tot} [{data}] in {round(stopwatch.time_since(), 2) / 60} min")

    def compile_reduced_outputs(self):
        self.parser.compile_reduced_outputs(self)


    ###############
    #### MISC. ####
    ###############

    @property
    def spec_module(self):
        return importlib.import_module(f"pychell.data.{self.spectrograph.lower()}")

