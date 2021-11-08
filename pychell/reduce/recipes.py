# Default Python modules
import os
import pickle
import warnings
import importlib

# Maths
import numpy as np
np.seterr(invalid='ignore', divide='ignore')
warnings.filterwarnings('ignore')

# Graphics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Parallelization
from joblib import Parallel, delayed

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
import pychell.data as pcdata
import pychell.reduce.precalib as pccalib

class ReduceRecipe:
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, spectrograph, data_input_path, output_path, do_bias=False, do_dark=True, do_flat=True, flat_percentile=0.5, mask_left=50, mask_right=50, mask_top=10, mask_bottom=10, tracer=None, extractor=None, n_cores=1):
        
        # The spectrograph
        self.spectrograph = spectrograph
        
        # Number of cores
        self.n_cores = n_cores
        
        # The data input path
        self.data_input_path = data_input_path
        
        # The output path
        self.output_path = output_path if output_path[-1] == os.sep else output_path + os.sep
        self.output_path += os.path.basename(data_input_path[0:-1]) + os.sep

        # Pre calibration
        self.do_bias = do_bias
        self.do_flat = do_flat
        self.do_dark = do_dark
        self.flat_percentile = flat_percentile

        # Image area
        self.mask_left = mask_left
        self.mask_right = mask_right
        self.mask_top = mask_top
        self.mask_bottom = mask_bottom

        # Reduction steps
        self.tracer = tracer
        self.extractor = extractor
        
    def create_output_dirs(self):
    
        # Make the root output directory for this run
        os.makedirs(self.output_path, exist_ok=True)

        # Trace information (positions, profiles)
        os.makedirs(self.output_path + "trace", exist_ok=True)

        # 1-dimensional spectra in fits files
        os.makedirs(self.output_path + "spectra", exist_ok=True)

        # Calibration (master bias, darks, flats, tellurics, wavecal, etc.)
        os.makedirs(self.output_path + "calib", exist_ok=True)
        
    def init_data(self):
        
        # Identify what's what.
        print("Categorizing Data ...", flush=True)
        self.data = self.spec_module.categorize_raw_data(self.data_input_path, self.output_path)
    
    ##########################
    #### PRIMARY ROUTINES ####
    ##########################

    def reduce(self):
        """Primary method to reduce a given directory.
        """

        # Start the main clock
        stopwatch = pcutils.StopWatch()

        # Create the output directories
        self.create_output_dirs()
        
        # Init the spectrograph and data
        self.init_data()

        # Generate pre calibration images
        pccalib.gen_master_calib_images(self.data, self.do_bias, self.do_dark, self.do_flat, self.flat_percentile)
        
        # Trace orders for appropriate images
        self.trace()
        
        # Extract all desired images
        self.extract()
        
        # Run Time
        print(f"REDUCTION COMPLETE! TOTAL TIME: {round(stopwatch.time_since() / 3600, 2)} hours")
    
    def trace(self):
        for order_map in self.data["order_maps"]:
            print(f"Tracing orders for {order_map} ...", flush=True)
            self.tracer.trace(self, order_map)
            with open(f"{self.output_path}{os.sep}trace{os.sep}{order_map.base_input_file_noext}_order_map.pkl", 'wb') as f:
                pickle.dump(order_map.orders_list, f)

    def extract(self):
        
        if self.n_cores > 1:
            
            print(f"Extracting Spectra In Parallel Using {self.n_cores} Cores ...", flush=True)
            
            # Call in parallel
            Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1, prefer="threads")(delayed(self.extract_image_wrapper)(self, data, i + 1, len(self.data["extract"])) for i, data in enumerate(self.data["extract"]))
            
        else:
            
            # One at a time
            print("Extracting Spectra In Series ...", flush=True)
            for i, data in enumerate(self.data["extract"]):
                self.extract_image_wrapper(self, data, i + 1, len(self.data["extract"]))

    @staticmethod
    def extract_image_wrapper(recipe, data, image_num, n_extract_tot):

        # Stopwatch
        stopwatch = pcutils.StopWatch()

        # Print start
        print(f"Extracting Image {image_num} of {n_extract_tot} [{data}]")

        # Load image
        data_image = data.parse_image()

        # Calibrate image
        pccalib.pre_calibrate(data, data_image, recipe.do_bias, recipe.do_dark, recipe.do_flat)

        # Mask
        if 'badpix_mask' in recipe.data:
            badpix_mask = recipe.data['badpix_mask']
        else:
            badpix_mask = None
        
        # Extract image
        recipe.extractor.extract_image(recipe, data, data_image, badpix_mask=badpix_mask)
        

        # Print end
        print(f"Extracted Image {image_num} of {n_extract_tot} [{data}] in {round(stopwatch.time_since(), 2) / 60} min")

    ###############
    #### MISC. ####
    ###############

    @property
    def spec_module(self):
        return importlib.import_module(f"pychell.data.{self.spectrograph.lower()}")

