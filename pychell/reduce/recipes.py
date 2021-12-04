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
        """Construct a recipe to reduce a directory.

        Args:
            spectrograph (str): The name of the spectrograph.
            data_input_path (str): The full path to the input directory.
            output_path (str): The full path to the output directory.
            do_bias (bool, optional): Whether or not to perform a bias correction. Defaults to False.
            do_dark (bool, optional): Whether or not to perform a dark subtraction. Defaults to True.
            do_flat (bool, optional): Whether or not to perform flat fielding. Defaults to True.
            flat_percentile (float, optional): The flat field percentile. Defaults to 0.5.
            mask_left (int, optional): The number of pixels to mask on the left of each image. Defaults to 50.
            mask_right (int, optional): The number of pixels to mask on the right of each image. Defaults to 50.
            mask_top (int, optional): The number of pixels to mask on the top of each image. Defaults to 10.
            mask_bottom (int, optional): The number of pixels to mask on the bottom of each image. Defaults to 10.
            tracer (OrderTracer, optional): The tracer object.
            extractor (SpectralExtractor, optional): The extractor object.
            n_cores (int, optional): The number of cpus to use. Defaults to 1.
        """
        
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
        """Creates the output folder and subfolders trace, spectra, calib.
        """
    
        # Make the root output directory for this run
        os.makedirs(self.output_path, exist_ok=True)

        # Trace information (positions, profiles)
        os.makedirs(self.output_path + "trace", exist_ok=True)

        # 1-dimensional spectra in fits files
        os.makedirs(self.output_path + "spectra", exist_ok=True)

        # Calibration (master bias, darks, flats, tellurics, wavecal, etc.)
        os.makedirs(self.output_path + "calib", exist_ok=True)
        
    def init_data(self):
        """Initialize the data by calling self.categorize_raw_data.
        """
        
        # Identify what's what.
        print("Categorizing Data ...", flush=True)
        self.data = self.spec_module.categorize_raw_data(self.data_input_path, self.output_path)

        # Print reduction summary
        self.print_reduction_summary()
    
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

        # Generate pre calibration images
        pccalib.gen_master_calib_images(self.data, self.do_bias, self.do_dark, self.do_flat, self.flat_percentile)
        
        # Trace orders for appropriate images
        self.trace()
        
        # Extract all desired images
        self.extract()
        
        # Run Time
        print(f"REDUCTION COMPLETE! TOTAL TIME: {round(stopwatch.time_since() / 3600, 2)} hours")
    
    def trace(self):
        """Traces the orders.
        """
        for order_map in self.data["order_maps"]:
            print(f"Tracing orders for {order_map} ...", flush=True)
            self.tracer.trace(self, order_map)
            with open(f"{self.output_path}{os.sep}trace{os.sep}{order_map.base_input_file_noext}_order_map.pkl", 'wb') as f:
                pickle.dump(order_map.orders_list, f)

    def extract(self):
        """Extract all spectra.
        """
        
        if self.n_cores > 1:
            
            print(f"Extracting Spectra In Parallel Using {self.n_cores} Cores ...", flush=True)
            
            # Call in parallel
            Parallel(n_jobs=self.n_cores, verbose=0, batch_size=1)(delayed(self.extract_image_wrapper)(self, data) for data in self.data["extract"])
            
        else:
            
            # One at a time
            print("Extracting Spectra In Series ...", flush=True)
            for i, data in enumerate(self.data["extract"]):
                self.extract_image_wrapper(self, data)

    @staticmethod
    def extract_image_wrapper(recipe, data):
        """Wrapper to extract an image for parallel processing. Performs the pre calibration.

        Args:
            recipe (ReduceRecipe): The recpe object.
            data (RawEchellogram): The data object.
        """

        # Stopwatch
        stopwatch = pcutils.StopWatch()

        # Print start
        print(f"Extracting {data} ...")

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
        print(f"Extracted {data} in {round(stopwatch.time_since() / 60, 2)} min")


    ###############
    #### MISC. ####
    ###############

    def print_reduction_summary(self):
        
        n_sci_tot = len(self.data['science'])
        targets_all = np.array([self.data['science'][i].object for i in range(n_sci_tot)], dtype='<U50')
        targets_unique = np.unique(targets_all)
        for i in range(len(targets_unique)):
            
            target = targets_unique[i]
            
            locs_this_target = np.where(targets_all == target)[0]
            
            sci_this_target = [self.data['science'][j] for j in locs_this_target]
            
            print('Target: ' + target)
            print('    N Exposures: ' + str(locs_this_target.size))
            if hasattr(sci_this_target[0], 'master_bias'):
                print('  Master Bias: ')
                print('    ' + str(sci_this_target[0].master_bias))
                
            if hasattr(sci_this_target[0], 'master_dark'):
                print('  Master Dark: ')
                print('    ' + str(sci_this_target[0].master_dark))
                
            if hasattr(sci_this_target[0], 'master_flat'):
                print('  Master Flat: ')
                print('    ' + str(sci_this_target[0].master_flat))

    @property
    def spec_module(self):
        return importlib.import_module(f"pychell.data.{self.spectrograph.lower()}")

