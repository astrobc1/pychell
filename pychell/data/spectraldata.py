# Base Python
import os
import pickle

# Maths
import numpy as np
from astropy.io import fits

# Pychell deps
import pychell.maths as pcmath
import pychell.reduce.order_map as pcomap

# Optimize deps
from optimize.data import Dataset

####################
#### BASE TYPES ####
####################

class SpecData(Dataset):
    
    def __init__(self, input_file=None):
        super().__init__()
        self.input_file = input_file
        self.base_input_file = os.path.basename(self.input_file)
        self.get_file_type()
    
    # Determine image extension (if any) and remove
    def get_file_type(self):
        if self.base_input_file.endswith('.fits'):
            self.input_file_noext = self.input_file[0:-5]
        elif self.base_input_file.endswith('.fz'):
            self.input_file_noext = self.input_file[0:-3]
        else:
            k = self.input_file.rfind('.')
            if k == -1: # no extentsion!
                self.input_file_noext = self.base_input_file
            else: # found extension!
                self.input_file_noext = self.base_input_file[0:k]
                
        self.base_input_file_noext = os.path.basename(self.input_file_noext)
        
    def parse_header(self):
        self.header = fits.open(self.input_file)[0].header
        return self.header
        
    def __repr__(self):
        return f"Spectrum: {self.base_input_file}"
    
    def __eq__(self, other):
        return self.input_file == other.input_file

    def __gt__(self, other):
        return self.time_obs_start > other.time_obs_start
        
    def __lt__(self, other):
        return self.time_obs_start < other.time_obs_start

class Echellogram(SpecData):
    
    # Given a n iterable of SpecDataImage objects
    # this parses the images from their respective files and returns them as a cube
    @staticmethod
    def generate_cube(data_list):
        """Generates a data-cube (i.e., stack) of images.

        Args:
            data_list (list): A list of data objects.
        Returns:
            data_cube (np.ndarray): The generated data cube, with shape=(n_images, ny, nx).
        """
        n_data = len(data_list)
        data0 = data_list[0].parse_image()
        ny, nx = data0.shape
        data_cube = np.empty(shape=(n_data, ny, nx), dtype=float)
        data_cube[0, :, :] = data0
        for idata in range(1, n_data):
            data_cube[idata, :, :] = data_list[idata].parse_image()
            
        return data_cube
        
    def parse_image(self):
        return self.parser.parse_image(self)
    
    def __repr__(self):
        return f"Echellogram: {self.base_input_file}"


######################
#### ECHELLOGRAMS ####
######################

class RawImage(Echellogram):
    
    def __init__(self, input_file=None, parser=None):
        
        # Call super init
        super().__init__(input_file=input_file)
        
        # Store the parser
        self.parser = parser
        
        # Parse the header
        if self.parser is not None:
            self.parse_header()
            
        # Parse the image number
        try:
            self.parser.parse_image_num(self)
        except:
            print("Unknown image number")
        
        # Parse the date of the observation
        try:
            self.parser.parse_utdate(self)
        except:
            print("Unknown UT date")
            
        # Determine the number of traces per order
        try:
            self.n_traces = self.parser.get_n_trace()
        except:
            self.n_traces = 1
            
        # Determine the number of orders
        try:
            self.n_orders = self.parser.get_n_orders()
        except:
            self.n_orders = 1
        
    def parse_data(self):
        return self.parser.parse_image(self)
    
    def parse_header(self):
        return self.parser.parse_image_header(self)
    
    def __repr__(self):
        return self.base_input_file

class MasterCalibImage(Echellogram):
    
    def __init__(self, individuals, input_file=None, parser=None):
        
        # Store the individual names
        self.individuals = individuals
            
        super().__init__(input_file=input_file)
        
        self.parser = parser
        
        self.parser.gen_master_calib_header(self)
        
    def save(self, master_image):
        hdu = fits.PrimaryHDU(master_image, header=self.header)
        hdu.writeto(self.input_file, overwrite=True)
        
    def __repr__(self):
        return f"Master Calibration Image: {self.base_input_file}"

class ImageMap(Echellogram):
    
    def __init__(self, input_file=None, source=None, parser=None, order_map_fun=None, orders_list=None):
            
        super().__init__(input_file=input_file)
        
        self.input_file_orders_list = self.input_file_noext + ".pkl"
        
        # The source for the image map (ie, slit flat, fiber flat)
        self.source = source
        
        # The parser
        self.parser = parser
        
        # orders_list (len=n_orders)
        self.orders_list = orders_list
        
    def trace_orders(self, config):
        self.order_map_fun(self, config)
    
    def load_map_image(self):
        return self.parse_image()
    
    def load_orders_list(self):
        with open(self.input_file_orders_list, 'rb') as handle:
            orders_list = pickle.load(self.orders_list, handle)
        self.orders_list = orders_list
    
    def save_map_image(self, order_map_image):
        hdu = fits.PrimaryHDU(order_map_image, header=self.source.header)
        hdu.writeto(self.input_file, overwrite=True)
        
    def save_orders_list(self):
        with open(self.input_file_orders_list, 'wb') as handle:
            pickle.dump(self.orders_list, handle)
    
    def save(self, order_map_image):
        self.save_map_image(order_map_image)
        self.save_orders_list()
        
    def __repr__(self):
        return f"Image map: {self.base_input_file}"


#####################
#### 1D SPECTRUM ####
#####################

class SpecData1d(SpecData):
    
    # Store the input file, spec, and order num
    def __init__(self, input_file, order_num, spec_num, parser, crop_pix):

        super().__init__(input_file)
        
        # The parser to load and write data
        self.parser = parser
        
        # Order number and observation number
        self.order_num = order_num
        self.spec_num = spec_num
            
        # Default wavelength and LSF grid, may be overwritten in custom parse method.
        self.apriori_wave_grid = None
        self.apriori_lsf = None
        
        # Extra cropping
        self.crop_pix = crop_pix
        
        # Parse
        self.parse()

    def parse(self):
        
        # Parse the data
        self.parser.parse_spec1d(self)
        
        # Normalize to 98th percentile
        medflux = pcmath.weighted_median(self.flux, percentile=0.98)
        self.flux /= medflux
        self.flux_unc /= medflux
        
        # Enforce the pixels are cropped (ideally they are already cropped and this has no effect, but still optional)
        if self.crop_pix is not None:
            if self.crop_pix[0] > 0:
                self.flux[0:self.crop_pix[0]] = np.nan
                self.flux_unc[0:self.crop_pix[0]] = np.nan
                self.mask[0:self.crop_pix[0]] = 0
            if self.crop_pix[1] > 0:
                self.flux[-self.crop_pix[1]:] = np.nan
                self.flux_unc[-self.crop_pix[1]:] = np.nan
                self.mask[-self.crop_pix[1]:] = 0
            
        # Sanity
        bad = np.where((self.flux <= 0.0) | ~np.isfinite(self.flux) | (self.mask == 0) | ~np.isfinite(self.mask) | ~np.isfinite(self.flux_unc))[0]
        if bad.size > 0:
            self.flux[bad] = np.nan
            self.flux_unc[bad] = np.nan
            self.mask[bad] = 0
            
        # More sanity
        if self.apriori_wave_grid is not None:
            bad = np.where(~np.isfinite(self.apriori_wave_grid))[0]
            if bad.size > 0:
                self.apriori_wave_grid[bad] = np.nan
                self.flux[bad] = np.nan
                self.flux_unc[bad] = np.nan
                self.mask[bad] = 0
            
        # Further flag any clearly deviant pixels
        flux_smooth = pcmath.median_filter1d(self.flux, width=7)
        bad = np.where(np.abs(flux_smooth - self.flux) > 0.3)[0]
        if bad.size > 0:
            self.flux[bad] = np.nan
            self.flux_unc[bad] = np.nan
            self.mask[bad] = 0
            
        # Check if 1d spectrum is even worth using
        if np.nansum(self.mask) < self.mask.size / 4:
            self.is_good = False
        else:
            self.is_good = True
            
  
    def __repr__(self):
        return f"1d spectrum: {self.base_input_file}"