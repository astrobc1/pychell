# Base Python
import os
import pickle

# Maths
import numpy as np
from astropy.io import fits

# Pychell deps
import pychell.maths as pcmath

####################
#### BASE TYPES ####
####################

class SpecData:
    
    def __init__(self, input_file=None):
        self.input_file = input_file
    
    def __eq__(self, other):
        return self.input_file == other.input_file

    @property
    def base_input_file(self):
        return os.path.basename(self.input_file)

    @property
    def input_file_noext(self):
        return os.path.splitext(self.base_input_file)[0]

    @property
    def base_input_file_noext(self):
        return os.path.basename(self.input_file_noext)

    @property
    def input_path(self):
        return os.path.split(self.input_file)[0] + os.sep

class Echellogram(SpecData):
    
    @staticmethod
    def generate_cube(data):
        """Generates a data-cube (i.e., stack) of images.

        Args:
            data_list (list): A list of data objects.
        Returns:
            data_cube (np.ndarray): The generated data cube, with shape=(n_images, ny, nx).
        """
        n_data = len(data)
        data0 = data[0].parse_image()
        ny, nx = data0.shape
        data_cube = np.empty(shape=(n_data, ny, nx), dtype=float)
        data_cube[0, :, :] = data0
        for idata in range(1, n_data):
            data_cube[idata, :, :] = data[idata].parse_image()
            
        return data_cube
        
    def parse_image(self):
        return self.specmod.parse_image(self)
    
    def parse_header(self):
        return self.specmod.parse_image_header(self)
    
    def __repr__(self):
        return self.base_input_file

class RawEchellogram(Echellogram):

    def __init__(self, input_file, specmod=None):
        
        # Call super init
        super().__init__(input_file)

        # Store the specmod
        self.specmod = specmod

        # Parse the header
        if self.specmod is not None:
            self.specmod.parse_image_header(self)

class MasterCal(Echellogram):

    def __init__(self, group, output_path):

        # The individual frames
        self.group = group

        # The specmod
        self.specmod = self.group[0].specmod

        # The input filename
        input_file = output_path + self.specmod.gen_master_calib_filename(self)
        
        # Call super init
        super().__init__(input_file)

        # Create a header
        self.header = self.specmod.gen_master_calib_header(self)

    def save(self, image):
        fits.writeto(self.input_file, image, self.header, overwrite=True, output_verify='ignore')


#####################
#### 1D SPECTRUM ####
#####################

class SpecData1d(SpecData):
    
    # Store the input file, spec, and order num
    def __init__(self, input_file, order_num, spec_num, specmod, crop_pix):

        super().__init__(input_file)
        
        # The specmod to parse data
        self.specmod = specmod
        
        # Order number and observation number
        self.order_num = order_num
        self.spec_num = spec_num
            
        # Default wavelength and LSF grid, may be overwritten in custom parse method.
        self.wave = None
        self.lsf_width = None
        
        # Extra cropping
        self.crop_pix = crop_pix
        
        # Parse
        self.parse()

    def parse(self):
        
        # Parse the data
        self.specmod.parse_spec1d(self)
        
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
        if self.wave is not None:
            bad = np.where(~np.isfinite(self.wave))[0]
            if bad.size > 0:
                self.wave[bad] = np.nan
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
            
    def parse_header(self):
        self.header = fits.open(self.input_file)[0].header
        return self.header

    def __repr__(self):
        return f"1d spectrum: {self.base_input_file}"