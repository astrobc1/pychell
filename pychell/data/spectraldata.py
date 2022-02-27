# Base Python
import os
import pickle
import importlib

import numpy as np

# Fits
from astropy.io import fits

# Pychell deps
import pychell.utils as pcutils

####################
#### BASE TYPES ####
####################

class SpecData:

    __slots__ = ['input_file', 'header', 'spectrograph', 'spec_mod_func']
    
    def __init__(self, input_file, spectrograph, spec_mod_func=None, parse_header=True):
        """Base constructor for a SpecData object.

        Args:
            fits (str): The astropy fits object.
            spectrograph (str): The name of the spectrographf for dispatch.
            spec_module_func (method): A method to modify an existing spectrograph module.
        """
        self.input_file = input_file
        self.spectrograph = spectrograph
        self.spec_mod_func = spec_mod_func
        if parse_header:
            self.header = self.parse_header()

    def parse_header(self):
        return self.spec_module.parse_header(self.input_file)

    @property
    def base_input_file(self):
        """The input file without the path.

        Returns:
            str: The input file without the path.
        """
        return os.path.basename(self.input_file)

    @property
    def input_file_noext(self):
        """The input file without the extension.

        Returns:
            str: The input file without the extension.
        """
        return os.path.splitext(self.base_input_file)[0]

    @property
    def base_input_file_noext(self):
        """The input file (filename only) with no extension.

        Returns:
            str: The input file (filename only) with no extension.
        """
        return os.path.basename(self.input_file_noext)

    @property
    def input_path(self):
        """The input path without the filename.

        Returns:
            str: The input path without the filename.
        """
        return os.path.split(self.input_file)[0] + os.sep

    @property
    def spec_module(self):
        return pcutils.get_spec_module(self.spectrograph, self.spec_mod_func)

    def __repr__(self):
        return self.base_input_file_noext


class Echellogram(SpecData):

    def parse_image(self):
        """Parses the image.

        Returns:
            numpy.ndarray: The image.
        """
        return self.spec_module.parse_image(self)

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


class MasterCal(Echellogram):

    def __init__(self, group, output_path):
        """Construct a MasterCal object for a master calibration frame.

        Args:
            group (list): A list of the individual exposures (RawEchellogram objects) used to create this master cal.
            output_path (str): The output path to store this master cal frame once created.
        """

        # The individual frames
        self.group = group

        # The input filename
        input_file = output_path + self.group[0].spec_module.gen_master_calib_filename(self)
        
        # Call super init
        super().__init__(input_file, self.group[0].spectrograph, parse_header=False)

        # Create a header
        self.header = self.spec_module.gen_master_calib_header(self)

    def save(self, image):
        fits.writeto(self.input_file, image, self.header, overwrite=True, output_verify='ignore')


#####################
#### 1D SPECTRUM ####
#####################

class SpecData1d(SpecData):

    __slots__ = ['input_file', 'header', 'data', 'spectrograph', 'spec_mod_func', 'sregion']
    
    def __init__(self, input_file, spectrograph, sregion, spec_mod_func=None):
        self.sregion = sregion # This might not be needed here!
        super().__init__(input_file, spectrograph, spec_mod_func)
        self.data = self.parse_spec1d(sregion)


    def parse_spec1d(self, sregion):
        """Parse the 1d spectrum (including wavelength, flux, flux uncertainty, and mask).
        """
        return self.spec_module.parse_spec1d(self.input_file, sregion)


    @property
    def wave(self):
        return self.data["wave"]

    @property
    def flux(self):
        return self.data["flux"]

    @property
    def fluxerr(self):
        return self.data["fluxerr"]

    @property
    def mask(self):
        return self.data["mask"]
