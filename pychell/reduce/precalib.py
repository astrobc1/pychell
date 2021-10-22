# Default Python modules
import os

# Graphics
import matplotlib.pyplot as plt

# Science/Math
import numpy as np
import astropy.units as units
import scipy.interpolate
from astropy.io import fits

# Clustering algorithms (DBSCAN)
import sklearn.cluster
import scipy.signal

# LLVM
from numba import jit, njit, prange

# Pychell modules
import pychell.maths as pcmath
import pychell.data.spectraldata as pcspecdata

class PreCalibrator:
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, do_bias=False, do_dark=False, do_flat=True, flat_percentile=0.5):
        self.do_bias = do_bias
        self.do_dark = do_dark
        self.do_flat = do_flat
        self.flat_percentile = flat_percentile

    ################################
    #### GENERATE MASTER IMAGES ####
    ################################
    
    def gen_master_calib_images(self, data):
        self._gen_master_calib_images(data, self.do_bias, self.do_dark, self.do_flat, self.flat_percentile)

    @staticmethod
    def _gen_master_calib_images(data, do_bias, do_dark, do_flat, flat_percentile=None):
        
        # Bias image (only 1)
        if do_bias:
            PreCalibrator.gen_master_bias()
            
        # Master Dark images
        if do_dark:
            print('Creating Master Dark(s) ...', flush=True)
            for mdark in data['master_darks']:
                mdark_image = PreCalibrator.gen_master_dark(mdark.group, do_bias)
                mdark.save(mdark_image)
            
        # Flat field images
        if do_flat:
            print('Creating Master Flat(s) ...', flush=True)
            for mflat in data['master_flats']:
                mflat_image = PreCalibrator.gen_master_flat(mflat.group, do_bias, do_dark, flat_percentile)
                mflat.save(mflat_image)

    @staticmethod
    def gen_master_flat(group, do_bias, do_dark, flat_percentile=0.5):
        
        # Generate a data cube
        n_flats = len(group)
        flats_cube = pcspecdata.Echellogram.generate_cube(group)
        
        # For each flat, subtract master dark and bias
        # Also normalize each image and remove obvious bad pixels
        for i in range(n_flats):
            if do_bias:
                master_bias = group[i].master_bias.parse_image()
                flats_cube[i, :, :] -= master_bias
            if do_dark:
                master_dark = group[i].master_dark.parse_image()
                flats_cube[i, :, :] -= master_dark
            bad = np.where(flats_cube[i, :, :] < 0)
            if bad[0].size > 0:
                flats_cube[i, :, :][bad] = np.nan
        
        # Median crunch, flag one more time
        master_flat = np.nanmedian(flats_cube, axis=0)
        master_flat /= pcmath.weighted_median(master_flat, percentile=flat_percentile)
        bad = np.where((master_flat <= flat_percentile*0.01) | (master_flat > flat_percentile * 100))
        if bad[0].size > 0:
            master_flat[bad] = np.nan
            
        # Return
        return master_flat

    @staticmethod
    def gen_master_dark(group, do_bias):
        """Computes a median master flat field image from a subset of flat field images. Dark and bias subtraction is also performed if set.

            Args:
                group (list): The list of DarkImages.
                bias_subtraction (bool): Whether or not to perform bias subtraction. If so, each dark must have a master_bias attribute.
            Returns:
                master_dark (np.ndarray): The median combined master dark image
        """
        darks_cube = pcspecdata.Echellogram.generate_cube(group)
        if do_bias:
            for i in darks_cube.shape[0]:
                master_bias = group[i].master_bias.parse_image()
                darks_cube[i, :, :] -= master_bias
        master_dark = np.nanmedian(darks_cube, axis=0)
        return master_dark

    @staticmethod
    def gen_master_bias(group):
        """Generates a median master bias image.

        Args:
            group (list): The list of BiasImages.
        Returns:
            master_bias (np.ndarray): The master bias image.
        """
        bias_cube = group[0].generate_cube(group)
        master_bias = np.nanmedian(bias_cube, axis=0)
        return master_bias
        
    ########################################################
    #### STANDARD CALIBRATION METHOD FOR A SINGLE TRACE ####
    ########################################################

    def pre_calibrate(self, data, data_image):
        self._pre_calibrate(data, data_image, self.do_bias, self.do_dark, self.do_flat)

    @staticmethod
    def _pre_calibrate(data, data_image, do_bias, do_dark, do_flat):
        
        # Bias correction
        if do_bias:
            master_bias_image = data.master_bias.parse_image()
            data_image -= master_bias_image
            
        # Dark correction
        if do_dark:
            master_dark_image = data.master_dark.parse_image()
            data_image -= master_dark_image
            
        # Flat division
        if do_flat:
            master_flat_image = data.master_flat.parse_image()
            data_image /= master_flat_image