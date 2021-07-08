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
import pychell.maths as pcmaths
import pychell.data as pcdata

class PreCalibrator:
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, do_bias=False, do_dark=False, do_flat=True, flat_percentile=0.5, remove_blaze_from_flat=False):
        self.do_bias = do_bias
        self.do_dark = do_dark
        self.do_flat = do_flat
        self.flat_percentile = flat_percentile
        self.remove_blaze_from_flat = remove_blaze_from_flat

    ################################
    #### GENERATE MASTER IMAGES ####
    ################################
    
    def generate_master_calib_images(self, reducer):
        
        # Bias image (only 1)
        if self.do_bias:
            self.generate_master_bias()
            
        # Master Dark image
        if self.do_dark:
            self.generate_master_darks()
            
        # Flat field image
        if self.do_flat:
            
            print('Creating Master Flat(s) ...', flush=True)
        
            # Create master flat image and save it
            for mflat in reducer.data['master_flats']:
                mflat_image = self.generate_master_flat(mflat.individuals)
                mflat.save(mflat_image)

    def generate_master_flat(self, individuals):
        
        # Generate a data cube
        n_flats = len(individuals)
        flats_cube = pcdata.Echellogram.generate_cube(individuals)
        
        # For each flat, subtract master dark and bias
        # Also normalize each image and remove obvious bad pixels
        for i in range(n_flats):
            if self.do_bias:
                master_bias = individuals[i].master_bias.parse_image()
                flats_cube[i, :, :] -= master_bias
            if self.do_dark:
                master_dark = individuals[i].master_dark.parse_image()
                flats_cube[i, :, :] -= master_dark
            flats_cube[i, :, :] /= pcmaths.weighted_median(flats_cube[i, :, :], percentile=self.flat_percentile)
            bad = np.where((flats_cube[i, :, :] < 0) | (flats_cube[i, :, :] > self.flat_percentile * 100))
            if bad[0].size > 0:
                flats_cube[i, :, :][bad] = np.nan
        
        # Median crunch, flag one more time
        master_flat = np.nanmedian(flats_cube, axis=0)
        bad = np.where((master_flat < 0) | (master_flat > self.flat_percentile * 100))
        if bad[0].size > 0:
            master_flat[bad] = np.nan
            
        # Return
        return master_flat

    def generate_master_dark(self, individuals):
        """Computes a median master flat field image from a subset of flat field images. Dark and bias subtraction is also performed if set.

            Args:
                individuals (list): The list of DarkImages.
                bias_subtraction (bool): Whether or not to perform bias subtraction. If so, each dark must have a master_bias attribute.
            Returns:
                master_dark (np.ndarray): The median combined master dark image
        """
        darks_cube = pcdata.Echellogram.generate_cube(individuals)
        if self.do_bias:
            for i in darks_cube.shape[0]:
                master_bias = individuals[i].master_bias.parse_image()
                darks_cube[i, :, :] -= master_bias
        master_dark = np.nanmedian(darks_cube, axis=0)
        return master_dark

    def generate_master_bias(self, individuals):
        """Generates a median master bias image.

        Args:
            individuals (list): The list of BiasImages.
        Returns:
            master_bias (np.ndarray): The master bias image.
        """
        bias_cube = pcdata.Echellogram.generate_cube(individuals)
        master_bias = np.nanmedian(bias_cube, axis=0)
        return master_bias
        
    ########################################################
    #### STANDARD CALIBRATION METHOD FOR A SINGLE TRACE ####
    ########################################################

    def calibrate(self, data, data_image, trace_map_image, trace_dict):
        
        # Calibrated image
        data_image_out = np.copy(data_image)
        
        # Bias correction
        if self.do_bias:
            master_bias_image = data.master_bias.parse_image()
            data_image_out -= master_bias_image
            
        # Dark correction
        if self.do_dark:
            master_dark_image = data.master_dark.parse_image()
            data_image_out -= master_dark_image
            
        # Flat division
        if self.do_flat:
            good_trace = np.where(trace_map_image == trace_dict["label"])
            bad_trace = np.where(trace_map_image != trace_dict["label"])
            master_flat_image = data.master_flat.parse_image()
            ny, nx = master_flat_image.shape
            master_flat_image[bad_trace] = np.nan
            master_flat_image_smooth = pcmaths.median_filter2d(master_flat_image, width=3, preserve_nans=True)
            blaze_init = np.full(nx, np.nan)
            for x in range(nx):
                good = np.where(np.isfinite(master_flat_image_smooth[:, x]))[0]
                if good.size < 1:
                    master_flat_image[:, x] = np.nan
                    continue
                med_flux = np.nanmedian(master_flat_image_smooth[:, x])
                good = np.where(master_flat_image_smooth[:, x] > 0.75 * med_flux)[0]
                blaze_init[x] = np.nanmedian(master_flat_image_smooth[good, x])
            blaze = pcmaths.poly_filter(blaze_init, width=1021, poly_order=3)
            master_flat_image /= np.outer(np.ones(ny), blaze)
            data_image_out /= master_flat_image
        return data_image_out
    
class FringingPreCalibrator(PreCalibrator):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, do_bias=False, do_dark=False, do_flat=True, flat_percentile=0.5, remove_blaze_from_flat=False, remove_fringing_from_flat=False):
        
        super().__init__(do_bias=do_bias, do_dark=do_dark, do_flat=do_flat, flat_percentile=flat_percentile, remove_blaze_from_flat=remove_blaze_from_flat)
        
        self.remove_fringing_from_flat = remove_fringing_from_flat
        
    ########################################################
    #### STANDARD CALIBRATION METHOD FOR A SINGLE TRACE ####
    ########################################################

    def calibrate(self, data, data_image, trace_map_image, trace_dict):
        
        # dims
        ny, nx = data_image.shape

        # Calibrated image
        data_image_out = np.copy(data_image)

        # Bias correction
        if self.do_bias:
            master_bias_image = data.master_bias.parse_image()
            data_image_out -= master_bias_image

        # Dark correction
        if self.do_dark:
            master_dark_image = data.master_dark.parse_image()
            data_image_out -= master_dark_image

        # Flat division
        if self.do_flat:
            good_trace = np.where(trace_map_image == trace_dict["label"])
            bad_trace = np.where(trace_map_image != trace_dict["label"])
            master_flat_image = data.master_flat.parse_image()
            ny, nx = master_flat_image.shape
            master_flat_image[bad_trace] = np.nan
            master_flat_image_smooth = pcmaths.median_filter2d(master_flat_image, width=3, preserve_nans=True)
            blaze_init = np.full(nx, np.nan)
            for x in range(nx):
                good = np.where(np.isfinite(master_flat_image_smooth[:, x]))[0]
                if good.size < 1:
                    master_flat_image[:, x] = np.nan
                    continue
                med_flux = np.nanmedian(master_flat_image_smooth[:, x])
                good = np.where(master_flat_image_smooth[:, x] > 0.75 * med_flux)[0]
                blaze_init[x] = np.nanmedian(master_flat_image_smooth[good, x])
            blaze = pcmaths.poly_filter(blaze_init, width=1021, poly_order=3)
            fringing_init = blaze_init / blaze
            fringing = pcmaths.poly_filter(fringing_init, width=21, poly_order=3)
            if self.remove_fringing_from_flat:
                master_flat_image /= np.outer(np.ones(ny), fringing)
            if self.remove_blaze_from_flat:
                master_flat_image /= np.outer(np.ones(ny), blaze)
            data_image_out /= master_flat_image
        return data_image_out