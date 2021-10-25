# Python default modules
import os
import copy
import pickle

# Science / Math
import numpy as np
import scipy.interpolate
import scipy.signal
from astropy.io import fits

# Graphics
import matplotlib.pyplot as plt

# Pyreduce
import pyreduce.extract as extract

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
from pychell.reduce.extract import SpectralExtractor

# From optimal

class DecompExtractor(SpectralExtractor):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, mask_left=100, mask_right=100, mask_top=100, mask_bottom=100,
                 remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05,
                 trace_pos_poly_order=4, oversample=4,
                 n_trace_refine_iterations=3, n_extract_iterations=3,
                 badpix_threshold=5,
                 extract_orders=None,
                 extract_aperture=None, lambda_sf=0.5, lambda_sp=0.0, tilt=None, shear=None):

        # Super init
        super().__init__(mask_left=mask_left, mask_right=mask_right, mask_top=mask_top, mask_bottom=mask_bottom, extract_orders=extract_orders)

        # Set params
        self.remove_background = remove_background
        self.background_smooth_poly_order = background_smooth_poly_order
        self.background_smooth_width = background_smooth_width
        self.flux_cutoff = flux_cutoff
        self.n_trace_refine_iterations = n_trace_refine_iterations
        self.n_extract_iterations = n_extract_iterations
        self.trace_pos_poly_order = trace_pos_poly_order
        self.oversample = oversample
        self.badpix_threshold = badpix_threshold
        self._extract_aperture = extract_aperture
        self.lambda_sf = lambda_sf
        self.lambda_sp = lambda_sp
        self.tilt = tilt
        self.shear = shear
        
        
    #######################################################################
    #### PRIMARY METHOD TO EXTRACT SINGLE TRACE FOR ENTIRE ORDER WIDTH ####
    #######################################################################

    def extract_trace(self, data, trace_image, trace_map_image, trace_dict, badpix_mask=None, read_noise=None):
        return self._extract_trace(data, trace_image, trace_map_image, trace_dict, badpix_mask, read_noise, self.tilt, self.shear, self.remove_background, self.background_smooth_poly_order, self.background_smooth_width, self.flux_cutoff, self.trace_pos_poly_order, self.oversample, self.n_trace_refine_iterations, self.n_extract_iterations, self.badpix_threshold, self.extract_orders, self._extract_aperture, self.lambda_sf, self.lambda_sp)

    @staticmethod
    def _extract_trace(data, image, trace_map_image, trace_dict, badpix_mask, read_noise=None, tilt=None, shear=None, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05, trace_pos_poly_order=4, oversample=4, n_trace_refine_iterations=3, n_extract_iterations=3, badpix_threshold=5, extract_orders=None, _extract_aperture=None, lambda_sf=0.5, lambda_sp=0):

        if read_noise is None:
            read_noise = data.parser.parse_itime(data) * data.parser.spec_module.read_noise
        else:
            read_noise = 0

        # Numbers
        nx = image.shape[1]

        if tilt is None:
            tilt = np.zeros(nx)
        if shear is None:
            shear = np.zeros(nx)

        # Don't overwrite image
        trace_image = np.copy(image)

        # Helpful array
        xarr = np.arange(nx)

        # Initiate mask
        if badpix_mask is None:
            badpix_mask = np.ones(trace_map_image.shape)
        else:
            badpix_mask = np.copy(badpix_mask)
        bad = np.where((trace_map_image != trace_dict['label']) | ~np.isfinite(trace_image) | (badpix_mask == 0))
        badpix_mask[bad] = 0
        trace_image[bad] = np.nan

        # Initial trace positions
        trace_positions = np.polyval(trace_dict['pcoeffs'], xarr)

        # Crop the image and mask to limit memory usage going forward
        goody, goodx = np.where(badpix_mask)
        y_start, y_end = np.min(goody), np.max(goody)
        trace_image = trace_image[y_start:y_end + 1, :]
        badpix_mask = badpix_mask[y_start:y_end + 1, :]
        trace_positions -= y_start
            
        # Flag obvious bad pixels
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=5)
        peak = pcmath.weighted_median(trace_image_smooth, percentile=0.99)
        bad = np.where((trace_image < 0) | (trace_image > 10 * peak))
        if bad[0].size > 0:
            trace_image[bad] = np.nan
            badpix_mask[bad] = 0

        # Estimate background
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)
        background = np.full(nx, np.nan)
        for x in range(nx):
            background[x] = pcmath.weighted_median(trace_image_smooth[:, x], percentile=0.1)
        background = pcmath.median_filter1d(background, width=7)
        bad = np.where(background < 0)[0]
        if bad.size > 0:
            trace_image[:, bad] = np.nan
            badpix_mask[:, bad] = 0
            background[bad] = np.nan
        
        # Iteratively refine trace positions and profile.
        for i in range(n_trace_refine_iterations):
            
            # Trace Profile
            print(f" [{data}, {trace_dict['label']}] Iteratively Refining Trace profile [{i + 1} / {n_trace_refine_iterations}] ...", flush=True)
            trace_profile_cspline = DecompExtractor.compute_trace_profile(trace_image, badpix_mask, trace_positions, background, remove_background, oversample)
            
            # Trace Position
            print(f" [{data}, {trace_dict['label']}] Iteratively Refining Trace positions [{i + 1} / {n_trace_refine_iterations}] ...", flush=True)
            trace_positions = DecompExtractor.compute_trace_positions(trace_image, badpix_mask, trace_profile_cspline, trace_positions, background, remove_background, trace_pos_poly_order)

            # Extract Aperture
            if _extract_aperture is None:
                extract_aperture = DecompExtractor.compute_extract_aperture(trace_profile_cspline)
            else:
                extract_aperture = _extract_aperture

            # Background signal
            print(f" [{data}, {trace_dict['label']}] Iteratively Refining Background [{i + 1} / {n_trace_refine_iterations}] ...", flush=True)
            background, background_err = DecompExtractor.compute_background(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, background_smooth_width, background_smooth_poly_order)
        
        # Iteratively extract spectrum
        for i in range(n_extract_iterations):
            
            print(f" [{data}] Iteratively Extracting Trace [{i + 1} / {n_extract_iterations}] ...", flush=True)
            
            # Optimal extraction
            spec1d, spec1d_unc = DecompExtractor.decomp_extraction(trace_image, badpix_mask, trace_positions, extract_aperture, background, remove_background, read_noise, oversample, lambda_sp, lambda_sf, tilt, shear)

            # Re-map pixels and flag in the 2d image.
            if i < n_extract_iterations - 1:
                DecompExtractor.flag_pixels_post_extraction(trace_image, badpix_mask, trace_positions, trace_profile_cspline, extract_aperture, spec1d, spec1d_unc, background, background_err, remove_background, badpix_threshold)

        # badpix mask
        badpix1d = np.ones(nx)
        bad = np.where(~np.isfinite(spec1d) | (spec1d <= 0) | ~np.isfinite(spec1d_unc) | (spec1d_unc <= 0))[0]
        if bad.size > 0:
            spec1d[bad] = np.nan
            spec1d_unc[bad] = np.nan
            badpix1d[bad] = 0
            
        # Data out
        data_out = np.array([spec1d, spec1d_unc, badpix1d], dtype=float).T
        
        return data_out


    ###########################
    #### DECOMP EXTRACTION ####
    ###########################

    @staticmethod
    def decomp_extraction(trace_image, badpix_mask, trace_positions, extract_aperture, background=None, remove_background=True, read_noise=0, oversample=1, lambda_sp=0, lambda_sf=0.5, tilt=None, shear=None):

        # Copy input
        trace_image_cp = np.copy(trace_image)
        trace_positions_cp = np.copy(trace_positions)
        badpix_mask_cp = np.copy(badpix_mask)

        # Dims
        ny, nx = trace_image.shape

        # Helpful array
        yarr = np.arange(ny)

        # Remove background
        if remove_background:
            for x in range(nx):
                trace_image_cp[:, x] -= background[x]

        # Flat negative pixels
        bad = np.where(trace_image_cp < 0)
        if bad[0].size > 0:
            trace_image_cp[bad] = np.nan
            badpix_mask[bad] = np.nan
        
        # Sanity check y positions
        goody, goodx = np.where(badpix_mask_cp)
        xi, xf = goodx.min(), goodx.max()
        for x in range(xi, xf+1):
            if trace_positions_cp[x] - extract_aperture[0] < 0 or trace_positions_cp[x] + extract_aperture[1] > ny - 1:
                return np.full(nx, np.nan), np.full(nx, np.nan)
            good = np.where((yarr >= trace_positions_cp[x] - extract_aperture[0]) & (yarr <= trace_positions_cp[x] + extract_aperture[1]) & (badpix_mask_cp[:, x] == 1))[0]

        # Define additional inputs
        yrange = [int(np.ceil(extract_aperture[0])), int(np.ceil(extract_aperture[1]))]
        goody, goodx = np.where(badpix_mask_cp)
        xrange = [goodx.min(), goodx.max()]
        trace_image_cp = np.ma.masked_array(trace_image_cp, mask=np.logical_not(badpix_mask_cp))

        # Call pyreduce
        result = extract.extract_spectrum(trace_image_cp, trace_positions_cp, yrange=yrange, xrange=xrange, lambda_sf=lambda_sf, lambda_sp=lambda_sp, osample=oversample, readnoise=read_noise, tilt=tilt, shear=shear)

        # Get relevant outputs
        spec1d = result[0]
        spec1d_unc = result[3]
        bad = np.where((spec1d == 0) | (spec1d_unc == 0))[0]
        spec1d[bad] = np.nan
        spec1d_unc[bad] = np.nan
        trace_profile = result[1]

        # Return
        return spec1d, spec1d_unc