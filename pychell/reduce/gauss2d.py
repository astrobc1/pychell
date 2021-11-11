# Python default modules
import os
import copy
import pickle

# Science / Math
import numpy as np
import scipy.interpolate
import scipy.signal
from astropy.io import fits
import scipy.sparse.linalg as slinalg

# LLVM
from numba import jit, njit

# Graphics
import matplotlib.pyplot as plt

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
from pychell.reduce.extract import SpectralExtractor

class Gauss2dExtractor(SpectralExtractor):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05,
                 trace_pos_poly_order=4, oversample=4,
                 n_trace_refine_iterations=3, n_extract_iterations=3, trace_pos_refine_window=3,
                 badpix_threshold=5,
                 extract_orders=None,
                 extract_aperture=None,
                 eta=None, theta=None, q=None, sigma=None):

        # Super init
        super().__init__(extract_orders=extract_orders)

        # Set params
        self.remove_background = remove_background
        self.background_smooth_poly_order = background_smooth_poly_order
        self.background_smooth_width = background_smooth_width
        self.flux_cutoff = flux_cutoff
        self.n_trace_refine_iterations = n_trace_refine_iterations
        self.trace_pos_refine_window = trace_pos_refine_window
        self.n_extract_iterations = n_extract_iterations
        self.trace_pos_poly_order = trace_pos_poly_order
        self.oversample = oversample
        self.badpix_threshold = badpix_threshold
        self._extract_aperture = extract_aperture
        self.eta = eta
        self.q = q
        self.sigma = sigma
        self.theta = theta
        
        
    #######################################################################
    #### PRIMARY METHOD TO EXTRACT SINGLE TRACE FOR ENTIRE ORDER WIDTH ####
    #######################################################################

    def extract_trace(self, data, trace_image, trace_map_image, trace_dict, badpix_mask=None, read_noise=None):
        return self._extract_trace(data, trace_image, trace_map_image, trace_dict, badpix_mask, read_noise, self.remove_background, self.background_smooth_poly_order, self.background_smooth_width, self.flux_cutoff, self.trace_pos_poly_order, self.oversample, self.n_trace_refine_iterations, self.n_extract_iterations, self.trace_pos_refine_window, self.badpix_threshold, self.extract_orders, self._extract_aperture, self.eta, self.theta, self.q)

    @staticmethod
    def _extract_trace(data, image, trace_map_image, trace_dict, badpix_mask, read_noise=None, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05, trace_pos_poly_order=4, oversample=4, n_trace_refine_iterations=3, n_extract_iterations=3, trace_pos_refine_window=5, badpix_threshold=5, extract_orders=None, _extract_aperture=None, eta=None, theta=None, q=None):

        if read_noise is None:
            read_noise = data.spec_module.parse_itime(data) * data.spec_module.read_noise
        else:
            read_noise = 0

        # Numbers
        nx = image.shape[1]

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

            trace_profile_cspline = Gauss2dExtractor.compute_trace_profile(trace_image, badpix_mask, trace_positions, background, remove_background, oversample)
            
            # Trace Position
            print(f" [{data}, {trace_dict['label']}] Iteratively Refining Trace positions [{i + 1} / {n_trace_refine_iterations}] ...", flush=True)
            trace_positions = Gauss2dExtractor.compute_trace_positions(trace_image, badpix_mask, trace_profile_cspline, trace_positions, trace_pos_refine_window, background, remove_background, trace_pos_poly_order)

            # Extract Aperture
            if _extract_aperture is None:
                extract_aperture = Gauss2dExtractor.compute_extract_aperture(trace_profile_cspline)
            else:
                extract_aperture = _extract_aperture

            # Background signal
            print(f" [{data}, {trace_dict['label']}] Iteratively Refining Background [{i + 1} / {n_trace_refine_iterations}] ...", flush=True)
            background, background_err = Gauss2dExtractor.compute_background(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, background_smooth_width, background_smooth_poly_order)
        
        # Iteratively extract spectrum
        for i in range(n_extract_iterations):
            
            print(f" [{data}] Iteratively Extracting Trace [{i + 1} / {n_extract_iterations}] ...", flush=True)
            
            # Optimal extraction
            spec1d, spec1d_unc = Gauss2dExtractor.gauss2dextraction(trace_image, badpix_mask, trace_positions, extract_aperture, background, remove_background, read_noise, oversample, eta, theta, q)

            # Re-map pixels and flag in the 2d image.
            if i < n_extract_iterations - 1:
                Gauss2dExtractor.flag_pixels_post_extraction(trace_image, badpix_mask, trace_positions, trace_profile_cspline, extract_aperture, spec1d, spec1d_unc, background, background_err, remove_background, badpix_threshold)

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
    def gauss2dextraction(trace_image, badpix_mask, trace_positions, extract_aperture, background=None, remove_background=True, read_noise=0, oversample=1, eta=None, theta=None, sigma=None, q=None):

        # Copy input
        trace_image_cp = np.copy(trace_image)
        trace_positions_cp = np.copy(trace_positions)
        badpix_mask_cp = np.copy(badpix_mask)

        # Good
        goody, goodx = np.where(badpix_mask)
        xi, xf = goodx[0], goodx[-1]

        # Dims
        ny, nx = trace_image.shape

        # Helpful array
        yarr = np.arange(ny)

        # Remove background
        if remove_background:
            for x in range(nx):
                trace_image_cp[:, x] -= background[x]

        # Profile
        if theta is None:
            theta = np.zeros(nx)
        if np.isscalar(theta):
            theta = np.full(nx, theta)
        if sigma is None:
            sigma = 3.0
        if np.isscalar(sigma):
            sigma = np.full(nx, sigma)
        if q is None:
            q = 1.0
        if np.isscalar(q):
            q = np.full(nx, q)
        
        # Reglurization
        if eta is None:
            eta = 0.01

        # Flag negative pixels
        bad = np.where(trace_image_cp < 0)
        if bad[0].size > 0:
            trace_image_cp[bad] = np.nan
            badpix_mask[bad] = np.nan

        # Now set all bad pixels to zero
        bad = np.where(~np.isfinite(trace_image_cp) | (badpix_mask == 0))
        if bad[0].size > 0:
            trace_image_cp[bad] = 0

        # Now loop over columns and mask regions low in flux
        for x in range(nx):
            bad = np.where((yarr < trace_positions_cp[x] - extract_aperture[0]) | (yarr > trace_positions_cp[x] + extract_aperture[1]))[0]
            if bad.size > 0:
                trace_image_cp[bad, x] = 0

        # Construct H
        H = Gauss2dExtractor.generate_H(np.ones((3, 3)), np.ones((3, 3)), np.array([1, 2, 3]), np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0, 0, 0])) # warmup
        H = Gauss2dExtractor.generate_H(trace_image_cp, badpix_mask_cp, trace_positions_cp, sigma, q, np.arctan(theta))

        # Initial guess
        f0 = np.nansum(trace_image_cp, axis=0)

        # Perform lsqr
        result = slinalg.lsqr(H.reshape((ny*nx, nx)), trace_image_cp.flatten(), damp=eta, x0=f0)

        # Parse results
        breakpoint()
        spec1d = result[0]
        bad = np.where(~np.isfinite(spec1d) | (spec1d <= 0))[0]
        spec1d[bad] = np.nan
        spec1d_unc = np.sqrt(spec1d)
        spec1d[0:xi] = np.nan
        spec1d[xf+1:] = np.nan

        # Return
        return spec1d, spec1d_unc

    @staticmethod
    @jit
    def generate_H(image, badpix_mask, trace_positions, sigma, q, theta):

        # Image dims
        ny, nx = image.shape

        # Apertures
        nm = nx

        # Initialize H tensor (not yet flattened)
        H = np.full(shape=(ny, nx, nm), fill_value=np.nan)

        # Helpful arrays
        xarr_detector = np.arange(nx)
        yarr_detector = np.arange(ny)
        xarr_aperture = np.arange(nx)
        yarr_aperture = np.arange(ny)

        # Loops!
        for i in range(nx):
            for j in range(ny):
                for m in range(nm):
                    xl = xarr_detector[i]
                    xkc = xarr_detector[m]
                    yl = yarr_detector[j]
                    ykc = trace_positions[m]
                    _theta = theta[m]
                    _sigma = sigma[m]
                    _q = q[m]
                    norm = 1 / (2 * np.pi * _q * _sigma**2)
                    xp = (xl - xkc) * np.sin(_theta) - (yl - ykc) * np.cos(_theta)
                    yp = (xl - xkc) * np.cos(_theta) + (yl - ykc) * np.sin(_theta)
                    H[j, i, m] = np.exp(-0.5 * ((xp / _sigma)**2 + (yp / (_q * _sigma))**2)) * badpix_mask[j, i]

        # Normalize each aperture
        for m in range(nm):
            H[:, :, m] /= np.nansum(H[:, :, m])

        return H

