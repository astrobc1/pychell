# Python default modules
import os
import copy
import pickle

# Science / Math
import numpy as np
import scipy.interpolate
import scipy.signal
from astropy.io import fits

# Pyreduce
import pyreduce.extract

# Graphics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
from pychell.reduce.extract import SpectralExtractor

# From optimal
class DecompExtractor(SpectralExtractor):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05,
                 trace_pos_poly_order=4, oversample=4,
                 n_trace_refine_iterations=3, n_extract_iterations=3, trace_pos_refine_window=None,
                 badpix_threshold=5,
                 extract_orders=None,
                 chunk_width=400,
                 extract_aperture=None, lambda_sf=0.5, lambda_sp=0.0, tilt=None, shear=None):

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
        self.chunk_width = chunk_width
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
        ny, nx = trace_image.shape
        tilt = self.tilt[:, int(trace_dict['label'])-1] if self.tilt is not None else np.zeros(nx)
        shear = self.shear[:, int(trace_dict['label'])-1] if self.shear is not None else np.zeros(nx)
        return self._extract_trace(data, trace_image, trace_map_image, trace_dict, badpix_mask, read_noise, tilt, shear, self.remove_background, self.background_smooth_poly_order, self.background_smooth_width, self.flux_cutoff, self.trace_pos_poly_order, self.oversample, self.n_trace_refine_iterations, self.trace_pos_refine_window, self.n_extract_iterations, self.badpix_threshold, self.extract_orders, self._extract_aperture, self.lambda_sf, self.lambda_sp)

    @staticmethod
    def _extract_trace(data, image, trace_map_image, trace_dict, badpix_mask, read_noise=None, tilt=None, shear=None, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05, trace_pos_poly_order=4, oversample=4, n_trace_refine_iterations=3, trace_pos_refine_window=None, n_extract_iterations=3, badpix_threshold=5, extract_orders=None, _extract_aperture=None, lambda_sf=0.5, lambda_sp=0):

        if read_noise is None:
            read_noise = data.spec_module.parse_itime(data) * data.spec_module.read_noise
        else:
            read_noise = 0

        # dims
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

        # Initiate trace_pos_refine_window
        if trace_pos_refine_window is None:
            trace_pos_refine_window = trace_dict['height'] / 2

        # Initial trace positions
        trace_positions = np.polyval(trace_dict['pcoeffs'], xarr)

        # Crop the image
        goody, goodx = np.where(badpix_mask)
        yi, yf = np.min(goody), np.max(goody)
        trace_image = trace_image[yi:yf + 1, :]
        badpix_mask = badpix_mask[yi:yf + 1, :]
        trace_positions -= yi

        # Flag obvious bad pixels
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=5)
        peak = pcmath.weighted_median(trace_image_smooth, percentile=0.99)
        bad = np.where((trace_image < 0) | (trace_image > 20 * peak))
        if bad[0].size > 0:
            trace_image[bad] = np.nan
            badpix_mask[bad] = 0

        # Starting background
        if remove_background:
            background = np.nanmin(trace_image, axis=0)
            background = pcmath.poly_filter(background, width=background_smooth_width, poly_order=background_smooth_poly_order)
            background_err = np.sqrt(background)
        else:
            background = None
            background_err = None

        # Extract Aperture
        if _extract_aperture is None:
            extract_aperture = DecompExtractor.compute_extract_aperture(trace_profile_cspline)
        else:
            extract_aperture = _extract_aperture
        
        # Iteratively refine trace positions.
        print(f" [{data}, {trace_dict['label']}] Iteratively Refining Trace positions ...", flush=True)
        for i in range(10):

            # Update trace profile
            trace_profile_cspline = DecompExtractor.compute_vertical_trace_profile(trace_image, badpix_mask, trace_positions, 4, None, background=background)

            # Update trace positions
            trace_positions = DecompExtractor.compute_trace_positions_ccf(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, spec1d=None, window=trace_pos_refine_window, background=background, remove_background=remove_background, trace_pos_poly_order=trace_pos_poly_order)

        # Iteratively extract spectrum
        for i in range(n_extract_iterations):
            
            print(f" [{data}] Iteratively Extracting Trace [{i + 1} / {n_extract_iterations}] ...", flush=True)
            
            # Optimal extraction
            spec1d, spec1d_unc = DecompExtractor.decomp_extraction(trace_image, badpix_mask, trace_positions, extract_aperture, background, remove_background, read_noise, oversample, lambda_sp, lambda_sf, tilt, shear)

            # Re-map pixels and flag in the 2d image.
            if i < n_extract_iterations - 1:

                # 2d model
                model2d = DecompExtractor.compute_model2d(trace_image, badpix_mask, spec1d, trace_positions, extract_aperture, remove_background, background, background_err, tilt, shear)

                # Flag
                DecompExtractor.flag_pixels2d(trace_image, badpix_mask, spec1d, badpix_threshold)

        # badpix mask
        badpix1d = np.ones(nx)
        bad = np.where(~np.isfinite(spec1d) | (spec1d <= 0) | ~np.isfinite(spec1d_unc) | (spec1d_unc <= 0))[0]
        if bad.size > 0:
            spec1d[bad] = np.nan
            spec1d_unc[bad] = np.nan
            badpix1d[bad] = 0
        
        return spec1d, spec1d_unc, badpix1d


    ###########################
    #### DECOMP EXTRACTION ####
    ###########################

    @staticmethod
    def decomp_extraction(trace_image, badpix_mask, trace_positions, extract_aperture, background=None, remove_background=True, read_noise=0, oversample=1, lambda_sp=0, lambda_sf=0.5, tilt=None, shear=None, return_trace_profile=False):

        # Copy input
        trace_image_cp = np.copy(trace_image)
        trace_positions_cp = np.copy(trace_positions)
        badpix_mask_cp = np.copy(badpix_mask)

        # Dims
        ny, nx = trace_image.shape

        # Helpful array
        yarr = np.arange(ny)

        # Aperture
        yrange = [int(np.ceil(extract_aperture[0])), int(np.ceil(extract_aperture[1]))]

        # Remove background
        if remove_background:
            for x in range(nx):
                trace_image_cp[:, x] -= background[x]

        # Flag negative pixels after background subtraction
        bad = np.where(trace_image_cp <= 0)
        if bad[0].size > 0:
            trace_image_cp[bad] = 0

        # Now change all bad pixels to zeros
        bad = np.where(badpix_mask == 0)
        if bad[0].size > 0:
            trace_image_cp[bad] = 0
        
        # Loop and identify bad columns (pyreduce doesn't like bad cols)
        # Note that with a tilted PSF we can't just remove bad cols and join non-neighboring columns that weren't already neighbors
        goody, goodx = np.where(badpix_mask_cp)
        xxi, xxf = goodx.min(), goodx.max()
        yyi, yyf = goody.min(), goody.max()
        nnx = xxf - xxi + 1
        spec1d = np.full(nx, np.nan)
        spec1d_unc = np.full(nx, np.nan)
        xrange = [0, nnx-1]
        S = trace_image_cp[yyi:yyf+1, xxi:xxf+1]
        M = np.logical_not(badpix_mask_cp[yyi:yyf+1, xxi:xxf+1])
        S = np.ma.masked_array(S, mask=M)
        ycen = trace_positions_cp[xxi:xxf+1] - yyi
        tilt = tilt[xxi:xxf+1]
        shear = shear[xxi:xxf+1]
        swath_width = np.min([int(nnx / 4), 400])
        snr = DecompExtractor.estimate_snr(trace_image)
        # With oversample=32, (snr, lambda_sp): (22, 0.01), (60, 0.0001)
        #if lambda_sp == "auto":
        #    lambda_sp = (8.06666667 / snr)**4.59001346
        result = pyreduce.extract.extract_spectrum(S, ycen, yrange=yrange, xrange=np.copy(xrange), lambda_sf=lambda_sf, lambda_sp=lambda_sp, osample=oversample, readnoise=read_noise, tilt=tilt, shear=shear, swath_width=swath_width)
        spec1d[xxi:xxf+1] = result[0]
        spec1d_unc[xxi:xxf+1] = result[3]
        trace_profile = result[1]

        # Flag zeros
        bad = np.where((spec1d <= 0) | (spec1d_unc <= 0))[0]
        if bad.size > 0:
            spec1d[bad] = np.nan
            spec1d_unc[bad] = np.nan

        # Return
        if return_trace_profile:
            return spec1d, spec1d_unc, trace_profile
        else:
            return spec1d, spec1d_unc

    #########################
    #### CREATE 2d MODEL ####
    #########################

    @staticmethod
    def compute_model2d(trace_image, badpix_mask, spec1d, trace_positions, remove_background=True, background=None, background_err=None):


        pyreduce.extract.model_image(trace_image, trace_positions, tilt, shear)
        
        # Dims
        ny, nx = trace_image.shape

        # Helpful arr
        yarr = np.arange(ny)

        # Initialize model
        model2d = np.full((ny, nx), np.nan)

        # Loop over cols
        for i in range(nx):

            # Compute trace profile for this column
            tp = scipy.interpolate.CubicSpline(trace_profile_cspline.x + trace_positions[i], trace_profile_cspline(trace_profile_cspline.x))(yarr)

            usey = np.where((yarr >= trace_positions[x] - extract_aperture[0]) & (yarr <= trace_positions[x] + extract_aperture[1]))[0]

            # Normalize
            tp = tp[usey] / np.nansum(tp[usey])

            # Model
            if remove_background:
                model2d[usey, i] = tp * spec1d[i] + background[i]
            else:
                model2d[usey, i] = tp * spec1d[i]
        
        # Return
        return model2d