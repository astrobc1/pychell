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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath
from pychell.reduce.extract import SpectralExtractor

class OptimalExtractor(SpectralExtractor):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05,
                 trace_pos_poly_order=4, oversample=4,
                 n_trace_refine_iterations=3, trace_pos_refine_window=None, n_extract_iterations=3,
                 badpix_threshold=5,
                 extract_aperture=None, extract_orders=None):

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
        
    #######################################################################
    #### PRIMARY METHOD TO EXTRACT SINGLE TRACE FOR ENTIRE ORDER WIDTH ####
    #######################################################################

    def extract_trace(self, data, trace_image, trace_map_image, trace_dict, badpix_mask=None, read_noise=None):
        return self._extract_trace(data, trace_image, trace_map_image, trace_dict, badpix_mask, read_noise, self.remove_background, self.background_smooth_poly_order, self.background_smooth_width, self.flux_cutoff, self.trace_pos_poly_order, self.oversample, self.n_trace_refine_iterations, self.trace_pos_refine_window, self.n_extract_iterations, self.badpix_threshold, self.extract_orders, self._extract_aperture)

    @staticmethod
    def _extract_trace(data, image, trace_map_image, trace_dict, badpix_mask, read_noise=None, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05, trace_pos_poly_order=4, oversample=4, n_trace_refine_iterations=3, trace_pos_refine_window=None, n_extract_iterations=3, badpix_threshold=5, extract_orders=None, _extract_aperture=None):
        
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

        # Initiate trace_pos_refine_window
        if trace_pos_refine_window is None:
            trace_pos_refine_window = trace_dict['height'] / 2

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
        bad = np.where((trace_image < 0) | (trace_image > 3 * peak))
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
            trace_profile_cspline = OptimalExtractor.compute_trace_profile(trace_image, badpix_mask, trace_positions, background, remove_background, oversample)

            # Extract Aperture
            if _extract_aperture is None:
                extract_aperture = OptimalExtractor.compute_extract_aperture(trace_profile_cspline)
            else:
                extract_aperture = _extract_aperture
            
            # Trace Position
            print(f" [{data}, {trace_dict['label']}] Iteratively Refining Trace positions [{i + 1} / {n_trace_refine_iterations}] ...", flush=True)
            trace_positions = OptimalExtractor.compute_trace_positions(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, trace_pos_refine_window, background, remove_background, trace_pos_poly_order)

            # Background signal
            print(f" [{data}, {trace_dict['label']}] Iteratively Refining Background [{i + 1} / {n_trace_refine_iterations}] ...", flush=True)
            background, background_err = OptimalExtractor.compute_background(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, background_smooth_width, background_smooth_poly_order)
        
        # Iteratively extract spectrum
        for i in range(n_extract_iterations):
            
            print(f" [{data}, {trace_dict['label']}] Iteratively Extracting Trace [{i + 1} / {n_extract_iterations}] ...", flush=True)
            
            # Optimal extraction
            spec1d, spec1d_unc = OptimalExtractor.optimal_extraction(trace_image, badpix_mask,
                                                         trace_profile_cspline, trace_positions, extract_aperture=extract_aperture,
                                                         remove_background=remove_background, background=background,
                                                         background_err=background_err, read_noise=read_noise)

            # Re-map pixels and flag in the 2d image.
            if i < n_extract_iterations - 1:

                # 2d model
                model2d = OptimalExtractor.compute_model2d(trace_image, badpix_mask, spec1d, trace_profile_cspline, trace_positions, extract_aperture, remove_background=remove_background, background=background, background_err=background_err)

                # Flag
                OptimalExtractor.flag_pixels2d(trace_image, badpix_mask, model2d, badpix_threshold)

        # badpix mask
        badpix1d = np.ones(nx)
        bad = np.where(~np.isfinite(spec1d) | (spec1d <= 0) | ~np.isfinite(spec1d_unc) | (spec1d_unc <= 0))[0]
        if bad.size > 0:
            spec1d[bad] = np.nan
            spec1d_unc[bad] = np.nan
            badpix1d[bad] = 0
        
        return spec1d, spec1d_unc, badpix1d


    ############################
    #### OPTIMAL EXTRACTION ####
    ############################
    
    @staticmethod
    def optimal_extraction(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, remove_background=True, background=None, background_err=None, read_noise=0):

        # Image dims
        ny, nx = trace_image.shape

        # Storage arrays
        spec = np.full(nx, fill_value=np.nan, dtype=float)
        spec_unc = np.full(nx, fill_value=np.nan, dtype=float)
        
        # Helper array
        yarr = np.arange(ny)

        # Loop over columns
        for x in range(nx):
            
            # Views
            badpix_x = np.copy(badpix_mask[:, x])
            if remove_background:
                data_x = trace_image[:, x] - background[x]
            else:
                data_x = np.copy(trace_image[:, x])
            
            # Flag negative values after sky subtraction
            bad = np.where(data_x < 0)[0]
            if bad.size > 0:
                badpix_x[bad] = 0
                data_x[bad] = np.nan
                
            # Check if column is worth extracting
            if np.nansum(badpix_x) <= 1:
                continue
            
            # Shift Trace Profile
            P = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x],
                                      trace_profile_cspline(trace_profile_cspline.x),
                                      yarr)
            
            # Determine which pixels to use from the aperture
            good = np.where((yarr >= trace_positions[x] - extract_aperture[0] - 0.5) & (yarr <= trace_positions[x] + extract_aperture[1] + 0.5))[0]
            P_use = P[good]
            data_use = data_x[good]
            badpix_use = badpix_x[good]
            P_use /= np.nansum(P_use)
            
            # Variance
            if remove_background:
                var_use = read_noise**2 + data_use + background[x] + background_err[x]**2
            else:
                var_use = read_noise**2 + data_use
            
            # Weights = bad pixels only
            weights_use = P_use**2 / var_use * badpix_use
            
            # Normalize the weights such that sum=1
            weights_use /= np.nansum(weights_use)
            
            # Final sanity check
            good = np.where(weights_use > 0)[0]
            if good.size <= 1:
                continue
            
            # 1d final flux at column x
            correction = np.nansum(P_use * weights_use)
            spec[x] = np.nansum(data_use * weights_use) / correction
            spec_unc[x] = np.sqrt(np.nansum(var_use)) / correction

        # Return
        return spec, spec_unc
    

    #########################
    #### CREATE 2d MODEL ####
    #########################

    @staticmethod
    def compute_model2d(trace_image, badpix_mask, spec1d, trace_profile_cspline, trace_positions, extract_aperture, remove_background=True, background=None, background_err=None):
        
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

            # Which pixels to use
            usey = np.where((yarr >= trace_positions[i] - extract_aperture[0] - 0.5) & (yarr <= trace_positions[i] + extract_aperture[1] + 0.5))[0]

            # Normalize
            tp = tp[usey] / np.nansum(tp[usey])

            # Model
            if remove_background:
                model2d[usey, i] = tp * spec1d[i] + background[i]
            else:
                model2d[usey, i] = tp * spec1d[i]
        
        # Return
        return model2d
