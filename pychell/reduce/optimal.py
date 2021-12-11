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
    
    def __init__(self, remove_background=False, background_smooth_poly_order=3, background_smooth_width=51,
                 trace_pos_poly_order=4, oversample=4, chunk_width=600,
                 n_iterations=3, trace_pos_refine_window=None,
                 badpix_threshold=5,
                 extract_aperture=None, extract_orders=None, refine_trace_positions=True):

        # Super init
        super().__init__(extract_orders=extract_orders)

        # Set params
        self.remove_background = remove_background
        self.background_smooth_poly_order = background_smooth_poly_order
        self.background_smooth_width = background_smooth_width
        self.n_iterations = n_iterations
        self.trace_pos_refine_window = trace_pos_refine_window
        self.trace_pos_poly_order = trace_pos_poly_order
        self.oversample = oversample
        self.chunk_width = chunk_width
        self.badpix_threshold = badpix_threshold
        self._extract_aperture = extract_aperture
        self.refine_trace_positions = refine_trace_positions
        
    #######################################################################
    #### PRIMARY METHOD TO EXTRACT SINGLE TRACE FOR ENTIRE ORDER WIDTH ####
    #######################################################################

    def extract_trace(self, data, trace_image, trace_map_image, trace_dict, badpix_mask=None, read_noise=None):
        return self._extract_trace(data, trace_image, trace_map_image, trace_dict, badpix_mask, read_noise, self.remove_background, self.background_smooth_poly_order, self.background_smooth_width, self.trace_pos_poly_order, self.oversample, self.trace_pos_refine_window, self.n_iterations, self.badpix_threshold, self.extract_orders, self._extract_aperture, self.chunk_width, self.refine_trace_positions)

    @staticmethod
    def _extract_trace(data, image, trace_map_image, trace_dict, badpix_mask, read_noise=None, remove_background=False, background_smooth_poly_order=3, background_smooth_width=51, trace_pos_poly_order=4, oversample=4, trace_pos_refine_window=None, n_iterations=5, badpix_threshold=5, extract_orders=None, _extract_aperture=None, chunk_width=600, refine_trace_positions=True):
        
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

        # Chunks
        chunks = OptimalExtractor.generate_chunks(trace_image, badpix_mask, chunk_width=chunk_width)
        n_chunks = len(chunks)

        # Storage arrays
        spec1d_chunks = np.full((nx, n_chunks), np.nan)
        spec1d_unc_chunks = np.full((nx, n_chunks), np.nan)

        # Iteratively Extract
        for i in range(n_chunks):

            # Chunk info
            xi, xf = chunks[i][0], chunks[i][1]
            goody, _ = np.where(badpix_mask[:, xi:xf+1])
            yi, yf = goody.min(), goody.max()

            # Crop image and mask
            trace_image_chunk = np.copy(trace_image[yi:yf+1, xi:xf+1])
            badpix_mask_chunk = np.copy(badpix_mask[yi:yf+1, xi:xf+1])

            # Starting trace positions for this chunk
            trace_positions_chunk = trace_positions[xi:xf+1] - yi

            # Starting trace profile for this chunk
            trace_profile_cspline = OptimalExtractor.compute_vertical_trace_profile(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, oversample, background=None)

            # Extract Aperture
            if _extract_aperture is None:
                extract_aperture = OptimalExtractor.compute_extract_aperture(trace_profile_cspline)
            else:
                extract_aperture = _extract_aperture

            # Starting background
            if remove_background:
                background = np.nanmin(trace_image_chunk, axis=0)
                background = pcmath.poly_filter(background, width=background_smooth_width, poly_order=background_smooth_poly_order)
                background_err = np.sqrt(background)
            else:
                background = None
                background_err = None

            # Update trace positions for this chunk
            if refine_trace_positions:
                trace_positions_chunk = OptimalExtractor.compute_trace_positions(trace_image_chunk, badpix_mask_chunk, trace_profile_cspline, trace_positions_chunk, extract_aperture, spec1d=None, window=trace_pos_refine_window, background=background, remove_background=remove_background, trace_pos_poly_order=trace_pos_poly_order)

            # Current spec1d
            spec1d_chunk = np.nansum(trace_image_chunk, axis=0)
            spec1d_unc_chunk = np.sqrt(spec1d_chunk)

            for j in range(n_iterations):

                print(f" [{data}] Extracting Trace {trace_dict['label']}, Chunk [{i + 1}/{n_chunks}], Iter [{j + 1}/{n_iterations}] ...", flush=True)

                # Update trace positions
                if refine_trace_positions:
                    trace_positions_chunk = OptimalExtractor.compute_trace_positions(trace_image_chunk, badpix_mask_chunk, trace_profile_cspline, trace_positions_chunk, extract_aperture, spec1d_chunk, trace_pos_refine_window, background, remove_background, trace_pos_poly_order)

                # Update trace profile
                trace_profile_cspline = OptimalExtractor.compute_vertical_trace_profile(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, oversample, spec1d_chunk, background=background)

                # Optimal extraction
                spec1d_chunk, spec1d_unc_chunk, background, background_err = OptimalExtractor.optimal_extraction(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, extract_aperture, trace_profile_cspline=trace_profile_cspline, remove_background=remove_background, background=background, background_err=background_err, read_noise=read_noise)

                # Store results
                spec1d_chunks[xi:xf+1, i] = spec1d_chunk
                spec1d_unc_chunks[xi:xf+1, i] = spec1d_unc_chunk

                # Re-map pixels and flag in the 2d image.
                if i < n_iterations - 1:

                    # 2d model
                    model2d = OptimalExtractor.compute_model2d(trace_image_chunk, badpix_mask_chunk, spec1d_chunk, trace_profile_cspline, trace_positions_chunk, extract_aperture, remove_background=remove_background, background=background, smooth=True)

                    # Flag
                    OptimalExtractor.flag_pixels2d(trace_image_chunk, badpix_mask_chunk, model2d, badpix_threshold)

        # Average chunks
        spec1d = np.nanmean(spec1d_chunks, axis=1)
        spec1d_unc = np.nanmean(spec1d_unc_chunks, axis=1)

        # 1d badpix mask
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
    def optimal_extraction(trace_image, badpix_mask, trace_positions, extract_aperture, trace_profile_cspline=None, remove_background=False, background=None, background_err=None, read_noise=0, background_smooth_poly_order=3, background_smooth_width=51):
        """A flavor of optimal extraction. A single column from the data is a function of y pixels ($S_{y}$), and is modeled as:

            F_{y} = A * P_{y} + B

            where P_{y} is the nominal vertical profile and may be arbitrarily scaled. If P_{y} is unknown, the data itself is used. A is the scaling of the input signal, and B is the background signal which is ignored if remove_background is False or the background variable is set. The sum is performed over a given window defined by extract_aperture. A and B are determined by minimizing the function:

            phi = \sum_{y} w_{y} (S_{y} - F_{y})^{2} = \sum_{y} w_{y} (S_{y} - (A * (\sum_{y} P_{y}) + B))^{2}

            where
            
            w_{y} = P_{y}^{2} / \sigma_{y}^{2}, \sigma_{2} = R^{2} + S_{y} + B/(N − 1), N = number of good pixels used in finding B.

            The final 1d value is then A * \sum_{y} P_{y}.

        Args:
            trace_image (np.ndarray): The trace image of shape ny, nx.
            badpix_mask (np.ndarray): The bad pix mask.
            trace_positions (np.ndarray): The trace positions of length nx.
            extract_aperture (np.ndarray): A list of the number of pixels above and below trace_positions for each column.
            trace_profile_cspline (scipy.interpolate.CubicSpline): A CubicSpline object to construct the trace profile, centered at zero.
            remove_background (bool, optional): Whether or not to remove the background. Defaults to False.
            background (np.ndarray, optional): The background. Defaults to None.
            background_err (np.ndarray, optional): The background error. Defaults to None.
            read_noise (float, optional): The detector read noise. Defaults to 0.
            background_smooth_poly_order (int, optional): The polynomial order to smooth the background with. Defaults to 51.. Defaults to 3.
            background_smooth_width (int, optional): The window size of the polynomial filter to smooth the background with. Defaults to 51.

        Returns:
            np.ndarray: The 1d flux.
            np.ndarray: The 1d flux uncertainty.
            np.ndarray: The 1d background.
            np.ndarray: The 1d background uncertainty.
        """

        # Image dims
        ny, nx = trace_image.shape

        # Copy
        trace_image_cp = np.copy(trace_image)
        badpix_mask_cp = np.copy(badpix_mask)

        # Storage arrays
        spec = np.full(nx, np.nan)
        spec2 = np.full(nx, np.nan)
        spec_err = np.full(nx, np.nan)
        
        # Helper array
        yarr = np.arange(ny)

        # Background
        if remove_background and background is not None:
            trace_image_cp -= np.outer(np.ones(ny), background)
            bad = np.where(trace_image_cp < 0)
            trace_image_cp[bad] = np.nan
            badpix_mask_cp[bad] = 0

        # Loop over columns and compute background and 1d spectrum
        if remove_background and background is None:
            background = np.full(nx, np.nan)
            for x in range(nx):

                # Copy arrs
                S_x = np.copy(trace_image_cp[:, x])
                M_x = np.copy(badpix_mask_cp[:, x])

                # Check if column is worth extracting
                if np.nansum(M_x) <= 1:
                    continue
                
                # Shift Trace Profile
                if trace_profile_cspline is not None:
                    P_x = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x], trace_profile_cspline(trace_profile_cspline.x), yarr)
                else:
                    P_x = np.copy(S_x)
                
                # Determine which pixels to use from the aperture
                #use = np.where((yarr >= trace_positions[x] - extract_aperture[0]) & (yarr <= trace_positions[x] + extract_aperture[1]))[0]
                use = np.where(M_x)[0]

                # Copy arrays
                S = np.copy(S_x[use])
                M = np.copy(M_x[use])
                P = np.copy(P_x[use])

                # Variance
                v = read_noise**2 + S

                # Normalize P
                P /= np.nansum(P) # not necessary but oh well

                # Weights
                w = P**2 * M / v

                # Least squares
                B = (np.nansum(w * S) - np.nansum(w * P) * np.nansum(w * P * S) / np.nansum(w * P**2)) / (np.nansum(w) - np.nansum(w * P)**2 / np.nansum(w * P**2))
                background[x] = B

            # Smooth background
            background, background_err = OptimalExtractor._compute_background1d(background, badpix_mask, extract_aperture, background_smooth_poly_order=background_smooth_poly_order, background_smooth_width=background_smooth_width)

        # Redo the optimal extraction with smoothing the smoothed background
        for x in range(nx):

            # Copy arrs
            S_x = np.copy(trace_image_cp[:, x])
            M_x = np.copy(badpix_mask_cp[:, x])

            # Flag negative vals
            bad = np.where(S_x < 0)[0]
            S_x[bad] = np.nan
            M_x[bad] = 0
                
            # Check if column is worth extracting
            if np.nansum(M_x) <= 1:
                continue

            # Shift Trace Profile
            if trace_profile_cspline is not None:
                P_x = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x], trace_profile_cspline(trace_profile_cspline.x), yarr)
            else:
                P_x = np.copy(S_x)
            
            # Determine which pixels to use from the aperture
            use = np.where((yarr >= trace_positions[x] - extract_aperture[0]) & (yarr <= trace_positions[x] + extract_aperture[1]))[0]

            # Copy arrays
            S = np.copy(S_x[use])
            M = np.copy(M_x[use])
            P = np.copy(P_x[use])
            P /= np.nansum(P) # not necessary but oh well

            # Variance
            if remove_background:
                v = read_noise**2 + S + background_err[x]**2
            else:
                v = read_noise**2 + S
            
            # Weights
            w = P**2 * M / v
            w /= np.nansum(w)

            # Least squares
            A = np.nansum(w * P * S) / np.nansum(w * P**2)

            # Final 1d spec
            spec[x] = A * np.nansum(P)
            spec_err[x] = np.sqrt(np.nansum(v) / (np.nansum(M) - 1))
        
        return spec, spec_err, background, background_err

    @staticmethod
    def _compute_background1d(background_opt, badpix_mask, extract_aperture, background_smooth_poly_order=3, background_smooth_width=51):

        # Bad
        bad = np.where(background_opt < 0)[0]
        background_opt[bad] = 0

        # Smooth the background
        background = pcmath.poly_filter(background_opt, width=background_smooth_width, poly_order=background_smooth_poly_order)
        bad = np.where(background < 0)[0]
        background[bad] = 0
        background_err = np.sqrt(background / np.nansum(badpix_mask, axis=0))

        # Return
        return background, background_err

    @staticmethod
    def compute_trace_positions(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, spec1d=None, window=10, background=None, remove_background=True, trace_pos_poly_order=4):

        # The image dimensions
        ny, nx = trace_image.shape
        
        # Helpful arrays
        yarr = np.arange(ny)
        xarr = np.arange(nx)
        
        # Remove background
        if remove_background:
            trace_image_no_background = trace_image - np.outer(np.ones(ny), background)
        else:
            trace_image_no_background = np.copy(trace_image)

        # Smooth image
        trace_image_no_background_smooth = pcmath.median_filter2d(trace_image_no_background, width=3, preserve_nans=False)
        
        # Trace profile
        trace_profile = trace_profile_cspline(trace_profile_cspline.x)
        
        # Smooth the image
        bad = np.where(trace_image_no_background_smooth <= 0)
        if bad[0].size > 0:
            trace_image_no_background[bad] = np.nan

        # Cross correlate each data column with the trace profile estimate
        y_positions_xc = np.full(nx, np.nan)
        
        # 1d spec info
        if spec1d is None:
            spec1d = np.nansum(trace_image, axis=0)
            spec1d = pcmath.median_filter1d(spec1d, width=3)
        spec1d_norm = spec1d / pcmath.weighted_median(spec1d, percentile=0.95)

        # Normalize trace profile to 1
        trace_profile = trace_profile_cspline(trace_profile_cspline.x)
        trace_profile /= np.nanmax(trace_profile)

        # Loop over columns
        for x in range(nx):
            
            # See if column is even worth looking at
            good = np.where((badpix_mask[:, x] == 1) & np.isfinite(trace_image_no_background_smooth[:, x]))[0]
            if good.size <= 3 or spec1d_norm[x] < 0.2:
                continue
            
            # Normalize data column to 1
            data_x = trace_image_no_background_smooth[:, x] / np.nanmax(trace_image_no_background_smooth[:, x])
            
            # CCF lags
            lags = np.arange(trace_positions[x] - window/2, trace_positions[x] + window/2)
            
            # Perform CCF
            ccf = pcmath.cross_correlate(yarr, data_x, trace_profile_cspline.x, trace_profile, lags, kind="xc")

            # Normalize to max=1
            ccf /= np.nanmax(ccf)
            
            # Fit ccf
            iymax = np.nanargmax(ccf)
            if iymax >= len(lags) - 2 or iymax <= 2:
                continue

            try:
                pfit = np.polyfit(lags[iymax-2:iymax+3], ccf[iymax-2:iymax+3], 2)
            except:
                continue

            # Store the nominal location
            y_positions_xc[x] = -1 * pfit[1] / (2 * pfit[0])
        
        # Smooth the deviations
        y_positions_xc_smooth = pcmath.median_filter1d(y_positions_xc, width=3)
        bad = np.where(np.abs(y_positions_xc - y_positions_xc_smooth) > 0.5)[0]
        if bad.size > 0:
            y_positions_xc[bad] = np.nan
        good = np.where(np.isfinite(y_positions_xc))[0]
        
        # Fit with a polynomial and generate
        if good.size > nx / 6:
            pfit = np.polyfit(xarr[good], y_positions_xc[good], trace_pos_poly_order)
            trace_positions = np.polyval(pfit, xarr)
        
        # Return
        return trace_positions

    #########################
    #### CREATE 2d MODEL ####
    #########################

    @staticmethod
    def compute_model2d(trace_image, badpix_mask, spec1d, trace_profile_cspline, trace_positions, extract_aperture, remove_background=False, background=None, smooth=False):
        
        # Dims
        ny, nx = trace_image.shape

        # Helpful arr
        yarr = np.arange(ny)

        # Initialize model
        model2d = np.full((ny, nx), np.nan)

        # Which 1d spec to use
        if smooth:
            spec1d_smooth = pcmath.median_filter1d(spec1d, width=3)
        else:
            spec1d_smooth = np.copy(spec1d)

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
                model2d[usey, i] = tp * spec1d_smooth[i] + background[i]
            else:
                model2d[usey, i] = tp * spec1d_smooth[i]
        
        # Return
        return model2d

    @staticmethod
    def generate_chunks(trace_image, badpix_mask, chunk_width=200):

        # Preliminary info
        goody, goodx = np.where(badpix_mask)
        xi, xf = goodx.min(), goodx.max()
        nnx = xf - xi + 1
        yi, yf = goody.min(), goody.max()
        nny = yf - yi + 1
        chunk_width = np.max([chunk_width, 200])
        chunks = []
        chunks.append((xi, xi + chunk_width))
        for i in range(1, int(2 * np.ceil(nnx / chunk_width))):
            vi = chunks[i-1][1] - int(chunk_width / 2)
            vf = np.min([vi + chunk_width, xf])
            chunks.append((vi, vf))
            if vf == xf:
                if vf - vi < chunk_width / 2:
                   del chunks[-1]
                   chunks[-1][-1] = xf
                break
        return chunks