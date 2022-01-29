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
    
    def __init__(self, extract_orders=None, chunk_width=None, chunk_overlap=0.5, oversample=1,
                 trace_pos_poly_order=2, trace_pos_refine_window=9,
                 remove_background=False, background_smooth_poly_order=None, background_smooth_width=None,
                 n_iterations=20, badpix_threshold=5,
                 min_profile_flux=1E-3, extract_aperture=None):
        """Constructor for the Optimal Extractor object.

        Args:
            extract_orders (list, optional): Which orders to extract. Defaults to all orders.
            chunk_width (int, optional): The width of a chunk. Defaults to None.
            chunk_overlap (int, optional): The fraction which chunks overlap. Defaults to 0.5.
            oversample (int, optional): The oversampling factor for the trace profile. Defaults to 1.
            trace_pos_poly_order (int, optional): The polynomial order for the trace positions. Defaults to 2.
            trace_pos_refine_window (float, optional): How many total pixels to consider above/below the estimated trace positions when refining it. Defaults to 10 (5 above, 5 below).
            remove_background (bool, optional): Whether or not to remove a background signal before extraction. Defaults to False.
            background_smooth_width (int, optional): How many pixels to use to smooth the background with a rolling median filter. Defaults to None (no smoothing).
            background_smooth_poly_order (int, optional): The order of the rolling polynomial filter for the background. Defaults to None (no smoothing).
            n_iterations (int, optional): How many iterations to refine the trace positions, trace profile, and flag bad pixels. Defaults to 20.
            badpix_threshold (int, optional): Deviations larger than badpix_threshold * stddev(residuals) are flagged. Defaults to 4.
            min_profile_flux (float, optional): The minimum flux (relative to 1) to consider in the trace profile. Defaults to 1E-3.
            extract_aperture (list): The number of pixels [below, above] the trace (relative to trace_positions) to consider for extraction of the desired signal.
        """

        # Super init
        super().__init__(extract_orders=extract_orders)

        # Chunks
        self.chunk_width = chunk_width
        self.chunk_overlap = chunk_overlap

        # Oversample trace profile
        self.oversample = oversample

        # Background
        self.remove_background = remove_background
        self.background_smooth_poly_order = background_smooth_poly_order
        self.background_smooth_width = background_smooth_width

        # Aperture
        self._extract_aperture = extract_aperture
        self.min_profile_flux = min_profile_flux

        # Trace pos
        self.trace_pos_poly_order = trace_pos_poly_order
        self.trace_pos_refine_window = trace_pos_refine_window
        
        # Number of iterations
        self.n_iterations = n_iterations

        # Bad pix flagging
        self.badpix_threshold = badpix_threshold
        
    #################################################################
    #### PRIMARY METHOD TO EXTRACT SINGLE TRACE FOR ENTIRE ORDER ####
    #################################################################

    def extract_trace(self, data, trace_image,
                      trace_map_image, trace_dict,
                      badpix_mask=None):
        return self._extract_trace(data, trace_image, trace_map_image, trace_dict, badpix_mask,
                                   self.chunk_width, self.chunk_overlap, self.oversample, self.trace_pos_poly_order, self.trace_pos_refine_window,
                                   self.remove_background, self.background_smooth_poly_order, self.background_smooth_width,
                                   self.n_iterations, self.badpix_threshold, self.min_profile_flux, self._extract_aperture)

    @staticmethod
    def _extract_trace(data, image, trace_map_image, trace_dict, badpix_mask,
                       chunk_width=None, chunk_overlap=None, oversample=1, trace_pos_poly_order=4, trace_pos_refine_window=None,
                       remove_background=False, background_smooth_poly_order=3, background_smooth_width=51,
                       n_iterations=5, badpix_threshold=5, min_profile_flux=1E-3, _extract_aperture=None):
        
        # Read noise
        read_noise = data.spec_module.parse_itime(data) * data.spec_module.read_noise

        # dims
        nx = image.shape[1]

        # Chunk width
        if chunk_width is None:
            chunk_width = nx

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

        # Chunks (each chunk overlaps 50% with both neighboring chunks)
        chunks = OptimalExtractor.generate_chunks(trace_image, badpix_mask, chunk_width=chunk_width, chunk_overlap=chunk_overlap)
        n_chunks = len(chunks)

        # Storage arrays
        spec1d_chunks = np.full((nx, n_chunks), np.nan)
        spec1d_unc_chunks = np.full((nx, n_chunks), np.nan)

        # Extract chunk by chunk
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

            # Initial background
            if remove_background:
                trace_image_chunk_smooth = pcmath.median_filter2d(trace_image_chunk, width=3)
                background = np.nanmin(trace_image_chunk_smooth, axis=0)
                background = pcmath.median_filter1d(background, width=3)
            else:
                background = None
                background_err = None

            # Starting trace profile for this chunk
            trace_profile_cspline = OptimalExtractor.compute_vertical_trace_profile(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, oversample, remove_background=remove_background, background=background)

            # Extract Aperture
            if _extract_aperture is None:
                extract_aperture = OptimalExtractor.compute_extract_aperture(trace_profile_cspline, min_profile_flux)
            else:
                extract_aperture = _extract_aperture

            # Current spec1d (smoothed)
            spec1d_chunk = np.nansum(trace_image_chunk, axis=0)
            spec1d_chunk = pcmath.median_filter1d(spec1d_chunk, width=3)
            spec1d_unc_chunk = np.sqrt(spec1d_chunk)

            # Initial loop
            for j in range(3):

                # Update trace positions
                trace_positions_chunk = OptimalExtractor.compute_trace_positions_ccf(trace_image_chunk, badpix_mask_chunk, trace_profile_cspline, extract_aperture, trace_positions_estimate=trace_positions_chunk, spec1d=None, ccf_window=trace_pos_refine_window, background=background, remove_background=remove_background, trace_pos_poly_order=trace_pos_poly_order)

                # Update trace profile
                trace_profile_cspline = OptimalExtractor.compute_vertical_trace_profile(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, oversample, None, remove_background=remove_background, background=background)

                # Extract Aperture
                if _extract_aperture is None:
                    extract_aperture = OptimalExtractor.compute_extract_aperture(trace_profile_cspline, min_profile_flux)
                else:
                    extract_aperture = _extract_aperture

                # Background
                background, background_err = OptimalExtractor.compute_background_1d(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, extract_aperture, background_smooth_width=background_smooth_width, background_smooth_poly_order=background_smooth_poly_order)

            # Initial spectrum
            spec1d_chunk, spec1d_unc_chunk = OptimalExtractor.optimal_extraction(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, trace_profile_cspline, extract_aperture, remove_background=remove_background, background=background, background_err=background_err, read_noise=read_noise, n_iterations=1, spec1d0=spec1d_chunk)

            # Main loop
            for j in range(n_iterations):

                print(f" [{data}] Extracting Trace {trace_dict['label']}, Chunk [{i + 1}/{n_chunks}], Iter [{j + 1}/{n_iterations}] ...", flush=True)

                # Update trace positions
                if j < n_iterations / 5:
                    trace_positions_chunk = OptimalExtractor.compute_trace_positions_ccf(trace_image_chunk, badpix_mask_chunk, trace_profile_cspline, extract_aperture, trace_positions_estimate=trace_positions_chunk, spec1d=spec1d_chunk, ccf_window=trace_pos_refine_window, background=background, remove_background=remove_background, trace_pos_poly_order=trace_pos_poly_order)

                # Update trace profile
                if j < n_iterations / 5:
                    trace_profile_cspline = OptimalExtractor.compute_vertical_trace_profile(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, oversample, spec1d_chunk, remove_background=remove_background, background=background)

                # Extract Aperture
                if _extract_aperture is None:
                    extract_aperture = OptimalExtractor.compute_extract_aperture(trace_profile_cspline, min_profile_flux)
                else:
                    extract_aperture = _extract_aperture

                # Background
                if remove_background and j < n_iterations / 5:
                    background, background_err = OptimalExtractor.compute_background_1d(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, extract_aperture, background_smooth_width=background_smooth_width, background_smooth_poly_order=background_smooth_poly_order)

                # Optimal extraction
                spec1d_chunk, spec1d_unc_chunk = OptimalExtractor.optimal_extraction(trace_image_chunk, badpix_mask_chunk, trace_positions_chunk, trace_profile_cspline, extract_aperture, remove_background=remove_background, background=background, background_err=background_err, read_noise=read_noise, n_iterations=1, spec1d0=spec1d_chunk)

                # Store results
                spec1d_chunks[xi:xf+1, i] = spec1d_chunk
                spec1d_unc_chunks[xi:xf+1, i] = spec1d_unc_chunk

                # Re-map pixels and flag in the 2d image.
                if j < n_iterations - 1:

                    # 2d model
                    model2d = OptimalExtractor.compute_model2d(trace_image_chunk, badpix_mask_chunk, spec1d_chunk, trace_profile_cspline, trace_positions_chunk, extract_aperture, remove_background=remove_background, background=background, smooth=True)

                    # Flag
                    trace_image_chunk_new, badpix_mask_chunk_new = OptimalExtractor.flag_pixels2d(trace_image_chunk, badpix_mask_chunk, model2d, badpix_threshold)
                    
                    # Break if nothing new is flagged
                    if np.all(badpix_mask_chunk_new == badpix_mask_chunk) and j + 1 >= n_iterations / 10:
                        trace_image_chunk = trace_image_chunk_new
                        badpix_mask_chunk = badpix_mask_chunk_new
                        break
                    else:
                        trace_image_chunk = trace_image_chunk_new
                        badpix_mask_chunk = badpix_mask_chunk_new

        # Average chunks
        spec1d = OptimalExtractor.combine_chunks(spec1d_chunks, chunks)
        spec1d_unc = OptimalExtractor.combine_chunks(spec1d_unc_chunks, chunks)

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
    def optimal_extraction(trace_image, badpix_mask, trace_positions, trace_profile_cspline, extract_aperture, remove_background=False, background=None, background_err=None, read_noise=0, n_iterations=1, spec1d0=None):
        """Standard optimal extraction. A single column from the data is a function of y pixels ($S_{y}$), and is modeled as:

            F_{y} = A * P_{y}

            where P_{y} is the nominal vertical profile and may be arbitrarily scaled. The parameter $A$ is the scaling of the input signal and is fit for in the least squares sense by minimizing the function:

            phi = \sum_{y} w_{y} (S_{y} - F_{y})^{2} = \sum_{y} w_{y} (S_{y} - A P_{y})^{2}

            where
            
            w_{y} = P_{y}^{2} M_{y} / \sigma_{y}^{2}, \sigma_{2} = R^{2} + S_{y} + B/(N − 1), N = number of good pixels used in finding the background, B, and M_{y} is a binary mask (1=good, 0=bad).

            The final 1d value is then A \sum_{y} P_{y}.

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
            spec1d0 (np.ndarray, optional): The intial 1d spectrum, which will used to determine the initial variance weights if provided.
            n_iterations (int, optional): The number of iterations. Defaults to 1.

        Returns:
            np.ndarray: The 1d flux.
            np.ndarray: The 1d flux uncertainty.
        """

        # Image dims
        ny, nx = trace_image.shape

        # Copy
        trace_image_cp = np.copy(trace_image)
        badpix_mask_cp = np.copy(badpix_mask)
        
        # Helper array
        yarr = np.arange(ny)

        # Background
        if remove_background:
            trace_image_cp -= np.outer(np.ones(ny), background)
            bad = np.where(trace_image_cp < 0)
            trace_image_cp[bad] = np.nan
            badpix_mask_cp[bad] = 0

        # Storage arrays
        spec1d = np.full(nx, np.nan)
        spec1d_err = np.full(nx, np.nan)

        # Loop over iterations
        for i in range(n_iterations):

            # Loop over cols
            for x in range(nx):

                # Copy arrs
                S_x = np.copy(trace_image_cp[:, x])
                M_x = np.copy(badpix_mask_cp[:, x])
                    
                # Check if column is worth extracting
                if np.nansum(M_x) <= 1:
                    continue

                # Shift Trace Profile
                P_x = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x], trace_profile_cspline(trace_profile_cspline.x), yarr)
                
                # Determine which pixels to use from the aperture
                use = np.where((yarr >= trace_positions[x] - extract_aperture[0]) & (yarr <= trace_positions[x] + extract_aperture[1]))[0]

                # Copy arrays
                S = np.copy(S_x[use])
                M = np.copy(M_x[use])
                P = np.copy(P_x[use])
                P /= np.nansum(P) # Not necessary

                # Variance
                if remove_background:
                    if i == 0 and spec1d0 is None:
                        v = read_noise**2 + S + background_err[x]**2
                    elif i == 0 and np.isfinite(spec1d0[x]):
                        v = read_noise**2 + spec1d0[x] * P + background[x] + background_err[x]**2
                    else:
                        v = read_noise**2 + spec1d[x] * P + background[x] + background_err[x]**2
                else:
                    if i == 0 and spec1d0 is None:
                        v = read_noise**2 + S
                    elif i == 0 and np.isfinite(spec1d0[x]):
                        v = read_noise**2 + spec1d0[x] * P
                    else:
                        v = read_noise**2 + spec1d[x] * P

                # Weights
                w = P**2 * M / v
                w /= np.nansum(w)

                # Least squares
                A = np.nansum(w * P * S) / np.nansum(w * P**2)

                # Final 1d spec
                spec1d[x] = A * np.nansum(P)
                spec1d_err[x] = np.sqrt(np.nansum(v) / (np.nansum(M) - 1)) / np.nansum(w * P)

        return spec1d, spec1d_err

    #########################
    #### CREATE 2d MODEL ####
    #########################

    @staticmethod
    def compute_model2d(trace_image, badpix_mask, spec1d, trace_profile_cspline, trace_positions, extract_aperture, remove_background=False, background=None, smooth=False):
        """Generates the nominal 2d model from the 1d spectrum.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            badpix_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
            spec1d (np.ndarray): The current 1d spectrum.
            trace_profile_cspline (scipy.interpolate.CubicSpline): A CubicSpline object used to create the trace profile (grid is relative to zero).
            trace_positions (np.ndarray): The trace positions.
            extract_aperture (list): The number of pixels [below, above] the trace (relative to trace_positions) to consider for extraction of the desired signal.
            remove_background (bool, optional): Whether or not to remove a background signal. Defaults to False.
            background (np.ndarray, optional): The background signal. Defaults to None.
            smooth (bool, optional): Whether or not to smooth the 1d spectrum before re-convolving. Defaults to False.

        Returns:
            np.ndarray: The 2d model.
        """
        
        # Dims
        ny, nx = trace_image.shape

        # Helpful arr
        yarr = np.arange(ny)

        # Initialize model
        model2d = np.full((ny, nx), np.nan)

        # Smooth
        spec1d_smooth = pcmath.median_filter1d(spec1d, width=3)

        # Loop over cols
        for i in range(nx):

            # Compute trace profile for this column
            tp = scipy.interpolate.CubicSpline(trace_profile_cspline.x + trace_positions[i], trace_profile_cspline(trace_profile_cspline.x))(yarr)

            # Which pixels to use
            usey = np.where((yarr >= trace_positions[i] - extract_aperture[0]) & (yarr <= trace_positions[i] + extract_aperture[1]))[0]

            # Normalize to sum=1
            tp = tp[usey] / np.nansum(tp[usey])

            # Model
            if remove_background:
                model2d[usey, i] = tp * spec1d_smooth[i] + background[i]
            else:
                model2d[usey, i] = tp * spec1d_smooth[i]
        
        # Return
        return model2d

    @staticmethod
    def generate_chunks(trace_image, badpix_mask, chunk_width, chunk_overlap):
        """Generates a set of chunks which overlap by 50 percent.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            badpix_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
            chunk_width (int, optional): The width of a chunk. Defaults to 200 pixels.

        Returns:
            list: Each entry is a tuple containing the endpoints of a chunk.
        """

        # Preliminary info
        goody, goodx = np.where(badpix_mask)
        xi, xf = goodx.min(), goodx.max()
        nnx = xf - xi + 1
        yi, yf = goody.min(), goody.max()
        nny = yf - yi + 1
        chunk_width = np.min([chunk_width, nnx])
        chunks = []
        chunks.append([xi, xi + chunk_width])
        if chunk_overlap is None:
            chunk_overlap = 0.5
        for i in range(1, int(1/(1-chunk_overlap) * np.ceil(nnx / chunk_width))):
            vi = chunks[i-1][0] + int((1 - chunk_overlap) * chunk_width)
            vf = np.min([vi + chunk_width, xf])
            chunks.append([vi, vf])
            if vf == xf:
                if vf - vi < (1 - chunk_overlap) * chunk_width:
                   del chunks[-1]
                   chunks[-1][-1] = xf
                break
        return chunks

    @staticmethod
    def combine_chunks(spec1d_chunks, chunks):
        nx, n_chunks = spec1d_chunks.shape
        spec1d = np.full(nx, np.nan)
        distances = np.full_like(spec1d_chunks, np.nan)
        xarr = np.arange(nx)
        for i in range(n_chunks):
            chunk_center = np.mean(chunks[i])
            distances[:, i] = np.abs(xarr - chunk_center)
        for x in range(nx):
            s = np.copy(spec1d_chunks[x, :])
            bad = np.where(~np.isfinite(s))[0]
            w = 1 / distances[x, :]**2
            w[bad] = 0
            s[bad] = np.nan
            bad = np.where(~np.isfinite(w))[0]
            w[bad] = 0
            s[bad] = np.nan
            if np.nansum(w) > 0:
                spec1d[x] = pcmath.weighted_mean(s, w)
        return spec1d
        