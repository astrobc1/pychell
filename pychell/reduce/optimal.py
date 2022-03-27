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
import pychell.maths as pcmath
from pychell.reduce.extract import SpectralExtractor

class OptimalExtractor(SpectralExtractor):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, extract_orders=None, oversample_profile=1,
                 trace_pos_poly_order=2,
                 remove_background=False,
                 n_iterations=20, badpix_threshold=5,
                 trace_pos_refine_window_x=None,
                 min_profile_flux=1E-3, extract_aperture=None):
        """Constructor for the Optimal Extractor object.

        Args:
            extract_orders (list, optional): Which orders to extract. Defaults to all orders.
            oversample_profile (int, optional): The oversampling factor for the trace profile. Defaults to 1.
            trace_pos_poly_order (int, optional): The polynomial order for the trace positions. Defaults to 2.
            remove_background (bool, optional): Whether or not to remove a background signal before extraction. Defaults to False.
            n_iterations (int, optional): How many iterations to refine the trace positions, trace profile, and flag bad pixels. Defaults to 20.
            badpix_threshold (int, optional): Deviations larger than badpix_threshold * stddev(residuals) are flagged. Defaults to 4.
            min_profile_flux (float, optional): The minimum flux (relative to 1) to consider in the trace profile. Defaults to 1E-3.
            extract_aperture (list): The number of pixels [below, above] the trace (relative to trace_positions) to consider for extraction of the desired signal.
        """

        # Super init
        super().__init__(extract_orders=extract_orders)

        # Oversample trace profile
        self.oversample_profile = oversample_profile

        # Background
        self.remove_background = remove_background

        # Aperture
        self.extract_aperture = extract_aperture
        self.min_profile_flux = min_profile_flux

        # Trace pos
        self.trace_pos_poly_order = trace_pos_poly_order
        self.trace_pos_refine_window_x = trace_pos_refine_window_x
        
        # Number of iterations
        self.n_iterations = n_iterations

        # Bad pix flagging
        self.badpix_threshold = badpix_threshold
        

    #################################################################
    #### PRIMARY METHOD TO EXTRACT SINGLE TRACE FOR ENTIRE ORDER ####
    #################################################################

    def extract_trace(self, data, image, sregion, trace_dict, badpix_mask, read_noise=None):

        # Copy image
        image = np.copy(image)

        # Full dims
        ny, nx = image.shape

        # Initiate mask
        if badpix_mask is None:
            badpix_mask = np.ones(image.shape)
        else:
            badpix_mask = np.copy(badpix_mask)
        
        # Read noise
        if read_noise is None:
            read_noise = data.spec_module.parse_itime(data) * data.spec_module.detector["read_noise"]

        # Mask image
        sregion.mask_image(image)
        trace_image = np.copy(image)
        trace_mask = np.copy(badpix_mask)
        xarr = np.arange(nx)
        trace_positions = np.polyval(trace_dict["pcoeffs"], xarr)
        for x in range(nx):
            ymid = trace_positions[x]
            y_low = int(np.floor(ymid - trace_dict['height'] / 2))
            y_high = int(np.ceil(ymid + trace_dict['height'] / 2))
            if y_low >= 0 and y_low <= ny - 1:
                trace_image[0:y_low, x] = np.nan
            else:
                trace_image[:, x] = np.nan
            if y_high >= 0 and y_high + 1 <= ny-1:
                trace_image[y_high+1:, x] = np.nan
            else:
                trace_image[:, x] = np.nan

        # Sync
        bad = np.where(~np.isfinite(trace_image) | (trace_mask == 0))
        if bad[0].size > 0:
            trace_image[bad] = np.nan
            trace_mask[bad] = 0
        #breakpoint()
        # New trace positions (approx.)
        if self.trace_pos_refine_window_x is None:
            self.trace_pos_refine_window_x = [sregion.pixmin, sregion.pixmax]
        trace_positions = self.compute_trace_positions_centroids(trace_image, trace_mask, self.trace_pos_poly_order, xrange=self.trace_pos_refine_window_x)

        # Mask image again based on new positions
        trace_image = np.copy(image)
        trace_mask = np.copy(badpix_mask)
        for x in range(nx):
            ymid = trace_positions[x]
            y_low = int(np.floor(ymid - trace_dict['height'] / 2))
            y_high = int(np.ceil(ymid + trace_dict['height'] / 2))
            if y_low >= 0 and y_low <= ny - 1:
                trace_image[0:y_low, x] = np.nan
            else:
                trace_image[:, x] = np.nan
            if y_high >= 0 and y_high + 1 <= ny-1:
                trace_image[y_high+1:, x] = np.nan
            else:
                trace_image[:, x] = np.nan

        # Sync
        bad = np.where(~np.isfinite(trace_image) | (trace_mask == 0))
        trace_image[bad] = np.nan
        trace_mask[bad] = 0

        # Crop in the y direction
        goody, _ = np.where(np.isfinite(trace_image))
        yi, yf = np.max([goody.min(), 0]), np.min([goody.max(), ny - 1])
        trace_image = trace_image[yi:yf+1, :]
        trace_mask = trace_mask[yi:yf+1, :]
        ny, nx = trace_image.shape

        # Flag obvious bad pixels again
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=5)
        peak = pcmath.weighted_median(trace_image_smooth, percentile=0.99)
        bad = np.where((trace_image < 0) | (trace_image > 50 * peak))
        if bad[0].size > 0:
            trace_image[bad] = np.nan
            trace_mask[bad] = 0

        # New starting trace positions
        trace_positions = self.compute_trace_positions_centroids(trace_image, trace_mask, trace_pos_poly_order=self.trace_pos_poly_order, xrange=self.trace_pos_refine_window_x)

        # Estimate background
        if self.remove_background:
            trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)
            background = np.nanmin(trace_image_smooth, axis=0)
            background = pcmath.poly_filter(background, width=51, poly_order=2)
            background_err = np.sqrt(background)
        else:
            background, background_err = None, None

        # Starting trace profile
        trace_profile_cspline = self.compute_vertical_trace_profile(trace_image, trace_mask, trace_positions, background=background)

        # Extract Aperture
        if self.extract_aperture is None:
            _extract_aperture = self.get_extract_aperture(trace_profile_cspline)
        else:
            _extract_aperture = self.extract_aperture

        # Initial opt spectrum
        spec1d, spec1d_unc = self.optimal_extraction(trace_image, trace_mask, trace_positions, trace_profile_cspline, _extract_aperture, background=background, background_err=background_err, read_noise=read_noise, n_iterations=3, spec1d0=None)

        # Main loop
        for i in range(self.n_iterations):

            print(f" [{data}] Extracting Trace {trace_dict['label']}, Iter [{i + 1}/{self.n_iterations}] ...", flush=True)

            # Update trace profile with new positions and mask
            trace_profile_cspline = self.compute_vertical_trace_profile(trace_image, trace_mask, trace_positions, spec1d, background=background)

            # Extract Aperture
            if self.extract_aperture is None:
                _extract_aperture = self.get_extract_aperture(trace_profile_cspline)
            else:
                _extract_aperture = self.extract_aperture

            # Background
            if self.remove_background:
                background, background_err = self.compute_background_1d(trace_image, trace_mask, trace_positions, _extract_aperture)

            trace_positions = self.compute_trace_positions_ccf(trace_image, trace_mask, trace_profile_cspline, trace_positions_estimate=trace_positions, spec1d=spec1d, ccf_window=np.ceil(trace_dict['height'] / 2), xrange=self.trace_pos_refine_window_x, background=background)

            # Optimal extraction
            spec1d, spec1d_unc = self.optimal_extraction(trace_image, trace_mask, trace_positions, trace_profile_cspline, _extract_aperture, background=background, background_err=background_err, read_noise=read_noise, n_iterations=1, spec1d0=spec1d)

            # Re-map pixels and flag in the 2d image.
            if i < self.n_iterations - 1:

                # 2d model
                model2d = self.compute_model2d(trace_image, trace_mask, spec1d, trace_profile_cspline, trace_positions, _extract_aperture, background=background)

                # Flag
                n_bad_current = np.sum(trace_mask)
                self.flag_pixels2d(trace_image, trace_mask, model2d)
                n_bad_new = np.sum(trace_mask)
                
                # Break if nothing new is flagged but force 3 iterations
                if n_bad_current == n_bad_new and i  > 1:
                    break

        # 1d badpix mask
        badpix1d = np.ones(nx)
        bad = np.where(~np.isfinite(spec1d) | (spec1d <= 0) | ~np.isfinite(spec1d_unc) | (spec1d_unc <= 0))[0]
        if bad.size > 0:
            spec1d[bad] = np.nan
            spec1d_unc[bad] = np.nan
            badpix1d[bad] = 0

        # Further flag bad pixels
        spec1d_smooth = pcmath.median_filter1d(spec1d, width=3)
        med_val = pcmath.weighted_median(spec1d_smooth, percentile=0.98)
        bad = np.where(np.abs(spec1d - spec1d_smooth) / med_val > 0.5)[0]
        if bad.size > 0:
            spec1d[bad] = np.nan
            spec1d_unc[bad] = np.nan
            badpix1d[bad] = 0
        
        return spec1d, spec1d_unc, badpix1d

    
    def compute_trace_positions_ccf(self, trace_image, trace_mask, trace_profile_cspline, trace_positions_estimate=None, spec1d=None, ccf_window=10, xrange=None, background=None):
        """Computes the trace positions by cross-correlating the trace profile with each column and fitting a polynomial to the nominal ccf locs.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            trace_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
            trace_profile_cspline (scipy.interpolate.CubicSpline): A CubicSpline object used to create the trace profile (grid is relative to zero).
            trace_positions_estimate (np.ndarray, optional): The current trace positions. If not provided, they are estimated from fitting a polynomial to the centroid of each column.
            spec1d (np.ndarray, optional): The current 1d spectrum used to determine if a column has sufficient signal. Defaults to a smoothed boxcar spectrum.
            ccf_window (int, optional): How many total pixels to consider above/below the current trace positions in the ccf. Defaults to 10 (5 above, 5 below).
            background (np.ndarray, optional): The nominal background signal. Defaults to None.

        Returns:
            np.ndarray: The updated trace positions.
        """

        # The image dimensions
        ny, nx = trace_image.shape
        
        # Helpful arrays
        yarr = np.arange(ny)
        xarr = np.arange(nx)
        
        # Remove background
        if self.remove_background:
            trace_image_no_background = trace_image - np.outer(np.ones(ny), background)
            bad = np.where(trace_image_no_background < 0)
            trace_image_no_background[bad] = np.nan
        else:
            trace_image_no_background = np.copy(trace_image)

        # Smooth image
        trace_image_no_background_smooth = pcmath.median_filter2d(trace_image_no_background, width=3, preserve_nans=False)
        bad = np.where(trace_image_no_background_smooth <= 0)
        if bad[0].size > 0:
            trace_image_no_background_smooth[bad] = np.nan

        # Cross correlate each data column with the trace profile estimate
        y_positions_xc = np.full(nx, np.nan)
        
        # 1d spec
        if spec1d is None:
            spec1d = np.nansum(trace_image_no_background, axis=0)
            spec1d = pcmath.median_filter1d(spec1d, width=3)
        spec1d_norm = spec1d / pcmath.weighted_median(spec1d, percentile=0.95)

        # Normalize trace profile to max=1
        trace_profile = trace_profile_cspline(trace_profile_cspline.x)
        trace_profile /= np.nanmax(trace_profile)

        if xrange is None:
            xrange = [0, nx - 1]

        # Loop over columns
        for x in range(xrange[0], xrange[1]+1):
            
            # See if column is even worth looking at
            good = np.where((trace_mask[:, x] == 1) & np.isfinite(trace_image_no_background_smooth[:, x]) & (yarr > trace_positions_estimate[x] - np.ceil(ccf_window / 2)) & (yarr < trace_positions_estimate[x] + np.ceil(ccf_window / 2)))[0]
            if good.size <= 3 or spec1d_norm[x] < 0.2:
                continue
            
            # Normalize data column to max=1
            data_x = trace_image_no_background_smooth[:, x] / np.nanmax(trace_image_no_background_smooth[:, x])
            
            # CCF lags
            lags = np.arange(trace_positions_estimate[x] - ccf_window / 2, trace_positions_estimate[x] + ccf_window / 2)
            
            # Perform CCF
            ccf = pcmath.cross_correlate(yarr, data_x, trace_profile_cspline.x, trace_profile, lags, kind="xc")

            # Fit ccf
            try:
                iymax = np.nanargmax(ccf)
                if iymax - 1 > 0 and iymax + 1 < len(lags):
                    pfit = np.polyfit(lags[iymax-1:iymax+2], ccf[iymax-1:iymax+2], 2)
                    ypos = -0.5 * pfit[1] / pfit[0]
                    if ypos >= 0 and ypos <= ny - 1:
                        y_positions_xc[x] = ypos
                else:
                    continue
            except:
                continue
        
        # Smooth the deviations
        y_positions_xc_smooth = pcmath.median_filter1d(y_positions_xc, width=3)
        bad = np.where(np.abs(y_positions_xc - y_positions_xc_smooth) > 0.5)[0]
        if bad.size > 0:
            y_positions_xc[bad] = np.nan
        good = np.where(np.isfinite(y_positions_xc))[0]
        
        # Fit with a polynomial and generate
        if good.size > 2 * self.trace_pos_poly_order:
            pfit = np.polyfit(xarr[good], y_positions_xc[good], self.trace_pos_poly_order)
            trace_positions = np.polyval(pfit, xarr)
        else:
            trace_positions = np.copy(trace_positions_estimate)
        
        # Return
        return trace_positions


    ############################
    #### OPTIMAL EXTRACTION ####
    ############################
    
    def optimal_extraction(self, trace_image, trace_mask, trace_positions, trace_profile_cspline, extract_aperture, background=None, background_err=None, read_noise=0, n_iterations=1, spec1d0=None):
        """Standard optimal extraction. A single column from the data is a function of y pixels ($S_{y}$), and is modeled as:

            F_{y} = A * P_{y}

            where P_{y} is the nominal vertical profile and may be arbitrarily scaled. The parameter $A$ is the scaling of the input signal and is fit for in the least squares sense by minimizing the function:

            phi = \sum_{y} w_{y} (S_{y} - F_{y})^{2} = \sum_{y} w_{y} (S_{y} - A P_{y})^{2}

            where
            
            w_{y} = P_{y}^{2} M_{y} / \sigma_{y}^{2}, \sigma_{2} = R^{2} + S_{y} + B/(N − 1), N = number of good pixels used in finding the background, B, and M_{y} is a binary mask (1=good, 0=bad).

            The final 1d value is then A \sum_{y} P_{y}.

        Args:
            trace_image (np.ndarray): The trace image of shape ny, nx.
            trace_mask (np.ndarray): The bad pix mask.
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
        trace_mask_cp = np.copy(trace_mask)
        
        # Helper array
        yarr = np.arange(ny)

        # Background
        if self.remove_background:
            trace_image_cp -= np.outer(np.ones(ny), background)
            bad = np.where(trace_image_cp < 0)
            trace_image_cp[bad] = np.nan
            trace_mask_cp[bad] = 0

        # Storage arrays
        spec1d = np.full(nx, np.nan)
        spec1d_err = np.full(nx, np.nan)

        # Tpy
        tpy = trace_profile_cspline(trace_profile_cspline.x)

        # Loop over iterations
        for i in range(n_iterations):

            # Loop over cols
            for x in range(nx):

                # Copy arrs
                S_x = np.copy(trace_image_cp[:, x])
                M_x = np.copy(trace_mask_cp[:, x])
                    
                # Check if column is worth extracting
                if np.nansum(M_x) <= 1:
                    continue

                # Shift Trace Profile
                P_x = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x], tpy, yarr)
                
                # Determine which pixels to use from the aperture
                use = np.where((yarr >= trace_positions[x] + extract_aperture[0]) & (yarr <= trace_positions[x] + extract_aperture[1]))[0]

                # Copy arrays
                S = np.copy(S_x[use])
                M = np.copy(M_x[use])
                P = np.copy(P_x[use])
                P /= np.nansum(P)

                # Variance
                if self.remove_background:
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

    def compute_model2d(self, trace_image, trace_mask, spec1d, trace_profile_cspline, trace_positions, extract_aperture, background=None):
        """Generates the nominal 2d model from the 1d spectrum.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            trace_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
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

        # Trace profile at zero point
        tpy = trace_profile_cspline(trace_profile_cspline.x)

        # Loop over cols
        for i in range(nx):

            # Compute trace profile for this column
            tp = scipy.interpolate.CubicSpline(trace_profile_cspline.x + trace_positions[i], tpy)(yarr)

            # Which pixels to use
            usey = np.where((yarr >= trace_positions[i] + extract_aperture[0]) & (yarr <= trace_positions[i] + extract_aperture[1]))[0]

            # Normalize to sum=1
            tp = tp[usey] / np.nansum(tp[usey])

            # Model
            if self.remove_background:
                model2d[usey, i] = tp * spec1d_smooth[i] + background[i]
            else:
                model2d[usey, i] = tp * spec1d_smooth[i]

        # Return
        return model2d

    def compute_vertical_trace_profile(self, trace_image, trace_mask, trace_positions, spec1d=None, background=None):
        """Computes the 1-dimensional trace profile (vertical).

        Args:
            trace_image (np.ndarray): The trace image.
            trace_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
            trace_positions (np.ndarray): The trace positions.
            oversample (int, optional): An oversampling factor for the trace profile. Defaults to 1.
            spec1d (np.ndarray, optional): The current 1d spectrum. Defaults to a summation over columns.
            background (np.ndarray, optional): If remove_background is True, this vector is removed before computing the trace profile. Defaults to None.

        Returns:
            CubicSpline: A scipy.interpolate.CubicSpline object.
        """
        
        # Image dims
        ny, nx = trace_image.shape
        
        # Helpful array
        xarr = np.arange(nx)
        yarr = np.arange(ny)

        # Smooth
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)
        
        # Create a fiducial high resolution grid centered at zero
        yarr_hr0 = np.arange(int(np.floor(-ny / 2)), int(np.ceil(ny / 2)) + 1 / self.oversample_profile, 1 / self.oversample_profile)
        trace_image_rect_norm = np.full((len(yarr_hr0), nx), np.nan)

        # 1d spec info (smooth)
        if spec1d is None:
            trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)
            spec1d = np.nansum(trace_image_smooth, axis=0)
        spec1d_smooth = pcmath.median_filter1d(spec1d, width=3)
        spec1d_smooth /= pcmath.weighted_median(spec1d_smooth, percentile=0.98)
        
        # Rectify
        for x in range(nx):
            good = np.where(np.isfinite(trace_image[:, x]) & (trace_mask[:, x] == 1))[0]
            if good.size >= 3 and spec1d_smooth[x] > 0.2:
                if self.remove_background:
                    col_hr_shifted = pcmath.lin_interp(yarr - trace_positions[x], trace_image_smooth[:, x] - background[x], yarr_hr0)
                else:
                    col_hr_shifted = pcmath.lin_interp(yarr - trace_positions[x], trace_image_smooth[:, x], yarr_hr0)
                bad = np.where(col_hr_shifted < 0)[0]
                if bad.size > 0:
                    col_hr_shifted[bad] = 0
                trace_image_rect_norm[:, x] = col_hr_shifted / spec1d[x]
        
        # Compute trace profile
        mask_temp = np.ones_like(trace_image_rect_norm)
        bad = np.where(~np.isfinite(trace_image_rect_norm))
        mask_temp[bad] = 0
        n_pix_per_row = np.sum(mask_temp, axis=1)
        bad = np.where(n_pix_per_row < nx / 100)[0]
        trace_profile_median = np.nanmedian(trace_image_rect_norm, axis=1)
        #trace_profile_median[bad] = np.nan
        #weights = 1 / (trace_image_rect_norm - np.outer(trace_profile_median, np.ones(nx)))**2
        #bad = np.where(~np.isfinite(weights))
        #weights[bad] = 0
        #trace_profile_mean = pcmath.weighted_mean(trace_image_rect_norm, weights, axis=1)
        
        # Compute cubic spline for profile
        good = np.where(np.isfinite(trace_profile_median))[0]
        trace_profile_cspline = scipy.interpolate.CubicSpline(yarr_hr0[good], trace_profile_median[good], extrapolate=False)

        # Trim 1 pixel on each end
        tpx, tpy = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
        trace_profile_cspline = scipy.interpolate.CubicSpline(tpx[1*self.oversample_profile:len(tpx)-1*self.oversample_profile], tpy[1*self.oversample_profile:len(tpx)-1*self.oversample_profile], extrapolate=False)

        # Set profile to zero where is less than min_profile_flux (relative to 1)
        #breakpoint()# matplotlib.use("MacOSX"); plt.plot(trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)); plt.show()
        #tpys = np.sort(tpy)
        #tpy -= np.nanmean(tpys[0:2*self.oversample_profile])
        #tpy /= np.nanmax(tpy)
        #breakpoint()

        # Center profile at zero
        prec = 1000
        tpxhr = np.arange(tpx[0], tpx[-1], 1 / prec)
        tpyhr = trace_profile_cspline(tpxhr)
        mid = tpx[np.nanargmax(tpy)]
        consider = np.where((tpxhr > mid - 3*self.oversample_profile) & (tpxhr < mid + 3*self.oversample_profile))[0]
        trace_max_pos = tpxhr[consider[np.nanargmax(tpyhr[consider])]]
        trace_profile_cspline = scipy.interpolate.CubicSpline(trace_profile_cspline.x - trace_max_pos,
                                                              trace_profile_cspline(trace_profile_cspline.x), extrapolate=False)
        
        # bad_left = np.where((tpx < 0) & (tpy < self.min_profile_flux))[0]
        # if bad_left.size > 0:
        #    tpy[0:bad_left.max()] = 0
        # bad_right = np.where((tpx > 0) & (tpy < self.min_profile_flux))[0]
        # if bad_right.size > 0:
        #    tpy[bad_right.min():] = 0

        # Subtract min_profile_flux
        # tpy -= self.min_profile_flux
        # bad = np.where(tpy < 0)[0]
        # if bad.size > 0:
        #    tpy[bad] = 0

        # Final profile
        tpx, tpy = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
        trace_profile_cspline = scipy.interpolate.CubicSpline(tpx, tpy / np.nanmax(tpy), extrapolate=False)

        # Return
        return trace_profile_cspline

    def get_extract_aperture(self, trace_profile_cspline):
        """Computes the extraction aperture based on where there is enough signal.

        Args:
            trace_profile_cspline (scipy.interpolate.CubicSpline): A CubicSpline object used to create the trace profile (grid is relative to zero).

        Returns:
            list: The recommended number of pixels [below, above] the trace (relative to trace_positions) to consider for extraction of the desired signal.
        """
        tpx, tpy = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
        tpy /= np.nanmax(tpy)
        imax = np.nanargmax(tpy)
        xleft = tpx.min()
        xright = tpx.max()
        for i in range(imax, -1, -1):
            if tpy[i] < self.min_profile_flux:
                xleft = tpx[i]
                break
        for i in range(imax, len(tpx)):
            if tpy[i] < self.min_profile_flux:
                xright = tpx[i]
                break
        
        #good = np.where(tpy >= self.min_profile_flux)[0]
        #xi, xf = good.min(), good.max()
        #extract_aperture = [-1 * int(np.abs(np.floor(tpx[xi]))) - 1, int(np.ceil(tpx[xf])) + 1]
        extract_aperture = [np.floor(xleft), np.ceil(xright)]
        return extract_aperture