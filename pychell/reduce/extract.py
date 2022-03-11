
# Python default modules
import os
import copy

# Science / Math
import numpy as np
import scipy.interpolate
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

class SpectralExtractor:
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, extract_orders=None):
        """Base constructor for a SpectralExtractor class.

        Args:
            extract_orders (list, optional): Which orders to extract (relative to the image, starting at 1). Defaults to all orders.
        """
        self.extract_orders = extract_orders


    ################################################
    #### PRIMARY METHOD TO EXTRACT ENTIRE IMAGE ####
    ################################################

    def extract_image(self, recipe, data, data_image, badpix_mask=None):
        """Primary method to extract a single image. Pre-calibration and order tracing are already performed.

        Args:
            recipe (ReduceRecipe): The recipe object.
            data (Echellogram): The data object to extract.
            data_image (np.ndarray): The actual image to extract.
            badpix_mask (np.ndarray, optional): An initial bad pix mask image (1=good, 0=bad). Defaults to None.
        """

        # Stopwatch
        stopwatch = pcutils.StopWatch()
        
        # The image dimensions
        ny, nx = data_image.shape

        # The number of orders and fibers (if relevant)
        n_orders = len(data.order_maps[0].orders_list)
        n_traces = len(data.order_maps)

        # Default storage is a single HDUList
        # Last dimension is extracted spectra, uncertainty, badpix mask (1=good, 0=bad)
        reduced_data = np.full((n_orders, n_traces, nx, 3), np.nan)

        # Loop over fibers
        for fiber_index, order_map in enumerate(data.order_maps):

            # Which orders
            if self.extract_orders is None:
                if recipe.tracer.orders[0] <= recipe.tracer.orders[1]:
                    self.extract_orders = np.arange(recipe.tracer.orders[0], recipe.tracer.orders[1] + 1).astype(int)
                else:
                    self.extract_orders = np.arange(recipe.tracer.orders[1], recipe.tracer.orders[0] + 1).astype(int)
            
            # Alias orders list
            orders_list = order_map.orders_list
        
            # Loop over orders, possibly multi-trace
            for order_index, trace_dict in enumerate(orders_list):

                if recipe.tracer.orders[0] <= recipe.tracer.orders[1]:
                    order = recipe.tracer.orders[0] + order_index
                else:
                    order = recipe.tracer.orders[0] - order_index

                if order in self.extract_orders:
                
                    # Timer
                    stopwatch.lap(trace_dict['label'])
                    
                    # Extract trace
                    try:
                        spec1d, spec1d_unc, badpix1d = self.extract_trace(data, data_image, recipe.xrange, trace_dict, badpix_mask=badpix_mask)
                    
                        # Store result
                        reduced_data[order_index, fiber_index, :, :] = np.array([spec1d, spec1d_unc, badpix1d], dtype=float).T

                        # Print end of trace
                        print(f" [{data}] Extracted Trace {trace_dict['label']} in {round(stopwatch.time_since(trace_dict['label']) / 60, 3)} min", flush=True)

                    except:
                        print(f"Warning! Could not extract trace [{trace_dict['label']}] for observation [{data}]")

        # Plot reduced data
        self.plot_extracted_spectra(recipe, data, reduced_data)
        
        # Create a filename
        obj = data.spec_module.parse_object(data).replace(' ', '_')
        fname = f"{recipe.output_path}spectra{os.sep}{data.base_input_file_noext}_{obj}_preview.png"
        
        # Save
        plt.savefig(fname)
        plt.close()

        # Save reduced data
        fname = f"{recipe.output_path}spectra{os.sep}{data.base_input_file_noext}_{obj}_reduced.fits"
        hdu = fits.PrimaryHDU(reduced_data, header=data.header)
        hdu.writeto(fname, overwrite=True, output_verify='ignore')


    ###############
    #### MISC. ####
    ###############

    @staticmethod
    def estimate_snr(trace_image):
        """Crude method to estimate the S/N of the spectrum per 1-dimensional spectral pixel for absorption spectra where the continuum dominates.

        Args:
            trace_image (np.ndarray): The image containing only one trace.

        Returns:
            float: The estimated S/N.
        """
        spec1d = np.nansum(trace_image, axis=0)
        spec1d /= pcmath.weighted_median(spec1d, percentile=0.99)
        spec1d_smooth = np.nansum(pcmath.median_filter2d(trace_image, width=3), axis=0)
        spec1d_smooth /= pcmath.weighted_median(spec1d_smooth, percentile=0.99)
        res_norm = spec1d - spec1d_smooth
        snr = 1 / np.nanstd(res_norm)
        return snr

    @staticmethod
    def plot_extracted_spectra(recipe, data, reduced_data):
        """Primary method to plot the extracted 1d spectrum for all orders.

        Args:
            data (Echellogram): The data object to extract.
            reduced_data (np.ndarray): The extracted spectra array with shape=(n_orders, n_traces_per_order, n_pixels, 3). The last dimension contains the flux, flux unc, and badpix mask (1=good, 0=bad).

        Returns:
            Figure: A Matplotlib Figure object (not yet saved).
        """
    
        # Numbers
        n_orders = reduced_data.shape[0]
        n_traces = reduced_data.shape[1]
        nx = reduced_data.shape[2]
        
        # The number of x pixels
        xarr = np.arange(nx)
        
        # Plot settings
        plot_width = 20
        plot_height = 20
        dpi = 250
        n_cols = 3
        n_rows = int(np.ceil(n_orders / n_cols))
        
        # Create a figure
        fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(plot_width, plot_height), dpi=dpi)
        axarr = np.atleast_2d(axarr)
        
        # For each subplot, plot all traces
        for row in range(n_rows):
            for col in range(n_cols):
                
                # The order index
                order_index = n_cols * row + col
                if recipe.tracer.orders[0] <= recipe.tracer.orders[1]:
                    order_num = order_index + recipe.tracer.orders[0]
                else:
                    order_num = recipe.tracer.orders[0] - order_index
                if order_index + 1 > n_orders:
                    continue

                # Loop over traces
                for trace_index in range(n_traces):
                    
                    # Views
                    spec1d = reduced_data[order_index, trace_index, :, 0]
                    badpix1d = reduced_data[order_index, trace_index, :, 2]
                    
                    # Good pixels
                    good = np.where(badpix1d)[0]
                    if good.size == 0:
                        continue
                    
                    # Plot the extracted spectrum
                    axarr[row, col].plot(xarr, spec1d / pcmath.weighted_median(spec1d, percentile=0.99) + 1.1*trace_index, label='Optimal', lw=0.5)

                # Title
                axarr[row, col].set_title(f"Order {order_num}", fontsize=6)
                
                # Axis labels
                axarr[row, col].set_xlabel('X Pixels', fontsize=6)
                axarr[row, col].set_ylabel('Norm. Flux', fontsize=6)
                axarr[row, col].tick_params(labelsize=6)
        
        # Tight layout
        plt.tight_layout()

        return fig


    #########################
    #### HELPER ROUTINES ####
    #########################

    @staticmethod
    def flag_pixels2d(trace_image, badpix_mask, model2d, badpix_threshold=4):
        """Flags bad pixels in the 2d image based on the residuals between the data and model, which is Extractor dependent.

        Args:
            trace_image (np.ndarray): The trace image.
            badpix_mask (np.ndarray): The current bad pixel mask
            model2d (np.ndarray): The nominal 2d model
            badpix_threshold (int, optional): Deviations larger than badpix_threshold * stddev(residuals) are flagged. Defaults to 4.

        Returns:
            np.ndarray: The updated trace image.
            np.ndarray: The updated badpix mask.
        """

        trace_image_cp = np.copy(trace_image)
        badpix_mask_cp = np.copy(badpix_mask)

        # Smooth the 2d image to normalize redisuals
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)

        # Normalized residuals
        norm_res = (trace_image - model2d) / np.sqrt(trace_image_smooth)

        # Flag
        use = np.where((norm_res != 0) & (badpix_mask == 1))
        bad = np.where(np.abs(norm_res) > badpix_threshold * np.nanstd(norm_res[use]))
        if bad[0].size > 0:
            badpix_mask_cp[bad] = 0
            trace_image_cp[bad] = np.nan
        
        return trace_image_cp, badpix_mask_cp

    @staticmethod
    def mask_image(image, xrange, poly_mask_bottom, poly_mask_top):
        """Masks an image in-place based on left/right enpoints, and top/bottom polynomials.

        Args:
            image (np.ndarray): The image to mask.
            xrange (int): The x range to consider.
            poly_mask_bottom (np.ndarray): The polynomial coefficients to mask the top of the image. The units are in detector pixels.
            poly_mask_top (np.ndarray): The polynomial coefficients to mask the bottom of the image. The units are in detector pixels.
        """

        # Dims
        ny, nx = image.shape
        
        # Helpful arrs
        xarr = np.arange(nx)
        yarr = np.arange(ny)

        # Mask left/right
        image[:, 0:xrange[0]] = np.nan
        image[:, xrange[1]:] = np.nan

        # Top polynomial
        x_bottom = np.array([p[0] for p in poly_mask_bottom], dtype=float)
        y_bottom = np.array([p[1] for p in poly_mask_bottom], dtype=float)
        pfit_bottom = np.polyfit(x_bottom, y_bottom, len(x_bottom) - 1)
        poly_bottom = np.polyval(pfit_bottom, xarr)

        # Bottom polynomial
        x_top = np.array([p[0] for p in poly_mask_top], dtype=float)
        y_top = np.array([p[1] for p in poly_mask_top], dtype=float)
        pfit_top = np.polyfit(x_top, y_top, len(x_top) - 1)
        poly_top = np.polyval(pfit_top, xarr)
        
        for x in range(nx):
            bad = np.where((yarr < poly_bottom[x]) | (yarr > poly_top[x]))[0]
            image[bad, x] = np.nan

    @staticmethod
    def compute_extract_aperture(trace_profile_cspline, min_profile_flux=0.01):
        """Computes the extraction aperture based on where there is enough signal.

        Args:
            trace_profile_cspline (scipy.interpolate.CubicSpline): A CubicSpline object used to create the trace profile (grid is relative to zero).
            min_profile_flux (float, optional): The minimum flux (relative to 1) to consider in the trace profile. Defaults to 0.05.

        Returns:
            list: The recommended number of pixels [below, above] the trace (relative to trace_positions) to consider for extraction of the desired signal.
        """
        tpx, tpy = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
        tpy /= np.nanmax(tpy)
        good = np.where(tpy >= min_profile_flux)[0]
        xi, xf = good.min(), good.max()
        extract_aperture = [-1 * int(np.abs(np.ceil(tpx[xi]))), int(np.ceil(tpx[xf]))]
        return extract_aperture

    @staticmethod
    def compute_background_1d(trace_image, badpix_mask, trace_positions, extract_aperture):
        """Computes the background signal based on regions of low flux. This works best for slit-fed spectrographs where there is still signal on either side of the trace. Fiber-fed spectrographs must resort to other methods not yet implemented.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            badpix_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
            trace_positions (np.ndarray): The trace positions.
            extract_aperture (list): The number of pixels [below, above] the trace (relative to trace_positions) to consider for extraction of the desired signal.
            background_smooth_width (int, optional): How many pixels to use to smooth the background with a rolling median filter. Defaults to None (no smoothing).
            background_smooth_poly_order (int, optional): The order of the rolling polynomial filter for the background. Defaults to None (no smoothing).

        Returns:
            np.ndarray: The background signal.
            np.ndarray: The uncertainty in the background signal.
        """
        
        # Image dims
        ny, nx = trace_image.shape
        
        # Helper array
        yarr = np.arange(ny)
        
        # Empty arrays
        background = np.full(nx, np.nan)
        background_err = np.full(nx, np.nan)
        n_pix_used = np.full(nx, np.nan)

        # Smooth image
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)
        
        # Loop over columns
        for x in range(nx):
            
            # Identify regions low in flux
            background_locs = np.where(((yarr < trace_positions[x] - extract_aperture[0]) | (yarr > trace_positions[x] + extract_aperture[1])) & np.isfinite(trace_image_smooth[:, x]))[0]
            
            # Continue if no locs
            if background_locs.size == 0:
                continue
            
            # Compute the median counts behind the trace (use 25th percentile to mitigate injecting negative values)
            background[x] = np.nanmedian(trace_image_smooth[background_locs, x])
            
            # Compute background
            if background[x] >= 0 and np.isfinite(background[x]):
                n_good = np.where(np.isfinite(trace_image_smooth[background_locs, x]))[0].size
                n_pix_used[x] = n_good
                if n_good <= 1:
                    background[x] = np.nan
                else:
                    background_err[x] = np.sqrt(background[x] / (n_good - 1))
            else:
                background[x] = np.nan
                background_err[x] = np.nan

        # Polynomial filter
        background = pcmath.poly_filter(background, width=31, poly_order=2)
        background_err = np.sqrt(background / (n_pix_used - 1))

        # Flag negative values
        bad = np.where(background < 0)[0]
        if bad.size > 0:
            background[bad] = 0
            background_err[bad] = 0

        # Return
        return background, background_err

    @staticmethod
    def compute_vertical_trace_profile(trace_image, badpix_mask, trace_positions, oversample=1, spec1d=None, remove_background=False, background=None, min_profile_flux=1E-2):
        """Computes the 1-dimensional trace profile (purely vertical).

        Args:
            trace_image (np.ndarray): The trace image.
            badpix_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
            trace_positions (np.ndarray): The trace positions.
            oversample (int, optional): An oversampling factor for the trace profile. Defaults to 1.
            spec1d (np.ndarray, optional): The current 1d spectrum. Defaults to a summation over columns.
            remove_background (bool, optional): Whether or not to remove the background. Defaults to False.
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
        yarr_hr0 = np.arange(int(np.floor(-ny / 2)), int(np.ceil(ny / 2)) + 1 / oversample, 1 / oversample)
        trace_image_rect_norm = np.full((len(yarr_hr0), nx), np.nan)

        # 1d spec info (smooth)
        if spec1d is None:
            spec1d = np.nansum(trace_image, axis=0)
        spec1d_smooth = pcmath.median_filter1d(spec1d, width=3)
        spec1d_smooth /= pcmath.weighted_median(spec1d_smooth, percentile=0.98)
        
        # Rectify
        for x in range(nx):
            good = np.where(np.isfinite(trace_image[:, x]) & (badpix_mask[:, x] == 1))[0]
            if good.size >= 3 and spec1d_smooth[x] > 0.2:
                if remove_background:
                    col_hr_shifted = pcmath.lin_interp(yarr - trace_positions[x], trace_image_smooth[:, x] - background[x], yarr_hr0)
                else:
                    col_hr_shifted = pcmath.lin_interp(yarr - trace_positions[x], trace_image_smooth[:, x], yarr_hr0)
                bad = np.where(col_hr_shifted < 0)[0]
                if bad.size > 0:
                    col_hr_shifted[bad] = np.nan
                trace_image_rect_norm[:, x] = col_hr_shifted / np.nansum(col_hr_shifted)
        
        # Compute trace profile
        mask_temp = np.ones_like(trace_image_rect_norm)
        bad = np.where(~np.isfinite(trace_image_rect_norm))
        mask_temp[bad] = 0
        n_pix_per_row = np.sum(mask_temp, axis=1)
        bad = np.where(n_pix_per_row < nx / 100)[0]
        trace_profile_median = np.nanmedian(trace_image_rect_norm, axis=1)
        trace_profile_median[bad] = np.nan
        weights = 1 / (trace_image_rect_norm - np.outer(trace_profile_median, np.ones(nx)))**2
        bad = np.where(~np.isfinite(weights))
        weights[bad] = 0
        trace_profile_mean = pcmath.weighted_mean(trace_image_rect_norm, weights, axis=1)
        
        # Compute cubic spline for profile
        good = np.where(np.isfinite(trace_profile_mean))[0]
        trace_profile_cspline = scipy.interpolate.CubicSpline(yarr_hr0[good], trace_profile_mean[good], extrapolate=False)

        # Trim 2 pixels on each end
        tpx, tpy = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
        trace_profile_cspline = scipy.interpolate.CubicSpline(tpx[2*oversample:len(tpx)-2*oversample], tpy[2*oversample:len(tpx)-2*oversample], extrapolate=False)
        
        # Ensure trace profile is centered at zero
        prec = 1000
        yhr = np.arange(trace_profile_cspline.x[0], trace_profile_cspline.x[-1], 1 / prec)
        tphr = trace_profile_cspline(yhr)
        mid = np.nanmean(trace_profile_cspline.x)
        consider = np.where((yhr > mid - 3*oversample) & (yhr < mid + 3*oversample))[0]
        trace_max_pos = yhr[consider[np.nanargmax(tphr[consider])]]
        trace_profile_cspline = scipy.interpolate.CubicSpline(trace_profile_cspline.x - trace_max_pos,
                                                              trace_profile_cspline(trace_profile_cspline.x), extrapolate=False)

        # Set profile to zero where is less than min_profile_flux (relative to 1)
        tpx, tpy = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
        tpys = np.sort(tpy)
        tpy -= np.nanmean(tpys[0:3])
        tpy /= np.nanmax(tpy)
        
        bad_left = np.where((tpx < 0) & (tpy < min_profile_flux))[0]
        if bad_left.size > 0:
            tpy[0:bad_left.max()] = 0
        bad_right = np.where((tpx > 0) & (tpy < min_profile_flux))[0]
        if bad_right.size > 0:
            tpy[bad_right.min():] = 0

        # Subtract min_profile_flux
        tpy -= min_profile_flux
        bad = np.where(tpy < 0)[0]
        if bad.size > 0:
            tpy[bad] = 0

        # Final profile
        trace_profile_cspline = scipy.interpolate.CubicSpline(tpx, tpy / np.nanmax(tpy), extrapolate=False)

        # Return
        return trace_profile_cspline


    @staticmethod
    def compute_trace_positions_ccf(trace_image, badpix_mask, trace_profile_cspline, extract_aperture, trace_positions_estimate=None, spec1d=None, ccf_window=10, remove_background=False, background=None, trace_pos_poly_order=2):
        """Computes the trace positions by cross-correlating the trace profile with each column and fitting a polynomial to the nominal ccf locs.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            badpix_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
            trace_profile_cspline (scipy.interpolate.CubicSpline): A CubicSpline object used to create the trace profile (grid is relative to zero).
            extract_aperture (list): The number of pixels [below, above] the trace (relative to trace_positions) to consider for extraction of the desired signal.
            trace_positions_estimate (np.ndarray, optional): The current trace positions. If not provided, they are estimated from fitting a polynomial to the centroid of each column.
            spec1d (np.ndarray, optional): The current 1d spectrum used to determine if a column has sufficient signal. Defaults to a smoothed boxcar spectrum.
            ccf_window (int, optional): How many total pixels to consider above/below the current trace positions in the ccf. Defaults to 10 (5 above, 5 below).
            remove_background (bool, optional): Whether or not to remove the background. Defaults to False.
            background (np.ndarray, optional): The nominal background signal. Defaults to None.
            trace_pos_poly_order (int, optional): The polynomial order to fit to the ccf positions. Defaults to 2.

        Returns:
            np.ndarray: The updated trace positions.
        """

        # The image dimensions
        ny, nx = trace_image.shape
        
        # Helpful arrays
        yarr = np.arange(ny)
        xarr = np.arange(nx)
        
        # Remove background
        if remove_background:
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

        # Loop over columns
        for x in range(nx):
            
            # See if column is even worth looking at
            good = np.where((badpix_mask[:, x] == 1) & np.isfinite(trace_image_no_background_smooth[:, x]) & (yarr > trace_positions_estimate[x] - np.ceil(ccf_window / 2)) & (yarr < trace_positions_estimate[x] + np.ceil(ccf_window / 2)))[0]
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
        if good.size > 2 * trace_pos_poly_order:
            pfit = np.polyfit(xarr[good], y_positions_xc[good], trace_pos_poly_order)
            trace_positions = np.polyval(pfit, xarr)
        else:
            trace_positions = np.copy(trace_positions_estimate)
        
        # Return
        return trace_positions

    @staticmethod
    def compute_trace_positions_centroids(trace_image, badpix_mask, trace_pos_poly_order=2):
        """Computes the trace positions by iteratively computing the centroids of each column.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            badpix_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
            spec1d (np.ndarray, optional): The current 1d spectrum. Defaults to a summation over columns.
            trace_pos_poly_order (int, optional): The polynomial order to fit the centroids with. Defaults to 2.

        Returns:
            np.ndarray: The trace positions.
        """

        # The image dimensions
        ny, nx = trace_image.shape
        
        # Helpful arrays
        yarr = np.arange(ny)
        xarr = np.arange(nx)

        # Smooth image
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=3, preserve_nans=False)

        # Y centroids
        ycen = np.full(nx, np.nan)

        # Loop over columns
        for x in range(nx):
            
            # See if column is even worth looking at
            good = np.where((badpix_mask[:, x] == 1) & np.isfinite(trace_image_smooth[:, x]))[0]
            if good.size <= 3:
                continue

            # Biased centroid
            ycen[x] = pcmath.weighted_mean(good, trace_image_smooth[good, x]**2)
    
        # Smooth the deviations
        ycen_smooth = pcmath.median_filter1d(ycen, width=3)
        bad = np.where(np.abs(ycen - ycen_smooth) > 0.5)[0]
        if bad.size > 0:
            ycen[bad] = np.nan
        good = np.where(np.isfinite(ycen))[0]
    
        # Fit with a polynomial
        pfit = np.polyfit(xarr[good], ycen[good], trace_pos_poly_order)
        trace_positions = np.polyval(pfit, xarr)
    
        # Return
        return trace_positions

    @staticmethod
    def boxcar_extraction(trace_image, badpix_mask, trace_positions, extract_aperture, trace_profile_cspline, remove_background=False, background=None, read_noise=0):
        """Standard boxcar extraction. A single column from the data is a function of y pixels ($S_{y}$), and is modeled as:

            F_{y} = A * P_{y}

            where P_{y} is the nominal vertical profile and may be arbitrarily scaled. The parameter $A$ is the scaling of the input signal and is fit for in the least squares sense by minimizing the function:

            phi = \sum_{y} w_{y} (S_{y} - F_{y})^{2} = \sum_{y} w_{y} (S_{y} - A P_{y})^{2}

            where
            
            w_{y} = M_{y}, where M_{y} is a binary mask (1=good, 0=bad).

            The final 1d value is then A \sum_{y} P_{y}.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            badpix_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
            trace_positions (np.ndarray): The trace positions of length nx.
            extract_aperture (np.ndarray): A list of the number of pixels above and below trace_positions for each column.
            trace_profile_cspline (scipy.interpolate.CubicSpline): A CubicSpline object used to create the trace profile (grid is relative to zero).
            remove_background (bool, optional): Whether or not to remove the background. Defaults to False.
            background (np.ndarray, optional): The background. Defaults to None.
            read_noise (float, optional): Defaults to 0.

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
            P /= np.nansum(P)

            # Variance
            if remove_background:
                v = read_noise**2 + S + background_err[x]**2
            else:
                v = read_noise**2 + S

            # Weights
            w = np.copy(M)
            w /= np.nansum(w)

            # Least squares
            A = np.nansum(w * P * S) / np.nansum(w * P**2)

            # Final 1d spec
            spec1d[x] = A * np.nansum(P)
            spec1d_err[x] = np.sqrt(np.nansum(v) / (np.nansum(M) - 1)) / np.nansum(w * P)

        return spec1d, spec1d_err