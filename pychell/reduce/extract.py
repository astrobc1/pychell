
# Python default modules
import os

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

    def extract_image(self, data, data_image, sregion, output_path, badpix_mask=None):
        """Primary method to extract a single image. Pre-calibration and order tracing are already performed.

        Args:
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
            
            # Alias orders list
            orders_list = order_map.orders_list
        
            # Loop over orders, possibly multi-trace
            for order_index, trace_dict in enumerate(orders_list):

                if sregion.orderbottom < sregion.ordertop:
                    order = sregion.orderbottom + order_index
                else:
                    order = sregion.orderbottom - order_index

                if order in self.extract_orders:
                
                    # Timer
                    stopwatch.lap(trace_dict['label'])
                    
                    # Extract trace
                    #try:
                    spec1d, spec1d_unc, badpix1d = self.extract_trace(data, data_image, sregion, trace_dict, badpix_mask=badpix_mask)
                
                    # Store result
                    reduced_data[order_index, fiber_index, :, :] = np.array([spec1d, spec1d_unc, badpix1d], dtype=float).T

                    # Print end of trace
                    print(f" [{data}] Extracted Trace {trace_dict['label']} in {round(stopwatch.time_since(trace_dict['label']) / 60, 3)} min", flush=True)

                    #except:
                    #    print(f"Warning! Could not extract trace [{trace_dict['label']}] for observation [{data}]")

        # Plot reduced data
        self.plot_extracted_spectra(data, reduced_data, sregion, output_path)

        # Save reduced data to fits file
        fname = f"{output_path}{data.base_input_file_noext}_{data.spec_module.parse_object(data).replace(' ', '_')}_reduced.fits"
        hdu = fits.PrimaryHDU(reduced_data, header=data.header)
        hdu.writeto(fname, overwrite=True, output_verify='ignore')


    ###############
    #### MISC. ####
    ###############

    
    def estimate_snr(trace_image):
        """Crude method to estimate the S/N of the spectrum per 1-dimensional spectral pixel for absorption spectra.

        Args:
            trace_image (np.ndarray): The image containing only one trace.

        Returns:
            float: The estimated S/N.
        """
        spec1d = np.nansum(trace_image, axis=0)
        spec1d_smooth = np.nansum(pcmath.median_filter2d(trace_image, width=3), axis=0)
        med_val = pcmath.weighted_median(spec1d_smooth, percentile=0.98)
        spec1d /= med_val
        spec1d_smooth /= med_val
        res_norm = spec1d - spec1d_smooth
        snr = 1 / np.nanstd(res_norm)
        return snr

    
    def plot_extracted_spectra(self, data, reduced_data, sregion, output_path):
        """Primary method to plot the extracted 1d spectrum for all orders.

        Args:
            data (Echellogram): The data object to extract.
            reduced_data (np.ndarray): The extracted spectra array with shape=(n_orders, n_traces_per_order, n_pixels, 3). The last dimension contains the flux, flux unc, and badpix mask (1=good, 0=bad).
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
                if sregion.orderbottom < sregion.ordertop:
                    order_num = sregion.orderbottom + order_index
                else:
                    order_num = sregion.orderbottom - order_index
                if order_index + 1 > n_orders or order_num not in self.extract_orders:
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

        # Save
        obj = data.spec_module.parse_object(data).replace(' ', '_')
        fname = f"{output_path}{os.sep}{data.base_input_file_noext}_{obj}_preview.png"
        plt.savefig(fname)
        plt.close()


    #########################
    #### HELPER ROUTINES ####
    #########################

    def flag_pixels2d(self, trace_image, trace_mask, model2d):
        """Flags bad pixels in the 2d image based on the residuals between the data and model, which is Extractor dependent.

        Args:
            trace_image (np.ndarray): The trace image.
            trace_mask (np.ndarray): The current bad pixel mask
            model2d (np.ndarray): The nominal 2d model
            badpix_threshold (int, optional): Deviations larger than badpix_threshold * stddev(residuals/model2d_smoothed) are flagged. Defaults to 4.
        """

        # Smooth the 2d image to normalize redisuals
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)

        # Normalized residuals
        norm_res = (trace_image - model2d) / np.sqrt(trace_image_smooth)
        norm_res_smooth = pcmath.median_filter2d(norm_res, width=5)

        # Flag
        use = np.where((norm_res != 0) & (trace_mask == 1) & np.isfinite(norm_res_smooth))
        rms = np.sqrt(np.nansum(norm_res[use]**2 / use[0].size))
        bad = np.where(np.abs(norm_res) > self.badpix_threshold * rms)
        if bad[0].size > 0:
            trace_mask[bad] = 0
            trace_image[bad] = np.nan

    def compute_background_1d(self, trace_image, trace_mask, trace_positions, extract_aperture):
        """Computes the background signal based on regions of low flux. This works best for slit-fed spectrographs where there is still signal on either side of the trace. Fiber-fed spectrographs can use this as well but may wish to rely on other methods not implemented here.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            trace_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
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
            background_locs = np.where(((yarr < np.floor(trace_positions[x] - extract_aperture[0])) | (yarr > np.ceil(trace_positions[x] + extract_aperture[1]))) & np.isfinite(trace_image_smooth[:, x]))[0]
            
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
    def compute_trace_positions_centroids(trace_image, trace_mask, trace_pos_poly_order=2, xrange=None):
        """Computes the trace positions by iteratively computing the centroids of each column.

        Args:
            trace_image (np.ndarray): The image containing only one trace.
            trace_mask (np.ndarray): The bad pixel mask (1=good, 0=bad).
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

        if xrange is None:
            xrange = [0, nx - 1]

        # Loop over columns
        for x in range(xrange[0], xrange[1]+1):
            
            # See if column is even worth looking at
            good = np.where((trace_mask[:, x] == 1) & np.isfinite(trace_image_smooth[:, x]))[0]
            if good.size <= 3:
                continue

            # Biased centroid
            ycen[x] = pcmath.weighted_mean(good, trace_image_smooth[good, x])
    
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
    def boxcar_extraction(trace_image, trace_mask, trace_positions, extract_aperture, trace_profile_cspline, remove_background=False, background=None, read_noise=0):
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
        trace_mask_cp = np.copy(trace_mask)
        
        # Helper array
        yarr = np.arange(ny)

        # Background
        if remove_background:
            trace_image_cp -= np.outer(np.ones(ny), background)
            bad = np.where(trace_image_cp < 0)
            trace_image_cp[bad] = np.nan
            trace_mask_cp[bad] = 0

        # Storage arrays
        spec1d = np.full(nx, np.nan)
        spec1d_err = np.full(nx, np.nan)

        # Loop over cols
        for x in range(nx):

            # Copy arrs
            S_x = np.copy(trace_image_cp[:, x])
            M_x = np.copy(trace_mask_cp[:, x])
                
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

    @staticmethod
    def fix_bad_baxels_interp(trace_image, trace_mask):
        bad = np.where(trace_mask == 0)
        trace_image_out = np.copy(trace_image)
        ny, nx = trace_image.shape
        xarr = np.arange(nx)
        yarr = np.arange(ny)
        if bad[0].size > 0:
            for i in range(bad[0].size):
                trace_image_out[bad[0][i], bad[1][i]] = scipy.interpolate.interp2d(xarr, yarr, trace_image)(bad[1][i], bad[0][i])
        return trace_image_out

        