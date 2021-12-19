
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

# LLVM
from numba import jit, njit

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath

class SpectralExtractor:
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, extract_orders=None):
        self.extract_orders = extract_orders


    ################################################
    #### PRIMARY METHOD TO EXTRACT ENTIRE IMAGE ####
    ################################################

    def extract_image(self, recipe, data, data_image, badpix_mask=None):

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
                self.extract_orders = np.arange(1, len(order_map.orders_list) + 1).astype(int)
            
            # Alias orders list
            orders_list = order_map.orders_list
        
            # A trace mask
            trace_map_image = recipe.tracer.gen_image(orders_list, ny, nx, mask_left=recipe.mask_left, mask_right=recipe.mask_right, mask_top=recipe.mask_top, mask_bottom=recipe.mask_bottom)
        
            # Mask edge pixels as nan
            self.mask_image(data_image, recipe.mask_left, recipe.mask_right, recipe.mask_top, recipe.mask_bottom)
        
            # Loop over orders, possibly multi-trace
            for order_index, trace_dict in enumerate(orders_list):

                if order_index + 1 in self.extract_orders:
                
                    # Timer
                    stopwatch.lap(trace_dict['label'])

                    # Print start of trace
                    print(f" [{data}] Extracting Trace {trace_dict['label']} ...", flush=True)
                    
                    # Extract trace
                    try:
                        spec1d, spec1d_unc, badpix1d = self.extract_trace(data, data_image, trace_map_image, trace_dict, badpix_mask=badpix_mask)
                        
                        # Store result
                        reduced_data[order_index, fiber_index, :, :] = np.array([spec1d, spec1d_unc, badpix1d], dtype=float).T

                        # Print end of trace
                        print(f" [{data}] Extracted Trace {trace_dict['label']} in {round(stopwatch.time_since(trace_dict['label']) / 60, 3)} min", flush=True)

                    except:
                        print(f"Warning! Could not extract trace [{trace_dict['label']}] for observation [{data}]")

        # Plot reduced data
        fig = self.plot_extracted_spectra(data, reduced_data)
        
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
        spec1d = np.nansum(trace_image, axis=0)
        spec1d /= pcmath.weighted_median(spec1d, percentile=0.99)
        spec1d_smooth = np.nansum(pcmath.median_filter2d(trace_image, width=3), axis=0)
        spec1d_smooth /= pcmath.weighted_median(spec1d_smooth, percentile=0.99)
        res_norm = spec1d - spec1d_smooth
        snr = 1 / np.nanstd(res_norm)
        return snr

    @staticmethod
    def plot_extracted_spectra(data, reduced_data):
    
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
                order_num = order_index + 1
                if order_num > n_orders:
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
    def flag_pixels2d(trace_image, badpix_mask, model2d, badpix_threshold=5):

        # Smooth the 2d image
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)

        # Smooth the 2d model
        model2d_smooth = pcmath.median_filter2d(trace_image, width=3)

        # Normalized residuals
        norm_res = (trace_image - model2d_smooth) / trace_image_smooth

        # Flag
        use = np.where(norm_res != 0)
        bad = np.where(np.abs(norm_res) > badpix_threshold * np.nanstd(norm_res[use]))
        if bad[0].size > 0:
            badpix_mask[bad] = 0
            trace_image[bad] = np.nan

    @staticmethod
    def mask_image(image, mask_left, mask_right, mask_top, mask_bottom):
        ny, nx = image.shape
        image[0:mask_bottom, :] = np.nan
        image[ny-mask_top:, :] = np.nan
        image[:, 0:mask_left] = np.nan
        image[:, nx-mask_right:] = np.nan

    @staticmethod
    def compute_extract_aperture(trace_profile_cspline, flux_cutoff=0.05):
        trace_profile_x, trace_profile = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
        trace_profile -= np.nanmin(trace_profile)
        trace_profile /= np.nanmax(trace_profile)
        good = np.where(trace_profile > flux_cutoff)[0]
        height = trace_profile_x[np.max(good)] - trace_profile_x[np.min(good)]
        extract_aperture = [int(np.ceil(height / 2)), int(np.ceil(height / 2))]
        return extract_aperture


    @staticmethod
    def compute_background(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, background_smooth_width=51, background_smooth_poly_order=3):
        
        # Image dims
        ny, nx = trace_image.shape
        
        # Helper array
        yarr = np.arange(ny)
        
        # Empty arrays
        background = np.full(nx, np.nan)
        background_err = np.full(nx, np.nan)

        # Smooth image
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=3)
        
        # Loop over columns
        for x in range(nx):
            
            # Compute trace profile for this column
            P = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x],
                                      trace_profile_cspline(trace_profile_cspline.x),
                                      yarr)
            
            # Normalize to max
            P /= np.nanmax(P)
            
            # Identify regions low in flux
            background_locs = np.where(((yarr < trace_positions[x] - extract_aperture[0]) | (yarr > trace_positions[x] + extract_aperture[1])) & np.isfinite(trace_image_smooth[:, x]))[0]
            
            # Continue if no locs
            if background_locs.size == 0:
                continue
            
            # Compute the median counts behind the trace (use 25th percentile to mitigate injecting negative values)
            background[x] = pcmath.weighted_median(trace_image_smooth[background_locs, x], percentile=0.25)
            
            # Compute background
            if background[x] >= 0 and np.isfinite(background[x]):
                n_good = np.where(np.isfinite(trace_image_smooth[background_locs, x]))[0].size
                if n_good <= 1:
                    background[x] = np.nan
                else:
                    background_err[x] = np.sqrt(background[x] / (n_good - 1))

        # Smooth the background
        background = pcmath.poly_filter(background, width=background_smooth_width, poly_order=background_smooth_poly_order)
        background_err = pcmath.poly_filter(background_err, width=background_smooth_width, poly_order=background_smooth_poly_order)

        # Return
        return background, background_err

    @staticmethod
    def compute_vertical_trace_profile(trace_image, badpix_mask, trace_positions, oversample, spec1d=None, background=None):
        
        # Image dims
        ny, nx = trace_image.shape
        
        # Helpful array
        yarr = np.arange(ny)

        # Background
        if background is None:
            background = np.zeros(nx)
        
        # Create a fiducial high resolution grid centered at zero
        yarr_hr = np.arange(int(np.floor(-ny / 2)), int(np.ceil(ny / 2)) + 1 / oversample, 1 / oversample)
        trace_image_rect = np.full((len(yarr_hr), nx), np.nan)

        # 1d spec info
        if spec1d is None:
            spec1d = np.nansum(trace_image, axis=0)
        spec1d_norm = spec1d / pcmath.weighted_median(spec1d, percentile=0.95)
        
        # Rectify
        for x in range(nx):
            good = np.where(np.isfinite(trace_image[:, x]))[0]
            if good.size >= 3 and spec1d_norm[x] > 0.2:
                col_hr = pcmath.lin_interp(yarr - trace_positions[x], trace_image[:, x], yarr_hr)
                trace_image_rect[:, x] = col_hr - background[x]
                bad = np.where(trace_image_rect[:, x] < 0)[0]
                if bad.size > 0:
                    trace_image_rect[bad, x] = np.nan
                trace_image_rect[:, x] /= np.nansum(trace_image_rect[:, x])
        
        # Compute trace profile
        trace_profile_mean = np.nanmean(trace_image_rect, axis=1)
        
        # Compute cubic spline for profile
        good = np.where(np.isfinite(trace_profile_mean))[0]
        trace_profile_cspline = scipy.interpolate.CubicSpline(yarr_hr[good], trace_profile_mean[good], extrapolate=False)
        
        # Ensure trace profile is centered at zero
        prec = 1000
        yhr = np.arange(trace_profile_cspline.x[0], trace_profile_cspline.x[-1], 1 / prec)
        tphr = trace_profile_cspline(yhr)
        mid = np.nanmean(trace_profile_cspline.x)
        consider = np.where((yhr > mid - 3*oversample) & (yhr < mid + 3*oversample))[0]
        trace_max_pos = yhr[consider[np.nanargmax(tphr[consider])]]
        trace_profile_cspline = scipy.interpolate.CubicSpline(trace_profile_cspline.x - trace_max_pos,
                                                              trace_profile_cspline(trace_profile_cspline.x), extrapolate=False)

        # Trim 1 pixel on each end
        tpx, tpy = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
        trace_profile_cspline = scipy.interpolate.CubicSpline(tpx[2*oversample:len(tpx)-2*oversample], tpy[2*oversample:len(tpx)-2*oversample], extrapolate=False)
        

        # Return
        return trace_profile_cspline

    @staticmethod
    def compute_trace_positions_ccf(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, spec1d=None, window=10, background=None, remove_background=True, trace_pos_poly_order=4):

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
            good = np.where((badpix_mask[:, x] == 1) & np.isfinite(trace_image_no_background_smooth[:, x]) & (yarr > trace_positions[x] - np.ceil(window / 2)) & (yarr < trace_positions[x] + np.ceil(window / 2)))[0]
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

    @staticmethod
    def compute_trace_positions_centroids(trace_image, badpix_mask, trace_positions_estimate=None, extract_aperture=None, trace_pos_poly_order=4, n_iterations=5):

        # The image dimensions
        ny, nx = trace_image.shape
        
        # Helpful arrays
        yarr = np.arange(ny)
        xarr = np.arange(nx)

        # Smooth image
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=3, preserve_nans=False)

        if extract_aperture is None:
            h2 = int(np.ceil(np.nanmax(np.nansum(badpix_mask, axis=0))))
            extract_aperture = [h2, h2]

        # Initiate trace positions
        trace_positions = np.full(nx, np.nan)

        for i in range(n_iterations):

            # Y centroids
            ycen = np.full(nx, np.nan)

            # Loop over columns
            for x in range(nx):
                
                # See if column is even worth looking at
                if i == 0 and trace_positions_estimate is None:
                    good = np.where((badpix_mask[:, x] == 1) & np.isfinite(trace_image_smooth[:, x]))[0]
                elif i == 0 and trace_positions_estimate is not None:
                    good = np.where((badpix_mask[:, x] == 1) & np.isfinite(trace_image_smooth[:, x]) & (yarr > trace_positions_estimate[x] - extract_aperture[0]) & (yarr < trace_positions_estimate[x] + extract_aperture[1]))[0]
                else:
                    good = np.where((badpix_mask[:, x] == 1) & np.isfinite(trace_image_smooth[:, x]) & (yarr > trace_positions[x] - extract_aperture[0]) & (yarr < trace_positions[x] + extract_aperture[1]))[0]
                if good.size <= 3:
                    continue

                # Centroid
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
    def boxcar_extraction(trace_image, badpix_mask, trace_positions, extract_aperture, trace_profile_cspline=None, remove_background=False, background=None, background_err=None, read_noise=0, background_smooth_poly_order=3, background_smooth_width=51):
        """A flavor of boxcar extraction. A single column from the data is a function of y pixels ($S_{y}$), and is modeled as:

            F_{y} = A * P_{y} + B

            where P_{y} is the nominal vertical profile and may be arbitrarily scaled. A is the scaling of the input signal, and B is the background signal which is ignored if remove_background is False or the background variable is set. The sum is performed over a given window defined by extract_aperture. A and B are determined by minimizing the function:

            phi = \sum_{y} w_{y} (S_{y} - F_{y})^{2} = \sum_{y} w_{y} (S_{y} - (A * (\sum_{y} P_{y}) + B))^{2}

            where
            
            w_{y} = M_{y}, where M_{y} is a binary mask (1=good, 0=bad).

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
                P_x = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x], trace_profile_cspline(trace_profile_cspline.x), yarr)
                
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
                w = np.copy(M)

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
            P_x = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x], trace_profile_cspline(trace_profile_cspline.x), yarr)
            
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
            w = np.copy(M)
            w /= np.nansum(w)

            # Least squares
            A = np.nansum(w * P * S) / np.nansum(w * P**2)

            # Final 1d spec
            spec[x] = A * np.nansum(P)
            spec_err[x] = np.sqrt(np.nansum(v) / (np.nansum(M) - 1))
        
        return spec, spec_err, background, background_err
