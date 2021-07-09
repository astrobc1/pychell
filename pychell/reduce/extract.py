
# Python default modules
import os
import copy
import pickle

# Science / Math
import numpy as np
import scipy.interpolate
import scipy.signal

# Graphics
import matplotlib.pyplot as plt

# Pychell modules
import pychell.utils as pcutils
import pychell.maths as pcmath

class SpectralExtractor:
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, mask_left=100, mask_right=100, mask_top=100, mask_bottom=100):
        self.mask_left = mask_left
        self.mask_right = mask_right
        self.mask_top = mask_top
        self.mask_bottom = mask_bottom
        
    ################################################
    #### PRIMARY METHOD TO EXTRACT ENTIRE IMAGE ####
    ################################################

    def extract_image(self, reducer, data, image_num):
        
        # Initialize a timer
        stopwatch = pcutils.StopWatch()
        print(f"Extracting Image {image_num} of {len(reducer.data['science'])} ... [{data}] ...", flush=True)
        
        # Load the full frame raw image
        data_image = data.parse_image()
        
        # The image dimensions
        ny, nx = data_image.shape
        
        # Load the order map
        trace_map_image, orders_list = data.order_map.load_map_image(), data.order_map.orders_list
        
        # The number of echelle orders, possibly composed of multiple traces.
        n_orders = len(orders_list)
        n_traces = len(orders_list[0])
        
        # Mask edge pixels as nan (not an actual crop)
        self.mask_image(data_image)
        
        # Also flag regions in between orders
        bad = np.where(~np.isfinite(trace_map_image))
        if bad[0].size > 0:
            data_image[bad] = np.nan
            
        # Default storage is an HDUList of length=n_orders.
        # reduced_data = np.full((n_orders, nx, 3), np.nan)
        reduced_data = np.full((n_orders, n_traces, nx, 3), np.nan)
        traces = []
        
        # Loop over orders, possibly multi-trace
        for order_index, single_order_list in enumerate(orders_list):
            
            stopwatch.lap(order_index)
            print(f"[{data}] Extracting Order {order_index + 1} of {n_orders} ...", flush=True)
            
            # Orders are composed of multiple traces
            if len(single_order_list) > 1:
                
                for sub_trace_index, single_trace_dict in enumerate(single_order_list):
                    
                    stopwatch.lap(sub_trace_index)
                    print('    Extracting Sub Trace ' + str(sub_trace_index + 1) + ' of ' + str(len(single_order_list)) + ' ...')
                    
                    reduced_orders[order_index, sub_trace_index, :, :], boxcar_spectra[order_index, sub_trace_index, :], trace_profile_csplines[order_index, sub_trace_index], y_positions[order_index, sub_trace_index, :] = extract_single_trace(data, data_image, trace_map_image, single_trace_dict, config)
                    
                    print('    Extracted Sub Trace ' + str(sub_trace_index + 1) + ' of ' + str(len(single_order_list)) + ' in ' + str(round(stopwatch.time_since(sub_trace_index), 3)) + ' min ')
                    
            # Orders are composed of single trace
            else:
                
                # Extract
                data_out, trace_out = self.extract_trace(reducer, data, data_image, trace_map_image, single_order_list[0])
                
                # Store
                reduced_data[order_index, 0, :, :] = data_out
                traces.append(trace_out)
                
            print(f"[{data}] Extracted Order {order_index + 1} of {n_orders} in {round(stopwatch.time_since(order_index) / 60, 3)} min", flush=True)

        # Plot reduced data
        self.plot_extracted_spectra(data, reduced_data)
        
        # Save reduced data
        data.parser.save_reduced_orders(data, reduced_data)
        
        # Save trace info
        with open(f"{reducer.output_path}trace{os.sep}{data.base_input_file_noext}_traces.pkl", "wb") as f:
            pickle.dump(traces, f)

    ##############
    #### PLOT ####
    ##############
    
    def plot_extracted_spectra(self, data, reduced_data):
    
        # The numbr of orders and traces
        n_orders = len(reduced_data)
        
        # The number of x pixels
        xpixels = np.arange(len(reduced_data[0, 0, :, 0]))
        
        # Plot settings
        plot_width = 20
        plot_height = 20
        dpi = 250
        n_cols = 3
        n_rows = int(np.ceil(n_orders / n_cols))
        
        # Create a figure
        fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols,
                                  figsize=(plot_width, plot_height), dpi=dpi)
        axarr = np.atleast_2d(axarr)
        
        # For each subplot, plot each single trace
        for row in range(n_rows):
            for col in range(n_cols):
                
                # The order index
                o = n_cols * row + col
                order_num = o + 1
                if order_num > n_orders:
                    continue
                    
                # Views
                # reduced_data = np.full((n_orders, n_traces, nx, 3), np.nan)
                spec1d = reduced_data[o, 0, :, 0]
                badpix1d = reduced_data[o, 0, :, 2]
                
                # Good pixels
                good = np.where(badpix1d == 1)[0]
                if good.size == 0:
                    continue
                
                # Plot the optimally extracted spectrum
                axarr[row, col].plot(xpixels, spec1d / pcmath.weighted_median(spec1d, percentile=0.98), color='black', label='Optimal', lw=0.5)

                # Title
                axarr[row, col].set_title(f"Order {order_num}", fontsize=6)
                
                # Axis labels
                axarr[row, col].set_xlabel('X Pixels', fontsize=6)
                axarr[row, col].set_ylabel('Norm. Flux', fontsize=6)
                axarr[row, col].tick_params(labelsize=6)
        
        # Tight layout
        plt.tight_layout()
        
        # Create a filename
        fname = f"{data.parser.output_path}spectra{os.sep}{data.base_input_file_noext}_{data.target.replace(' ', '_')}_preview.png"
        
        # Save
        plt.savefig(fname)
        plt.close()

    ###############
    #### MISC. ####
    ###############

    def mask_image(self, data_image):
        ny, nx = data_image.shape
        data_image[0:self.mask_bottom, :] = np.nan
        data_image[ny-self.mask_top:, :] = np.nan
        data_image[:, 0:self.mask_left] = np.nan
        data_image[:, nx-self.mask_right:] = np.nan

    def convert_image_to_pe(self, trace_image, detector_props):
        if len(detector_props) == 1:
            trace_image *= detector_props[0]["gain"]
        else:
            trace_image_pe = np.full_like(trace_image, np.nan)
            for detector in detector_props:
                xmin, xmax, ymin, ymax = detector["xmin"], detector["xmax"], detector["ymin"], detector["ymax"]
                trace_image_pe[ymin:ymax+1, xmin:xmax + 1] = trace_image[ymin:ymax+1, xmin:xmax + 1] * detector["gain"]

class OptimalSlitExtractor(SpectralExtractor):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, mask_left=100, mask_right=100, mask_top=100, mask_bottom=100, trace_pos_poly_order=2, n_trace_iterations=3, n_extract_iterations=3, oversample=1, badpix_threshold_2d=100, badpix_threshold_1d=0.5):
        super().__init__(mask_left=mask_left, mask_right=mask_right, mask_top=mask_top, mask_bottom=mask_bottom)
        self.trace_pos_poly_order = trace_pos_poly_order
        self.n_trace_iterations = n_trace_iterations
        self.n_extract_iterations = n_extract_iterations
        self.oversample = oversample
        self.badpix_threshold_2d = badpix_threshold_2d
        self.badpix_threshold_1d = badpix_threshold_1d
        
    #######################################################################
    #### PRIMARY METHOD TO EXTRACT SINGLE TRACE FOR ENTIRE ORDER WIDTH ####
    #######################################################################

    def extract_trace(self, reducer, data, data_image, trace_map_image, trace_dict):
        
        # Stopwatch
        stopwatch = pcutils.StopWatch()
        
        # Convert to PE
        self.convert_image_to_pe(data_image, reducer.spec_module.detector_props)
        
        # dims
        nx = data_image.shape[1]
        xarr = np.arange(nx)
        
        # The order height
        height = trace_dict["height"]
        
        # The current trace position
        trace_positions = np.polyval(trace_dict["pcoeffs"], xarr)
        
        # Standard dark, bias, flat calibration.
        data_image = reducer.pre_calib.calibrate(data, data_image, trace_map_image, trace_dict)
        
        # Create trace_image where only the relevant trace is seen, still ny x nx
        trace_image = np.copy(data_image)
        good = np.where(np.isfinite(trace_image))
        badpix_mask = np.zeros_like(trace_image)
        badpix_mask[good] = 1
        good_trace = np.where(trace_map_image == trace_dict['label'])
        bad_trace = np.where(trace_map_image != trace_dict['label'])
        if bad_trace[0].size > 0:
            trace_image[bad_trace] = np.nan
            badpix_mask[bad_trace] = 0
            
        # Crop the image and mask to limit memory usage going forward
        goody, goodx = np.where(badpix_mask == 1)
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
            
        # Estimate crude background
        background = np.nanmin(trace_image_smooth, axis=0)
        background = pcmath.median_filter1d(background, width=7)
        bad = np.where(background < 0)[0]
        if bad.size > 0:
            background[bad] = np.nan
        
        # Estimate trace positions
        trace_positions = np.full(nx, np.nan)
        for x in range(nx):
            if np.where(np.isfinite(trace_image_smooth[:, x]))[0].size > 5:
                trace_positions[x] = np.nanargmax(trace_image_smooth[:, x])
        trace_positions_smooth = pcmath.median_filter1d(trace_positions, width=5)
        good = np.where(np.isfinite(trace_positions))[0]
        pfit = np.polyfit(xarr[good], trace_positions[good], self.trace_pos_poly_order)
        trace_positions = np.polyval(pfit, xarr)
        
        # Iteratively refine trace positions and profile.
        for i in range(self.n_trace_iterations):
            
            # Trace Profile
            print(f" [{data}] Iteratively Refining Trace profile [{i + 1} / {self.n_trace_iterations}] ...", flush=True)
            trace_profile_cspline = self.compute_trace_profile(trace_image, badpix_mask, trace_positions, background, height)
            
            # Trace Position
            print(f" [{data}] Iteratively Refining Trace positions [{i + 1} / {self.n_trace_iterations}] ...", flush=True)
            trace_positions = self.compute_trace_positions(trace_image, badpix_mask, trace_positions, trace_profile_cspline, background, height)
            
            # Aperture
            aperture = self.compute_aperture(trace_image, badpix_mask, trace_profile_cspline, trace_positions, background)
            
            # Background signal
            print(f" [{data}] Iteratively Refining Background [{i + 1} / {self.n_trace_iterations}] ...", flush=True)
            background, background_err = self.compute_background(trace_image, badpix_mask, trace_profile_cspline, trace_positions, height, aperture)
        
        # Iteratively extract spectrum.
        for i in range(self.n_extract_iterations):
            
            print(f" [{data}] Iteratively Extracting Trace [{i + 1} / {self.n_extract_iterations}] ...", flush=True)
            
            # Optimal extraction
            spec1d, spec1d_unc = self.optimal_extraction(trace_image, badpix_mask, trace_profile_cspline, trace_positions, reducer.spec_module.detector_props, data, aperture, background=background, background_err=background_err)

            # Re-map pixels and flag in the 2d image.
            if i < self.n_extract_iterations - 1:
                self.flag_pixels_post_extraction(trace_image, badpix_mask, trace_profile_cspline, trace_positions, spec1d, spec1d_unc, background, aperture)

        # Final bad pixel flagging
        spec1d_smooth = pcmath.median_filter1d(spec1d, width=3)
        med_val = pcmath.weighted_median(spec1d_smooth, percentile=0.98)
        bad = np.where((np.abs(spec1d_smooth - spec1d) / med_val) > self.badpix_threshold_1d)[0]
        badpix1d = np.ones_like(spec1d)
        if bad.size > 0:
             badpix1d[bad] = 0
             spec1d[bad] = np.nan
             spec1d_unc[bad] = np.nan
            
        # Data out
        data_out = np.array([spec1d, spec1d_unc, badpix1d], dtype=float).T
        
        # Profile information out, add on y_start
        profile_out = (trace_profile_cspline, trace_positions + y_start)
        
        return data_out, profile_out
        
    ##############################################################
    #### HELPERS FOR EXTRACTION TO COMPUTE PRIMARY COMPONENTS ####
    ##############################################################
    
    def compute_trace_profile(self, trace_image, badpix_mask, trace_positions, background, height):
        
        # Image dims
        ny, nx = trace_image.shape
        
        # Helpful array
        yarr = np.arange(ny)
        
        # Create a fiducial high resolution grid centered at zero
        yarr_hr = np.arange(int(-ny / 2), int(ny / 2) + 1, 1 / self.oversample)
        trace_image_rect = np.full((len(yarr_hr), nx), np.nan)
        
        # Remove background and rectify
        for x in range(nx):
            good = np.where(np.isfinite(trace_image[:, x]))[0]
            if good.size >= 3:
                col_hr = pcmath.lin_interp(yarr - trace_positions[x], trace_image[:, x], yarr_hr)
                trace_image_rect[:, x] = col_hr - background[x]
                trace_image_rect[:, x] /= np.nansum(trace_image_rect[:, x])
            else:
                trace_image_rect[:, x] = np.nan
        
        # Fix negatives
        bad = np.where(trace_image_rect < 0)
        if bad[0].size > 0:
            trace_image_rect[bad] = np.nan
        
        # Compute trace profile
        trace_profile = np.nanmedian(trace_image_rect, axis=1)
        
        # Compute cubic spline for profile
        good = np.where(np.isfinite(trace_profile))[0]
        trace_profile_cspline = scipy.interpolate.CubicSpline(yarr_hr[good], trace_profile[good], extrapolate=False)
        
        # Trim 3 pixels on each side
        trace_profile_cspline = scipy.interpolate.CubicSpline(trace_profile_cspline.x[3*self.oversample:-3*self.oversample], trace_profile_cspline(trace_profile_cspline.x[3*self.oversample:-3*self.oversample]), extrapolate=False)
        
        # Ensure trace profile is centered at zero
        prec = 1000
        yhr = np.arange(trace_profile_cspline.x[0], trace_profile_cspline.x[-1], 1 / prec)
        tphr = trace_profile_cspline(yhr)
        mid = np.nanmean(trace_profile_cspline.x)
        consider = np.where((yhr > mid - 8*self.oversample) & (yhr < mid + 8*self.oversample))[0]
        trace_max_pos = yhr[consider[np.nanargmax(tphr[consider])]]
        trace_profile_cspline = scipy.interpolate.CubicSpline(trace_profile_cspline.x - trace_max_pos,
                                                              trace_profile_cspline(trace_profile_cspline.x), extrapolate=False)

        # Return
        return trace_profile_cspline
    
    def compute_trace_positions(self, trace_image, badpix_mask, trace_positions, trace_profile_cspline, background, height):
        
        # The image dimensions
        ny, nx = trace_image.shape
        
        # Helpful arrays
        yarr = np.arange(ny)
        xarr = np.arange(nx)
        
        # Remove background
        trace_image_no_background = np.full_like(trace_image, np.nan)
        for x in range(nx):
            trace_image_no_background[:, x] = trace_image[:, x] - background[x]
            
        
        trace_image_no_background_smooth = pcmath.median_filter2d(trace_image_no_background, width=3, preserve_nans=False)
        
        # Trace profile
        trace_profile = trace_profile_cspline(trace_profile_cspline.x)
        
        # Smooth the image
        bad = np.where(trace_image_no_background_smooth <= 0)
        if bad[0].size > 0:
            trace_image_no_background[bad] = np.nan

        # Cross correlate each data column with the trace profile estimate
        y_positions_xc = np.full(nx, np.nan)
        
        # Compute the boxcar spectrum
        spec1d_boxcar = np.nansum(trace_image_no_background_smooth, axis=0)
        spec1d_boxcar = pcmath.median_filter1d(spec1d_boxcar, width=3)
        spec1d_boxcar /= pcmath.weighted_median(spec1d_boxcar, percentile=0.95)
        
        # Loop over columns
        for x in range(nx):
            
            # See if column is even worth looking at
            good = np.where((badpix_mask[:, x] == 1) & np.isfinite(trace_image_no_background_smooth[:, x]))[0]
            if good.size <= 3 or spec1d_boxcar[x] < 0.2:
                continue
            
            # Define CCF shifts for this column
            lags = np.arange(trace_positions[x] - height / 2,
                             trace_positions[x] + height / 2 + 1)
            
            # Normalize data column to 1
            data_x = trace_image_no_background_smooth[:, x] / np.nanmax(trace_image_no_background_smooth[:, x])
            
            # Perform CCF
            ccf = pcmath.cross_correlate2(yarr, data_x, trace_profile_cspline.x, trace_profile, lags)
            
            # Bias the ccf
            ccf *= np.exp(-1 * (np.arange(ccf.size) - height / 2)**2 / (2 * lags.size**2)*3)
            
            # Normalize to max=1
            ccf /= np.nanmax(ccf)
            
            # Check if it is useful
            good = np.where(np.isfinite(ccf) & (ccf > 0.3))[0]
            if good.size <= 3:
                continue
            
            # Fit ccf
            pfit = np.polyfit(lags[good], ccf[good], 2)
                
            # Store the nominal location
            y_positions_xc[x] = -1 * pfit[1] / (2 * pfit[0])
        
        # Smooth the deviations
        y_positions_xc_smooth = pcmath.median_filter1d(y_positions_xc, width=3)
        bad = np.where((np.abs(y_positions_xc - y_positions_xc_smooth) > 0.5) | (np.abs(trace_positions - y_positions_xc) > 5))[0]
        if bad.size > 0:
            y_positions_xc[bad] = np.nan
        good = np.where(np.isfinite(y_positions_xc))[0]
        
        # Fit with a polynomial and generate
        if good.size > nx / 6:
            pfit = np.polyfit(xarr[good], y_positions_xc[good], self.trace_pos_poly_order)
            trace_positions = np.polyval(pfit, xarr)
        
        # Return
        return trace_positions
    
    def compute_background(self, trace_image, badpix_mask, trace_profile_cspline, trace_positions, height, aperture):
        
        # Image dims
        ny, nx = trace_image.shape
        
        # Helper array
        yarr = np.arange(ny)
        
        # Empty arrays
        background = np.full(nx, np.nan)
        background_err = np.full(nx, np.nan)
        
        # Loop over columns
        for x in range(nx):
            
            # Compute trace profile for this column
            P = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x],
                                      trace_profile_cspline(trace_profile_cspline.x),
                                      yarr)
            
            # Normalize to max
            P /= np.nanmax(P)
            
            # Identify regions low in flux
            background_locs = np.where(((yarr < trace_positions[x] - aperture / 2) | (yarr > trace_positions[x] + aperture / 2)) & np.isfinite(trace_image[:, x]))[0]
            
            if background_locs.size == 0:
                continue
            
            # Compute the average counts behind the trace
            background[x] = np.nanmedian(trace_image[background_locs, x])
            
            # Check if negative
            if background[x] <= 0 or ~np.isfinite(background[x]):
                background[x] = np.nan
                background_err[x] = np.nan
            else:
                # Error according to Poisson stats
                background_err[x] = np.sqrt(background[x] / (background_locs.size - 1))
                
        # Savgol filter
        #background = scipy.signal.savgol_filter(background, window_length=7, polyorder=2)
        #background_err = scipy.signal.savgol_filter(background, window_length=7, polyorder=2)
        
        # Return
        return background, background_err
    
    ###################################
    #### ACTUAL OPTIMAL EXTRACTION ####
    ###################################
    
    def optimal_extraction(self, trace_image, badpix_mask, trace_profile_cspline, trace_positions, detector_props, data, aperture, dark_subtraction=False, background=None, background_err=None):

        # Image dims
        ny, nx = trace_image.shape
        
        # Exposure time
        exp_time = data.parser.parse_itime(data)

        # Storage arrays
        spec = np.full(nx, fill_value=np.nan, dtype=np.float64)
        spec_unc = np.full(nx, fill_value=np.nan, dtype=np.float64)
        
        # Helper array
        yarr = np.arange(ny)

        # Loop over columns
        for x in range(nx):
            
            # Views
            badpix_x = np.copy(badpix_mask[:, x])
            data_x = trace_image[:, x] - background[x]
            
            # Flag negative values after sky subtraction
            bad = np.where(data_x < 0)[0]
            if bad.size > 0:
                badpix_x[bad] = 0
                
            # Check if column is worth extracting
            if np.nansum(badpix_x) <= 1:
                continue
            
            # Effective read noise
            eff_read_noise = self.compute_read_noise(detector_props, x, trace_positions[x],
                                                     exp_time, dark_subtraction=dark_subtraction)
            
            # Shift Trace Profile
            P = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x],
                                      trace_profile_cspline(trace_profile_cspline.x),
                                      yarr)
            
            # Determine which pixels to use from the trace
            good = np.where((yarr >= trace_positions[x] - aperture / 2) & (yarr <= trace_positions[x] + aperture / 2))[0]
            P_use = P[good]
            data_use = data_x[good]
            badpix_use = badpix_x[good]
            P_use /= np.nansum(P_use)
            
            # Variance
            var_use = eff_read_noise**2 + data_use + background[x] + background_err[x]**2
            
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
    
    def flag_pixels_post_extraction(self, trace_image, badpix_mask, trace_profile_cspline, trace_positions, spec1d, spec1d_unc, background, aperture):
        
        # Image dims
        ny, nx = trace_image.shape
        
        # Model array
        residuals = np.full_like(trace_image, np.nan)
        
        # Helper array
        yarr = np.arange(ny)
        
        # Smooth 1d spectrum
        spec1d_smooth = pcmath.median_filter1d(spec1d, width=3)
        
        # Remap 1d sepctrum into 2d space
        for x in range(nx):
            
            # See if column is useful
            good = np.where(np.isfinite(trace_image[:, x]) & (badpix_mask[:, x] == 1))[0]
            if good.size < 2:
                continue
            
            # Generate trace profile
            P = pcmath.cspline_interp(trace_profile_cspline.x + trace_positions[x], trace_profile_cspline(trace_profile_cspline.x), yarr)
            
            # Get useful area
            good = np.where((yarr >= trace_positions[x] - aperture / 2) & (yarr <= trace_positions[x] + aperture / 2))[0]
            if good.size <= 1:
                continue
            P_use = P[good]
            P_use /= np.nansum(P_use)
            
            # Remap into 2d space
            residuals[good, x] = (P_use * spec1d_smooth[x] + background[x]) - trace_image[good, x]
        
        # Smooth the residuals
        residuals_smooth = pcmath.median_filter2d(residuals, width=3)
        bad = np.where(residuals_smooth == 0)
        if bad[0].size > 0:
            residuals_smooth[bad] = np.nan
        deviations = (residuals - residuals_smooth)
        n_use = np.where(np.isfinite(deviations))[0].size
        rms = np.nansum(deviations**2 / n_use)**0.5
        bad = np.where(np.abs(deviations) > 11*rms)
        
        if bad[0].size > 0:
            badpix_mask[bad] = 0
            trace_image[bad] = np.nan
        
    ###############
    #### MISC. ####
    ###############
    
    def compute_aperture(self, trace_image, badpix_mask, trace_profile_cspline, trace_positions, background):
        trace_profile_x, trace_profile = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
        trace_profile /= np.nanmax(trace_profile)
        good = np.where(trace_profile > 0.025)[0]
        x_start, x_end = trace_profile_x[np.min(good)], trace_profile_x[np.max(good)]
        return x_end - x_start
        
    def compute_read_noise(self, detector_props, x, y, exp_time, dark_subtraction=False):
    
        # Get the detector
        detector = self.get_detector(detector_props, x, y)
    
        # Detector read noise
        if dark_subtraction:
            eff_read_noise = detector['read_noise']
        else:
            return detector['read_noise'] + detector['dark_current'] * exp_time
        
    def get_detector(self, detector_props, x, y):
        if len(detector_props) == 1:
            return detector_props[0]
        for detector in detector_props:
            if (detector['xmin'] < x < detector['xmin']) and (detector['ymin'] < y < detector['ymin']):
                return detector
            
        return ValueError("Point (" + str(x) + str(y) + ") not part of any detector !")