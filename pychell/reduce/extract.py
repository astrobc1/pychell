
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
        bad = np.where(np.abs(norm_res) > badpix_threshold * np.nanstd(norm_res))
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
    def compute_trace_profile(trace_image, badpix_mask, trace_positions, background, remove_background, oversample):
        
        # Image dims
        ny, nx = trace_image.shape
        
        # Helpful array
        yarr = np.arange(ny)
        
        # Create a fiducial high resolution grid centered at zero
        yarr_hr = np.arange(int(np.floor(-ny / 2)), int(np.ceil(ny / 2)) + 1, 1 / oversample)
        trace_image_rect = np.full((len(yarr_hr), nx), np.nan)
        
        # Remove background and rectify
        for x in range(nx):
            good = np.where(np.isfinite(trace_image[:, x]))[0]
            if good.size >= 3:
                col_hr = pcmath.lin_interp(yarr - trace_positions[x], trace_image[:, x], yarr_hr)
                trace_image_rect[:, x] = col_hr - background[x]
                bad = np.where(trace_image_rect[:, x] < 0)[0]
                if bad.size > 0:
                    trace_image_rect[bad, x] = 0
                trace_image_rect[:, x] /= np.nansum(trace_image_rect[:, x])
            else:
                trace_image_rect[:, x] = np.nan
        
        # Compute trace profile
        trace_profile = np.nanmedian(trace_image_rect, axis=1)
        
        # Compute cubic spline for profile
        good = np.where(np.isfinite(trace_profile))[0]
        trace_profile_cspline = scipy.interpolate.CubicSpline(yarr_hr[good], trace_profile[good], extrapolate=False)
        
        # Ensure trace profile is centered at zero
        prec = 1000
        yhr = np.arange(trace_profile_cspline.x[0], trace_profile_cspline.x[-1], 1 / prec)
        tphr = trace_profile_cspline(yhr)
        mid = np.nanmean(trace_profile_cspline.x)
        consider = np.where((yhr > mid - 8*oversample) & (yhr < mid + 8*oversample))[0]
        trace_max_pos = yhr[consider[np.nanargmax(tphr[consider])]]
        trace_profile_cspline = scipy.interpolate.CubicSpline(trace_profile_cspline.x - trace_max_pos,
                                                              trace_profile_cspline(trace_profile_cspline.x), extrapolate=False)

        # Further remove the minimum of the trace profile
        trace_profile = trace_profile_cspline(trace_profile_cspline.x)
        trace_profile -= np.nanmin(trace_profile)
        trace_profile_cspline = scipy.interpolate.CubicSpline(trace_profile_cspline.x,
                                                              trace_profile, extrapolate=False)

        # Return
        return trace_profile_cspline

    @staticmethod
    def compute_trace_positions(trace_image, badpix_mask, trace_profile_cspline, trace_positions_estimate, trace_pos_refine_window=10, background=None, remove_background=True, trace_pos_poly_order=4):

        # The image dimensions
        ny, nx = trace_image.shape
        
        # Helpful arrays
        yarr = np.arange(ny)
        xarr = np.arange(nx)
        
        # Remove background
        if remove_background:
            trace_image_no_background = np.full_like(trace_image, np.nan)
            for x in range(nx):
                trace_image_no_background[:, x] = trace_image[:, x] - background[x]
        else:
            trace_image_no_background = trace_image

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

        # Define CCF shifts for this column
        height = np.nanmax(np.nansum(badpix_mask, axis=0))

        # Normalize trace profile to 1
        trace_profile = trace_profile_cspline(trace_profile_cspline.x)
        trace_profile /= np.nanmax(trace_profile)

        # Loop over columns
        for x in range(nx):
            
            # See if column is even worth looking at
            good = np.where((badpix_mask[:, x] == 1) & np.isfinite(trace_image_no_background_smooth[:, x]))[0]
            if good.size <= 3 or spec1d_boxcar[x] < 0.2:
                continue
            
            # Normalize data column to 1
            data_x = trace_image_no_background_smooth[:, x] / np.nanmax(trace_image_no_background_smooth[:, x])
            
            # CCF lags
            lags = np.arange(trace_positions_estimate[x] - trace_pos_refine_window, trace_positions_estimate[x] + trace_pos_refine_window)
            
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
        else:
            trace_positions = trace_positions_estimate

        
        # Return
        return trace_positions


# Optimal
from pychell.reduce.optimal import OptimalExtractor
#from pychell.reduce.optimaltilted import TiltedOptimalExtractor

# Slit decomp
try:
    from pychell.reduce.decomp import DecompExtractor
except:
    print("Warning! Could not import pyreduce")

# Deconv 2d
from pychell.reduce.deconv2d import Deconv2dExtractor
