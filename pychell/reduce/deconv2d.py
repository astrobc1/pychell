# Python default modules
import os
import copy
import pickle

# Science / Math
import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.sparse as sparse
from astropy.io import fits
import scipy.sparse.linalg as slinalg

# LLVM
from numba import jit, njit

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

class GaussianDeconv2dExtractor(SpectralExtractor):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05,
                 trace_pos_poly_order=4, oversample=1,
                 n_trace_refine_iterations=3, n_extract_iterations=3, trace_pos_refine_window=3,
                 badpix_threshold=5,
                 extract_orders=None,
                 extract_aperture=None,
                 theta=None, q=None, sigma=None,
                 chunk_width=100, distrust_width=20):

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
        self.q = q
        self.sigma = sigma
        self.theta = theta
        self.chunk_width = chunk_width
        self.distrust_width = distrust_width
        
        
    #######################################################################
    #### PRIMARY METHOD TO EXTRACT SINGLE TRACE FOR ENTIRE ORDER WIDTH ####
    #######################################################################

    def extract_trace(self, data, trace_image, trace_map_image, trace_dict, badpix_mask=None, read_noise=None):
        sigma = self.sigma[:, int(trace_dict['label'])-1] if self.sigma is not None else None
        theta = self.theta[:, int(trace_dict['label'])-1] if self.theta is not None else None
        q = self.q[:, int(trace_dict['label'])-1] if self.q is not None else None
        return self._extract_trace(data, trace_image, trace_map_image, trace_dict, badpix_mask, read_noise, self.remove_background, self.background_smooth_poly_order, self.background_smooth_width, self.flux_cutoff, self.trace_pos_poly_order, self.oversample, self.n_trace_refine_iterations, self.n_extract_iterations, self.trace_pos_refine_window, self.badpix_threshold, self._extract_aperture, sigma, q, theta, self.chunk_width, self.distrust_width)
    
    @staticmethod
    def compute_model2d(trace_image, badpix_mask, spec1d, trace_positions, sigma, q, theta, extract_aperture, background, remove_background):

        # Dims
        ny, nx = trace_image.shape

        # Number of chunks
        chunks = SPExtractor.generate_chunks(trace_image, badpix_mask, chunk_width=200)

        # Copy input spectrum and fix nans (inside bounds)
        f = np.copy(spec1d)
        xarr = np.arange(nx)
        good = np.where(np.isfinite(f))[0]
        bad = np.where(~np.isfinite(f))[0]
        f[bad] = pcmath.lin_interp(xarr[good], f[good], xarr[bad])

        # Stitch points
        model2d = np.full((ny, nx, len(chunks)), np.nan)

        # Loop over chunks
        for i in range(len(chunks)):
            xxi, xxf = chunks[i][0], chunks[i][1]
            nnnx = xxf - xxi + 1
            goodyy, _ = np.where(badpix_mask[:, xxi:xxf+1])
            yyi, yyf = goodyy.min(), goodyy.max()
            nnny = yyf - yyi + 1
            S = trace_image[yyi:yyf+1, xxi:xxf+1]
            tp = trace_positions[xxi:xxf+1] - yyi
            xarr_aperture = np.arange(nnnx)
            A = GaussianDeconv2dExtractor.generate_A(S, xarr_aperture, tp, sigma[xxi:xxf+1], q[xxi:xxf+1], theta[xxi:xxf+1], extract_aperture)
            Aflat = A.reshape((nnny*nnnx, nnnx))
            Sflat = S.flatten()
            model2d[yyi:yyf+1, xxi:xxf+1, i] = np.matmul(Aflat, f[xxi:xxf+1]).reshape((nnny, nnnx))
            if remove_background:
                model2d[yyi:yyf+1, xxi:xxf+1, i] += np.outer(np.ones(nnny), background[xxi:xxf+1])
            
            model2d[:, xxi:xxi+20, i] = np.nan
            model2d[:, xxf-20:xxf, i] = np.nan

        model2d = np.nanmean(model2d, axis=2)
        return model2d

    @staticmethod
    @njit(nogil=True)
    def generate_A(image, xarr_aperture, trace_positions, sigma, q, theta, extract_aperture):

        # Image dims
        ny, nx = image.shape

        # Aperture size
        nl, nu = extract_aperture[0], extract_aperture[1]
        aperture_size_x = int((nu + nl + 1) * len(xarr_aperture) / nx)
        aperture_size_y = int((nu + nl + 1) * len(xarr_aperture) / nx)
        if aperture_size_x % 2 != 1:
            aperture_size_x += 1
        if aperture_size_y % 2 != 1:
            aperture_size_y += 1

        # Initialize A
        n_apertures = len(xarr_aperture)
        A = np.zeros(shape=(ny, nx, n_apertures), dtype=np.float64)

        # Helpful arrays
        xarr_detector = np.arange(nx)
        yarr_detector = np.arange(ny)
        yarr_aperture = trace_positions

        # Loops!
        for m in range(n_apertures):
            for i in range(nx):
                for j in range(ny):

                    # (x, y) of the center of the aperture, relative to dims of image
                    xkc = xarr_aperture[m]
                    ykc = yarr_aperture[m]

                    # Coordinates
                    # (x, y) of the psf, relative to dims of image
                    xl = xarr_detector[i]
                    yl = yarr_detector[j]

                    # Diffs
                    dx = xl - xkc
                    dy = yl - ykc
                    if np.abs(dx) > aperture_size_x or np.abs(dy) > aperture_size_y:
                        continue

                    # PSF params
                    _theta = theta[m]
                    _sigma = sigma[m]
                    _q = q[m]

                    # Tilted Coordinates relative to center of aperture
                    xp = dx * np.sin(_theta) - dy * np.cos(_theta)
                    yp = dx * np.cos(_theta) + dy * np.sin(_theta)

                    # Compute PSF
                    A[j, i, m] = np.exp(-0.5 * ((xp / _sigma)**2 + (yp / (_q * _sigma))**2))

        # Normalize each aperture
        for m in range(n_apertures):

            # Normalize each aperture
            s = np.sum(A[:, :, m])
            if s != 0:
                A[:, :, m] /= s

        return A

    @staticmethod
    def generate_chunks(trace_image, badpix_mask, chunk_width=150):

        # Preliminary info
        goody, goodx = np.where(badpix_mask)
        xi, xf = goodx.min(), goodx.max()
        nnx = xf - xi + 1
        yi, yf = goody.min(), goody.max()
        nny = yf - yi + 1
        chunk_width = np.min([chunk_width, 200])
        chunks = []
        chunks.append((xi, xi + chunk_width))
        for i in range(1, int(2 * np.ceil(nnx / chunk_width))):
            vi = chunks[i-1][1] - int(chunk_width / 2)
            vf = np.min([vi + chunk_width, xf])
            chunks.append((vi, vf))
            if vf == xf:
                break
        return chunks

    @staticmethod
    def bin_spec1d(spec1dhr, oversample):
        nx = int(len(spec1dhr) / oversample)
        return np.nansum(spec1dhr.reshape((nx, oversample)), axis=1)


    @staticmethod
    def fix_nans_2d(trace_image, badpix_mask, trace_positions, extract_aperture, n_iterations=10):
        ny, nx = trace_image.shape
        yarr = np.arange(ny)
        trace_image_out = np.copy(trace_image)
        bad0 = np.where(~np.isfinite(trace_image))
        for i in range(n_iterations):
            bady, badx = np.where(~np.isfinite(trace_image_out))
            if len(bady) == 0:
                break
            for j in range(len(bady)):
                x, y = badx[j], bady[j]
                xi, xf = np.max([x - 1, 0]), np.min([x + 1, nx - 1])
                yi, yf = np.max([y - 1, 0]), np.min([y + 1, ny - 1])
                trace_image_out[bady[j], badx[j]] = np.nanmean(trace_image_out[yi:yf+1, xi:xf+1])

        for x in range(nx):
            bad = np.where((yarr <= trace_positions[x] - extract_aperture[0]) | (yarr >= trace_positions[x] + extract_aperture[1]))
            trace_image_out[bad, x] = 0
        
        return trace_image_out


class LSQRExtractor(GaussianDeconv2dExtractor):

    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05,
                 trace_pos_poly_order=4, oversample=1,
                 n_trace_refine_iterations=3, n_extract_iterations=3, trace_pos_refine_window=3,
                 badpix_threshold=5,
                 extract_orders=None,
                 extract_aperture=None,
                 theta=None, q=None, sigma=None,
                 chunk_width=100, distrust_width=20):

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
        self.q = q
        self.sigma = sigma
        self.theta = theta
        self.chunk_width = chunk_width
        self.distrust_width = distrust_width

    @staticmethod
    def _extract_trace(data, image, trace_map_image, trace_dict, badpix_mask, read_noise=None, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05, trace_pos_poly_order=4, oversample=1, n_trace_refine_iterations=3, n_extract_iterations=3, trace_pos_refine_window=5, badpix_threshold=5, _extract_aperture=None, sigma=None, q=None, theta=None, chunk_width=200, distrust_width=20):
        
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
            background = np.nanmin(trace_image_chunk, axis=0)
            background = pcmath.poly_filter(background, width=background_smooth_width, poly_order=background_smooth_poly_order)
            background_err = np.sqrt(background)
        else:
            background = None
            background_err = None

        # Extract Aperture
        if _extract_aperture is None:
            extract_aperture = LSQR2dExtractor.compute_extract_aperture(trace_profile_cspline)
        else:
            extract_aperture = _extract_aperture

        # Iteratively refine trace positions.
        print(f" [{data}, {trace_dict['label']}] Iteratively Refining Trace positions ...", flush=True)
        for i in range(3):

            # Update trace profile
            trace_profile_cspline = GaussianDeconv2dExtractor.compute_vertical_trace_profile(trace_image, badpix_mask, trace_positions, 4, None, background=background)

            # Update trace positions
            trace_positions = GaussianDeconv2dExtractor.compute_trace_positions_ccf(trace_image, badpix_mask, trace_profile_cspline, trace_positions, extract_aperture, spec1d=None, window=trace_pos_refine_window, background=background, remove_background=remove_background, trace_pos_poly_order=trace_pos_poly_order)
        
        # Iteratively extract spectrum
        for i in range(n_extract_iterations):
            
            print(f" [{data}] Iteratively Extracting Trace [{i + 1} / {n_extract_iterations}] ...", flush=True)
            
            # Deconv extraction
            spec1d, spec1d_unc = LSQR2dExtractor.deconv2dextraction(trace_image, badpix_mask, trace_positions, sigma, q, theta, extract_aperture, background, remove_background, read_noise, chunk_width, distrust_width, oversample)

            # Re-map pixels and flag in the 2d image.
            if i < n_extract_iterations - 1:

                # Create the 2d model
                model2d = LSQR2dExtractor.compute_model2d(trace_image, badpix_mask, spec1d, trace_positions, sigma, q, theta, extract_aperture, background, remove_background)

                # Flag bad pixels
                LSQR2dExtractor.flag_pixels2d(trace_image, badpix_mask, model2d, badpix_threshold)

        # badpix mask
        badpix1d = np.ones(nx)
        bad = np.where(~np.isfinite(spec1d) | (spec1d <= 0) | ~np.isfinite(spec1d_unc) | (spec1d_unc <= 0))[0]
        if bad.size > 0:
            spec1d[bad] = np.nan
            spec1d_unc[bad] = np.nan
            badpix1d[bad] = 0
        
        return spec1d, spec1d_unc, badpix1d


    @staticmethod
    def deconv2dextraction(trace_image, badpix_mask, trace_positions, sigma, q, theta, extract_aperture, background=None, remove_background=True, read_noise=0, chunk_width=200, distrust_width=20, oversample=1, damp=0.1):
    
        # Copy input
        trace_image_cp = np.copy(trace_image)
        badpix_mask_cp = np.copy(badpix_mask)

        # Dims
        ny, nx = trace_image_cp.shape

        # Helpful array
        yarr = np.arange(ny)

        # Remove background
        if remove_background:
            for x in range(nx):
                trace_image_cp[:, x] -= background[x]

        # Flag negative pixels
        bad = np.where(trace_image_cp < 0)
        if bad[0].size > 0:
            trace_image_cp[bad] = np.nan
            badpix_mask_cp[bad] = 0

        # Now set all nans to zero
        bad = np.where(~np.isfinite(trace_image_cp) | (badpix_mask_cp == 0))
        if bad[0].size > 0:
            trace_image_cp[bad] = 0

        # Chunks
        chunks = LSQR2dExtractor.generate_chunks(trace_image, badpix_mask_cp, chunk_width=chunk_width)
        n_chunks = len(chunks)

        # Outputs
        spec1d = np.full((nx, n_chunks), np.nan)
        spec1dt = np.full((nx, n_chunks), np.nan)
        spec1dhr = np.full((nx*oversample, n_chunks), np.nan)
        spec1dthr = np.full((nx*oversample, n_chunks), np.nan)
            
        # Loop over and extract chunks
        for i in range(n_chunks):

            # X pixel bounds for this chunk
            xi, xf = chunks[i][0], chunks[i][1]
            nnx = xf - xi + 1

            # Y pixel bounds  for this chunk
            goody, _ = np.where(badpix_mask_cp[:, xi:xf+1])
            yi, yf = goody.min(), goody.max()
            nny = yf - yi + 1

            # Crop image and mask to this chunk
            S = np.copy(trace_image_cp[yi:yf+1, xi:xf+1])
            M = np.copy(badpix_mask_cp[yi:yf+1, xi:xf+1])

            # Crop aperture params to this chunk and upsample
            tp = trace_positions[xi:xf+1] - yi
            _sigma = sigma[xi:xf+1]
            _q = q[xi:xf+1]
            _theta = theta[xi:xf+1]
            xarr_aperture = np.arange(-0.5 + 0.5 / oversample, nnx - 1 + 0.5 - 0.5 / oversample + 1 / oversample, 1 / oversample)
            xarr = np.arange(nnx)
            tp_hr = scipy.interpolate.interp1d(xarr, tp, fill_value="extrapolate")(xarr_aperture)
            sigma_hr = scipy.interpolate.interp1d(xarr, _sigma, fill_value="extrapolate")(xarr_aperture)
            q_hr = scipy.interpolate.interp1d(xarr, _q, fill_value="extrapolate")(xarr_aperture)
            theta_hr = scipy.interpolate.interp1d(xarr, _theta, fill_value="extrapolate")(xarr_aperture)

            # Generate Aperture tensor for this chunk
            A = LSQR2dExtractor.generate_A(S, xarr_aperture, tp_hr, sigma_hr, q_hr, theta_hr, extract_aperture)

            # Prep inputs for sparse extraction
            Aflat = A.reshape((nny*nnx, len(xarr_aperture)))
            S = SPExtractor.fix_nans_2d(S, M, tp, extract_aperture)

            W = np.zeros((nny, nnx))
            for k in range(nny):
                for j in range(nnx):
                    W[k, j] = 1 / np.sqrt(S[k, j] + read_noise**2)
            bad = np.where(~np.isfinite(W))
            W[bad] = 0
            S[bad] = 0
            Wbig = np.diag(W.flatten())
            Sflat = S.flatten()
            Aprimeflat = np.matmul(Wbig, Aflat)
            Sprimeflat = Sflat * np.diagonal(Wbig)


            x0 = np.nansum(S, axis=0)
            x0 = scipy.interpolate.interp1d(xarr, x0, fill_value="extrapolate")(xarr_aperture)
            syhr = slinalg.lsqr(Aprimeflat, Sprimeflat, damp=damp, x0=x0)[0]

            # Bin back to detector grid
            sy = LSQR2dExtractor.bin_spec1d(syhr, oversample)

            # Mask ends
            sy[0:distrust_width] = np.nan
            sy[-distrust_width:] = np.nan

            # Store results
            spec1d[xi:xf+1, i] = sy

        # Final trim of edges
        spec1d[0:xi+distrust_width, 0] = np.nan
        spec1d[xf-distrust_width:, -1] = np.nan
        spec1d[0:xi+distrust_width, 0] = np.nan
        spec1d[xf-distrust_width:, -1] = np.nan

        # Correct negatives in reconvolved spectrum
        bad = np.where(spec1d < 0)[0]
        spec1d[bad] = np.nan

        # Average each chunk
        spec1d = np.nanmean(spec1d, axis=1)
        spec1d_unc = np.sqrt(spec1d)

        # Return
        return spec1d, spec1d_unc
        

class SPExtractor(GaussianDeconv2dExtractor):

    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05,
                 trace_pos_poly_order=4, oversample=1,
                 n_trace_refine_iterations=3, n_extract_iterations=3, trace_pos_refine_window=3,
                 badpix_threshold=5,
                 extract_orders=None,
                 extract_aperture=None,
                 theta=None, q=None, sigma=None,
                 chunk_width=100, distrust_width=20):

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
        self.q = q
        self.sigma = sigma
        self.theta = theta
        self.chunk_width = chunk_width
        self.distrust_width = distrust_width


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

        # Convolution matrices
        conv_matrices = np.empty((n_orders, n_traces, nx, nx), dtype=object)

        # Loop over fibers
        for fiber_index, order_map in enumerate(data.order_maps):

            # Which orders
            if self.extract_orders is None:
                self.extract_orders = np.arange(1, len(order_map.orders_list) + 1).astype(int)
            
            # Alias orders list
            orders_list = order_map.orders_list
        
            # A trace mask
            trace_map_image = recipe.tracer.gen_image(orders_list, ny, nx, xrange=recipe.xrange, poly_mask_top=recipe.poly_mask_top, poly_mask_bottom=recipe.poly_mask_bottom)
        
            # Mask edge pixels as nan
            self.mask_image(data_image, recipe.xrange, recipe.poly_mask_bottom, recipe.poly_mask_top)
        
            # Loop over orders, possibly multi-trace
            for order_index, trace_dict in enumerate(orders_list):

                if order_index + 1 in self.extract_orders:
                
                    # Timer
                    stopwatch.lap(trace_dict['label'])

                    # Print start of trace
                    print(f" [{data}] Extracting Trace {trace_dict['label']} ...", flush=True)
                    
                    # Extract trace
                    try:
                        spec1d, spec1d_unc, badpix1d, R = self.extract_trace(data, data_image, trace_map_image, trace_dict, badpix_mask=badpix_mask)
                        
                        # Store result
                        reduced_data[order_index, fiber_index, :, :] = np.array([spec1d, spec1d_unc, badpix1d], dtype=float).T

                        # Reformat the convolution matrices to save space
                        breakpoint()
                        #R_out = np.

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

        # Reformat the convolution matrices to save space
        #for 
        #hdul = fits.HDUList([fits.PrimaryHDU(reduced_data, header=data.header), fits.])
        matplotlib.use("MacOSX")
        #for i in range(nx):
        #    plt.plot(np.arange(nx) - i, R[i, :])
        #plt.show()
        breakpoint()
        hdu.writeto(fname, overwrite=True, output_verify='ignore')


    @staticmethod
    def _extract_trace(data, image, trace_map_image, trace_dict, badpix_mask, read_noise=None, remove_background=True, background_smooth_poly_order=3, background_smooth_width=51, flux_cutoff=0.05, trace_pos_poly_order=4, oversample=1, n_trace_refine_iterations=3, n_extract_iterations=3, trace_pos_refine_window=5, badpix_threshold=5, _extract_aperture=None, sigma=None, q=None, theta=None, chunk_width=200, distrust_width=20):
        
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
            background = np.nanmin(trace_image_chunk, axis=0)
            background = pcmath.poly_filter(background, width=background_smooth_width, poly_order=background_smooth_poly_order)
            background_err = np.sqrt(background)
        else:
            background = None
            background_err = None

        # Extract Aperture
        if _extract_aperture is None:
            extract_aperture = SPExtractor.compute_extract_aperture(trace_profile_cspline)
        else:
            extract_aperture = _extract_aperture

        # Iteratively refine trace positions.
        print(f" [{data}, {trace_dict['label']}] Iteratively Refining Trace positions ...", flush=True)
        for i in range(10):

            # Update trace profile
            trace_profile_cspline = SPExtractor.compute_vertical_trace_profile(trace_image, badpix_mask, trace_positions, 4, None, background=background)

            # Update trace positions
            trace_positions = SPExtractor.compute_trace_positions_ccf(trace_image, badpix_mask, trace_profile_cspline, extract_aperture, trace_positions_estimate=trace_positions, spec1d=None, ccf_window=trace_pos_refine_window, remove_background=remove_background, background=background, trace_pos_poly_order=trace_pos_poly_order)
        
        # Iteratively extract spectrum
        for i in range(n_extract_iterations):
            
            print(f" [{data}] Iteratively Extracting Trace [{i + 1} / {n_extract_iterations}] ...", flush=True)
            
            # Deconv extraction
            spec1d, spec1d_unc, spec1dt, spec1dt_unc, R = SPExtractor.deconv2dextraction(trace_image, badpix_mask, trace_positions, sigma, q, theta, extract_aperture, background, remove_background, read_noise, chunk_width, distrust_width, oversample)

            # Re-map pixels and flag in the 2d image.
            if i < n_extract_iterations - 1:

                # Create the 2d model
                model2d = SPExtractor.compute_model2d(trace_image, badpix_mask, spec1dt, trace_positions, sigma, q, theta, extract_aperture, background, remove_background)

                # Flag bad pixels
                SPExtractor.flag_pixels2d(trace_image, badpix_mask, model2d, badpix_threshold)

        # badpix mask
        badpix1d = np.ones(nx)
        bad = np.where(~np.isfinite(spec1dt) | (spec1dt <= 0) | ~np.isfinite(spec1dt_unc) | (spec1dt_unc <= 0))[0]
        if bad.size > 0:
            spec1dt[bad] = np.nan
            spec1dt_unc[bad] = np.nan
            badpix1d[bad] = 0
            R[bad, :] = np.nan
        
        return spec1dt, spec1dt_unc, badpix1d, R


    ##############################
    #### 2d DECONV EXTRACTION ####
    ##############################

    @staticmethod
    def deconv2dextraction(trace_image, badpix_mask, trace_positions, sigma, q, theta, extract_aperture, background=None, remove_background=True, read_noise=0, chunk_width=100, distrust_width=20, oversample=1):
    
        # Copy input
        trace_image_cp = np.copy(trace_image)
        badpix_mask_cp = np.copy(badpix_mask)

        # Dims
        ny, nx = trace_image_cp.shape

        # Helpful array
        yarr = np.arange(ny)

        # Remove background
        if remove_background:
            for x in range(nx):
                trace_image_cp[:, x] -= background[x]

        # Flag negative pixels
        bad = np.where(trace_image_cp < 0)
        if bad[0].size > 0:
            trace_image_cp[bad] = np.nan
            badpix_mask_cp[bad] = 0

        # Now set all nans to zero
        bad = np.where(~np.isfinite(trace_image_cp) | (badpix_mask_cp == 0))
        if bad[0].size > 0:
            trace_image_cp[bad] = 0

        # Chunks
        chunks = SPExtractor.generate_chunks(trace_image, badpix_mask_cp, chunk_width=chunk_width)
        n_chunks = len(chunks)

        # Outputs (averaged over chunks before returning)
        spec1d = np.full((nx, n_chunks), np.nan)
        spec1dt = np.full((nx, n_chunks), np.nan)
        R = np.full((nx, nx, n_chunks), np.nan)

        # Loop over and extract chunks
        for i in range(n_chunks):

            # X pixel bounds for this chunk
            xi, xf = chunks[i][0], chunks[i][1]
            nnx = xf - xi + 1

            # Y pixel bounds  for this chunk
            goody, _ = np.where(badpix_mask_cp[:, xi:xf+1])
            yi, yf = goody.min(), goody.max()
            nny = yf - yi + 1

            # Crop image and mask to this chunk
            S = trace_image_cp[yi:yf+1, xi:xf+1]
            M = badpix_mask_cp[yi:yf+1, xi:xf+1]

            # Crop aperture params to this chunk and upsample
            tp = trace_positions[xi:xf+1] - yi
            _sigma = sigma[xi:xf+1]
            _q = q[xi:xf+1]
            _theta = theta[xi:xf+1]
            xarr_aperture = np.arange(-0.5 + 0.5 / oversample, nnx - 1 + 0.5 - 0.5 / oversample + 1 / oversample, 1 / oversample)
            xarr = np.arange(nnx)
            tp_hr = scipy.interpolate.interp1d(xarr, tp, fill_value="extrapolate")(xarr_aperture)
            sigma_hr = scipy.interpolate.interp1d(xarr, _sigma, fill_value="extrapolate")(xarr_aperture)
            q_hr = scipy.interpolate.interp1d(xarr, _q, fill_value="extrapolate")(xarr_aperture)
            theta_hr = scipy.interpolate.interp1d(xarr, _theta, fill_value="extrapolate")(xarr_aperture)

            # Generate Aperture tensor for this chunk
            A = SPExtractor.generate_A(S, xarr_aperture, tp_hr, sigma_hr, q_hr, theta_hr, extract_aperture)

            # Prep inputs for sparse extraction
            Aflat = A.reshape((nny*nnx, len(xarr_aperture)))
            S = SPExtractor.fix_nans_2d(S, M, tp, extract_aperture)
            Sflat = S.flatten()
            # W = np.zeros((nny, nnx))
            # for k in range(nny):
            #     for j in range(nnx):
            #         W[k, j] = A[k, j, :].sum()**2 / (S[k, j] + read_noise**2)
            #bad = np.where(~np.isfinite(W))
            #W[bad] = 0
            #Wbig = np.diag(W.flatten())
            Ninv = np.diag(1 / Sflat)
            bad = np.where(~np.isfinite(Ninv))
            Ninv[bad] = 0

            # Call sparse extraction
            syhr, sythr, _Rhr = SPExtractor.extract_SP2d(Aflat, Sflat, Ninv)

            # Bin back to detector grid
            sy = SPExtractor.bin_spec1d(syhr, oversample)
            syt = SPExtractor.bin_spec1d(sythr, oversample)

            # Mask edge errors
            sy[0:distrust_width] = np.nan
            sy[-distrust_width:] = np.nan
            syt[0:distrust_width] = np.nan
            syt[-distrust_width:] = np.nan
            _Rhr[0:distrust_width, 0:distrust_width] = np.nan
            _Rhr[-distrust_width:, -distrust_width:] = np.nan

            # Store results
            spec1d[xi:xf+1, i] = sy
            spec1dt[xi:xf+1, i] = syt
            spec1d[xi:xf+1, i] = sy
            spec1dt[xi:xf+1, i] = syt
            R[xi:xf+1, xi:xf+1, i] = _Rhr

        # Final trim of edges
        spec1d[0:xi+distrust_width, 0] = np.nan
        spec1d[xf-distrust_width:, -1] = np.nan
        spec1dt[0:xi+distrust_width, 0] = np.nan
        spec1dt[xf-distrust_width:, -1] = np.nan
        R[:, 0:xi+distrust_width, 0] = np.nan
        R[:, xf-distrust_width:, -1] = np.nan

        # Correct negatives in reconvolved spectrum
        bad = np.where(spec1dt < 0)[0]
        spec1dt[bad] = np.nan

        # Average each chunk
        spec1d = np.nanmean(spec1d, axis=1)
        spec1d_unc = np.sqrt(spec1d)
        spec1dt = np.nanmean(spec1dt, axis=1)
        spec1dt_unc = np.sqrt(spec1dt)

        # For R, make sure each psf is normalized to sum=1
        RR = np.nanmean(R, axis=2)
        for i in range(nx):
            RR[i, :] /= np.nansum(RR[i, :])
        for i in range(nx):
            if not np.isclose(np.nansum(RR[i, :]), 1):
                RR[i, :] = np.nan
            else:
                bad = np.where(~np.isfinite(RR[i, :]))[0]
                RR[i, bad] = 0

        #breakpoint()
        goody, goodx = np.where(np.isfinite(RR))
        xi, xf = goodx.min(), goodx.max()
        for x in range(nx):
            try:
                RR[x, 0:xi+20] = 0
                RR[x, xf - 20:] = 0
            except:
                pass

        breakpoint()
        bad = np.where(RR < 0)
        RR[bad] = 0
        for x in range(nx):
            RR[x, :] /= np.nansum(RR[x, :])

        RR[0:xi-20, :] = np.nan
        RR[xf - 20:, :] = np.nan
        #matplotlib.use("MacOSX"); plt.imshow(RR); plt.show()
        #breakpoint()

        # Return
        return spec1d, spec1d_unc, spec1dt, spec1dt_unc, RR

    @staticmethod
    def extract_SP2d(A, S, Ninv):
        """Written by Matthew A. Cornachione, tweaked here.

        Args:
            A (np.ndarray): The aperture tensor (weights).
            S (np.ndarray): The data vector.
            W (np.ndarray): The inverse of the pixel noise matrix. 

        Returns:
            np.ndarray: The raw deconvolved flux.
            np.ndarray: The reconvolved flux.
            np.ndarray: The reconvolution matrix used.
        """

        # Convert inputs
        A = np.matrix(A)
        S = np.reshape(np.matrix(S), (len(S), 1))
        Ninv = np.matrix(Ninv)
        
        # Convert to sparse
        A = sparse.csr_matrix(A)
        S = sparse.csr_matrix(S)
        Ninv = sparse.csr_matrix(Ninv)
        
        # Compute helpful vars
        Cinv = A.T * Ninv * A
        U, ss, Vt = np.linalg.svd(Cinv.todense())
        Cpsuedo = Vt.T * np.matrix(np.diag(1 / ss)) * U.T
        Cpsuedo = sparse.csr_matrix(Cpsuedo)

        # Initial flux
        flux = Cpsuedo * (A.T * Ninv * S)
        flux = flux.todense()
            
        # Compute reconvolution matrix
        Cinv = sparse.csc_matrix(Cinv)
        f, Wt = sparse.linalg.eigs(Cinv, k=int(np.min(Cinv.shape) - 2))
        F = np.matrix(np.diag(np.asarray(f)))
        F = np.abs(F)
        
        # Faster to do dense than sparse (at least in my one test session)
        Wt = np.real(Wt)
        WtDhW = Wt * np.sqrt(F) * Wt.T
        WtDhW = np.asarray(WtDhW)
        ss = np.sum(WtDhW, axis=1)
        Sinv = np.linalg.inv(np.diag(ss))
        WtDhW = np.matrix(WtDhW)
        R = Sinv * WtDhW
        
        # Reconvolve
        fluxtilde = R * flux

        # Convert to final arrays
        flux = np.asarray(flux).reshape((len(flux),))
        fluxtilde = np.asarray(fluxtilde).reshape((len(fluxtilde),))

        return flux, fluxtilde, R