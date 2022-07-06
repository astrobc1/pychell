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
#matplotlib.use('Agg')
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
    
    def __init__(self, trace_pos_poly_order=4, oversample=1,
                 badpix_threshold=5,
                 extract_orders=None,
                 n_iterations=20,
                 extract_aperture=None,
                 theta=None, q=None, sigma=None,
                 chunk_width=100):

        # Super init
        super().__init__(extract_orders=extract_orders)

        # Set params
        self.trace_pos_poly_order = trace_pos_poly_order
        self.oversample = oversample
        self.badpix_threshold = badpix_threshold
        self.extract_aperture = extract_aperture
        self.q = q
        self.sigma = sigma
        self.theta = theta
        self.chunk_width = chunk_width
        self.n_iterations = n_iterations
    

    def compute_model2d(self, data, trace_image, trace_mask, trace_dict, spec1d, trace_positions, extract_aperture):

        # Dims
        ny, nx = trace_image.shape

        # Number of chunks
        chunks = self.generate_chunks(trace_image, trace_mask)

        # Copy input spectrum and fix nans (inside bounds)
        f = np.copy(spec1d)
        xarr = np.arange(nx)
        good = np.where(np.isfinite(f))[0]
        bad = np.where(~np.isfinite(f))[0]
        f[bad] = pcmath.lin_interp(xarr[good], f[good], xarr[bad])

        bad = np.where(~np.isfinite(f))[0]
        f[bad] = 0

        # Stitch points
        model2d = np.full((ny, nx, len(chunks)), np.nan)

        # PSF params
        if self.sigma is None:
            sigma, q, theta = data.spec_module.get_psf_parameters(data, trace_dict["order"], trace_dict["fiber"])
        else:
            sigma, q, theta = self.sigma, self.q, self.theta

        # Loop over chunks
        for i in range(len(chunks)):
            xxi, xxf = chunks[i][0], chunks[i][1]
            nnnx = xxf - xxi + 1
            goodyy, _ = np.where(trace_mask[:, xxi:xxf+1])
            yyi, yyf = goodyy.min(), goodyy.max()
            nnny = yyf - yyi + 1
            S = trace_image[yyi:yyf+1, xxi:xxf+1]
            tp = trace_positions[xxi:xxf+1] - yyi
            xarr_aperture = np.arange(nnnx)
            A = self.gen_psf_matrix(S, xarr_aperture, tp, sigma[xxi:xxf+1], q[xxi:xxf+1], theta[xxi:xxf+1], extract_aperture)
            Aflat = A.reshape((nnny*nnnx, nnnx))
            model2d[yyi:yyf+1, xxi:xxf+1, i] = np.matmul(Aflat, f[xxi:xxf+1]).reshape((nnny, nnnx))
            model2d[:, xxi:xxi+20, i] = np.nan
            model2d[:, xxf-20:xxf+1, i] = np.nan

        model2d = np.nanmean(model2d, axis=2)

        return model2d

    @staticmethod
    @njit(nogil=True)
    def gen_psf_matrix(image, xarr_aperture, trace_positions, sigma, q, theta, extract_aperture):

        # Image dims
        ny, nx = image.shape

        # Aperture size (half)
        aperture_size_half = int(np.ceil((extract_aperture[1] - extract_aperture[0]) / 2))

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
                    if np.abs(dx) > aperture_size_half or np.abs(dy) > aperture_size_half:
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


    def generate_chunks(self, trace_image, trace_mask):
        goody, goodx = np.where(trace_mask)
        xi, xf = goodx.min(), goodx.max()
        nnx = xf - xi + 1
        yi, yf = goody.min(), goody.max()
        nny = yf - yi + 1
        chunk_width = np.min([self.chunk_width, 200])
        chunks = []
        chunks.append((xi, xi + self.chunk_width))
        for i in range(1, int(2 * np.ceil(nnx / self.chunk_width))):
            vi = chunks[i-1][1] - int(self.chunk_width / 2)
            vf = np.min([vi + self.chunk_width, xf])
            chunks.append((vi, vf))
            if vf == xf:
                break
        return chunks

    def bin_spec1d(self, spec1dhr):
        nx = int(len(spec1dhr) / self.oversample)
        return np.nansum(spec1dhr.reshape((nx, self.oversample)), axis=1)

        
class SPExtractor(GaussianDeconv2dExtractor):


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

        trace_positions = self.compute_trace_positions_centroids(image, badpix_mask, sregion, trace_dict, self.trace_pos_poly_order, n_iterations=10)

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
        trace_positions -= yi

        # Flag obvious bad pixels again
        trace_image_smooth = pcmath.median_filter2d(trace_image, width=5)
        peak = pcmath.weighted_median(trace_image_smooth, percentile=0.99)
        bad = np.where((trace_image < 0) | (trace_image > 50 * peak))
        if bad[0].size > 0:
            trace_image[bad] = np.nan
            trace_mask[bad] = 0

        # Extract Aperture
        if self.extract_aperture is None:
            _extract_aperture = [-int(np.ceil(trace_dict['height'] / 2)), int(np.ceil(trace_dict['height'] / 2))]
        else:
            _extract_aperture = self.extract_aperture

        # Main loop
        for i in range(self.n_iterations):

            print(f" [{data}] Extracting Trace {trace_dict['label']}, Iter [{i + 1}/{self.n_iterations}] ...", flush=True)

            # extraction
            spec1d, spec1d_unc, spec1dt, spec1dt_unc, RR = self.deconv2dextraction(data, trace_dict, trace_image, trace_mask, trace_positions, _extract_aperture, read_noise=read_noise)

            # Re-map pixels and flag in the 2d image.
            if i < self.n_iterations - 1:

                # 2d model
                model2d = self.compute_model2d(data, trace_image, trace_mask, trace_dict, spec1dt, trace_positions, _extract_aperture)
                #breakpoint() # matplotlib.use("MacOSX")

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
        spec1dt_smooth = pcmath.median_filter1d(spec1dt, width=3)
        med_val = pcmath.weighted_median(spec1dt_smooth, percentile=0.98)
        bad = np.where(np.abs(spec1dt - spec1dt_smooth) / med_val > 0.5)[0]
        if bad.size > 0:
            spec1dt[bad] = np.nan
            spec1dt_unc[bad] = np.nan
            badpix1d[bad] = 0
        
        return spec1dt, spec1dt_unc, badpix1d

    ##############################
    #### 2d DECONV EXTRACTION ####
    ##############################

    def deconv2dextraction(self, data, trace_dict, trace_image, trace_mask, trace_positions, extract_aperture, read_noise=0):
    
        # Copy input
        trace_image_cp = np.copy(trace_image)
        trace_mask_cp = np.copy(trace_mask)

        trace_image_cp = self.fix_nans_2d(trace_image_cp)

        # Dims
        ny, nx = trace_image_cp.shape

        # Flag negative pixels
        bad = np.where(trace_image_cp < 0)
        if bad[0].size > 0:
            trace_image_cp[bad] = np.nan
            trace_mask_cp[bad] = 0

        # Now set all nans to zero
        bad = np.where(~np.isfinite(trace_image_cp) | (trace_mask_cp == 0))
        if bad[0].size > 0:
            trace_image_cp[bad] = 0

        # Chunks
        chunks = self.generate_chunks(trace_image, trace_mask_cp)
        n_chunks = len(chunks)

        # Outputs (averaged over chunks before returning)
        spec1d = np.full((nx, n_chunks), np.nan)
        spec1dt = np.full((nx, n_chunks), np.nan)
        R = np.full((nx, nx, n_chunks), np.nan)

        # PSF params
        if self.sigma is None:
            sigma, q, theta = data.spec_module.get_psf_parameters(data, trace_dict["order"], trace_dict["fiber"])
        else:
            sigma, q, theta = self.sigma, self.q, self.theta

        # Loop over and extract chunks
        for i in range(n_chunks):

            # X pixel bounds for this chunk
            xi, xf = chunks[i][0], chunks[i][1]
            nnx = xf - xi + 1

            # Y pixel bounds  for this chunk
            goody, _ = np.where(trace_mask_cp[:, xi:xf+1])
            yi, yf = goody.min(), goody.max()
            nny = yf - yi + 1

            # Crop image and mask to this chunk
            S = trace_image_cp[yi:yf+1, xi:xf+1]
            M = trace_mask_cp[yi:yf+1, xi:xf+1]

            # Crop aperture params to this chunk and upsample
            tp = trace_positions[xi:xf+1] - yi
            _sigma = sigma[xi:xf+1]
            _q = q[xi:xf+1]
            _theta = theta[xi:xf+1]
            xarr_aperture = np.arange(-0.5 + 0.5 / self.oversample, nnx - 1 + 0.5 - 0.5 / self.oversample + 1 / self.oversample, 1 / self.oversample)
            xarr = np.arange(nnx)
            tp_hr = scipy.interpolate.interp1d(xarr, tp, fill_value="extrapolate")(xarr_aperture)
            sigma_hr = scipy.interpolate.interp1d(xarr, _sigma, fill_value="extrapolate")(xarr_aperture)
            q_hr = scipy.interpolate.interp1d(xarr, _q, fill_value="extrapolate")(xarr_aperture)
            theta_hr = scipy.interpolate.interp1d(xarr, _theta, fill_value="extrapolate")(xarr_aperture)

            # Generate Aperture tensor for this chunk
            A = self.gen_psf_matrix(S, xarr_aperture, tp_hr, sigma_hr, q_hr, theta_hr, extract_aperture)

            # Prep inputs for sparse extraction
            Aflat = A.reshape((nny*nnx, len(xarr_aperture)))
            Sflat = S.flatten()
            #W = np.zeros((nny, nnx))
            #for k in range(nny):
            #    for j in range(nnx):
            #        W[k, j] = A[k, j, :].sum()**2 / (S[k, j] + read_noise**2)
            W = 1 / (S + read_noise**2)
            bad = np.where(~np.isfinite(W))
            W[bad] = 0
            Wbig = np.diag(W.flatten())

            # Call sparse extraction
            syhr, sythr, _Rhr = self.extract_SP2d(Aflat, Sflat, Wbig)

            # Bin back to detector grid
            sy = self.bin_spec1d(syhr)
            syt = self.bin_spec1d(sythr)

            # Mask edge errors
            distrust_width = int(np.nanmax(np.abs(self.extract_aperture)) * 2)
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

        # Correct negatives and zeros in reconvolved spectrum
        bad = np.where(spec1dt <= 0)
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
        goody, goodx = np.where(np.isfinite(RR))
        xi, xf = goodx.min(), goodx.max()
        for x in range(nx):
            try:
                RR[x, 0:xi+20] = 0
                RR[x, xf - 20:] = 0
            except:
                pass

        bad = np.where(RR < 0)
        RR[bad] = 0
        for x in range(nx):
            RR[x, :] /= np.nansum(RR[x, :])

        RR[0:xi-20, :] = 0
        RR[xf - 20:, :] = 0

        bad = np.where((RR <= 0) | ~np.isfinite(RR))
        RR[bad] = 0

        bad = np.where(~np.isfinite(spec1dt) | (spec1dt <= 0))[0]
        spec1dt[bad] = np.nan

        # Take average of each
        #lsf = np.full(nx, np.nan)
        #for i in range(nx):
            #lsf[i] = np.nanmean()
         #matplotlib.use("MacOSX"); plt.imshow(RR); plt.show()

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