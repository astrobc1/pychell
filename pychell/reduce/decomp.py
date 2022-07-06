# Python default modules
import os

# Science / Math
import numpy as np
import scipy.interpolate

# Pyreduce
import pyreduce.extract

# Graphics
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Pychell modules
import pychell.maths as pcmath
from pychell.reduce.extract import SpectralExtractor

# From optimal
class DecompExtractor(SpectralExtractor):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, n_iterations=20,
                 trace_pos_poly_order=4, oversample=4,
                 badpix_threshold=5,
                 extract_orders=None,
                 chunk_width=600,
                 extract_aperture=None, lambda_sf=0.5, lambda_sp=0.0, tilt=None, shear=None):

        # Super init
        super().__init__(extract_orders=extract_orders)
        self.n_iterations = n_iterations
        self.trace_pos_poly_order = trace_pos_poly_order
        self.oversample = oversample
        self.chunk_width = chunk_width
        self.badpix_threshold = badpix_threshold
        self.extract_aperture = extract_aperture
        self.lambda_sf = lambda_sf
        self.lambda_sp = lambda_sp
        self.tilt = tilt
        self.shear = shear
        
        
    #######################################################################
    #### PRIMARY METHOD TO EXTRACT SINGLE TRACE FOR ENTIRE ORDER WIDTH ####
    #######################################################################

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

            # Decomp extraction
            spec1d, spec1d_unc, _ = self.decomp_extraction(trace_image, trace_mask, trace_positions, _extract_aperture, read_noise=read_noise)

            # Re-map pixels and flag in the 2d image.
            if i < self.n_iterations - 1:

                # 2d model
                model2d = self.compute_model2d(data, trace_image, trace_mask, trace_dict, spec1d, trace_positions, _extract_aperture)

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


    ###########################
    #### DECOMP EXTRACTION ####
    ###########################

    def decomp_extraction(self, trace_image, trace_mask, trace_positions, extract_aperture, read_noise=0):

        # Copy input
        trace_image_cp = np.copy(trace_image)
        trace_positions_cp = np.copy(trace_positions)
        trace_mask_cp = np.copy(trace_mask)

        # Dims
        ny, nx = trace_image.shape

        # Aperture
        yrange = [int(np.ceil(np.abs(extract_aperture[0]))), int(np.ceil(extract_aperture[1]))]

        # Now change all bad pixels to zeros
        bad = np.where(trace_mask == 0)
        if bad[0].size > 0:
            trace_image_cp[bad] = 0
        
        # Fix bad columns
        trace_mask_cp = self.fix_nans_2d(trace_image_cp, trace_positions, extract_aperture)

        # Only pass good data
        goody, goodx = np.where(trace_mask_cp)
        xxi, xxf = goodx.min(), goodx.max()
        yyi, yyf = goody.min(), goody.max()
        nnx = xxf - xxi + 1
        xrange = [0, nnx-1]
        S = trace_image_cp[yyi:yyf+1, xxi:xxf+1]
        M = np.logical_not(trace_mask_cp[yyi:yyf+1, xxi:xxf+1])
        S = np.ma.masked_array(S, mask=M)
        ycen = trace_positions_cp[xxi:xxf+1] - yyi
        tilt = np.copy(self.tilt[xxi:xxf+1])
        shear = np.copy(self.shear[xxi:xxf+1])

        # Outputs
        spec1d = np.full(nx, np.nan)
        spec1d_unc = np.full(nx, np.nan)
        
        # Call
        result = pyreduce.extract.extract_spectrum(S, ycen, yrange=yrange, xrange=np.copy(xrange), lambda_sf=self.lambda_sf, lambda_sp=self.lambda_sp, osample=self.oversample, readnoise=read_noise, tilt=tilt, shear=shear)

        # Store outputs
        #breakpoint()

        #matplotlib.use("MacOSX"); plt.plot(result[0]); plt.show()
        spec1d[xxi:xxf+1] = result[0]
        spec1d_unc[xxi:xxf+1] = result[3]
        trace_profile = result[1]

        # Flag zeros
        bad = np.where((spec1d <= 0) | (spec1d_unc <= 0))[0]
        if bad.size > 0:
            spec1d[bad] = np.nan
            spec1d_unc[bad] = np.nan

        # Return
        return spec1d, spec1d_unc, trace_profile

    #########################
    #### CREATE 2d MODEL ####
    #########################

    #def compute_model2d(self, data, trace_image, trace_mask, trace_dict, spec1d, trace_positions, _extract_aperture, trace_profile):
        #breakpoint() # import matplotlib; matplotlib.pyplot as plt; plt.imshow(trace_image); plt.show()
        #for x in range(nx):

        #return 

    @staticmethod
    def fix_nans_2d(trace_image, trace_positions, extract_aperture):
        ny, nx = trace_image.shape
        goody, goodx = np.where(np.isfinite(trace_image))
        xi, xf = goodx.min(), goodx.max()
        bady, badx = np.where(~np.isfinite(trace_image))
        trace_image_out = np.copy(trace_image)
        trace_image_out = np.ma.masked_invalid(trace_image_out)
        trace_image_out.fill_value = 0
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
        x1 = xx[~trace_image_out.mask]
        y1 = yy[~trace_image_out.mask]
        trace_image_out = trace_image[~trace_image_out.mask]
        trace_image_out = scipy.interpolate.griddata((x1, y1), trace_image_out.ravel(), (xx, yy), method='linear')
        return trace_image_out