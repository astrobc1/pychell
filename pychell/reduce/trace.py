# Built in python modules
from functools import reduce
import glob
import os
import sys
import operator
import json
import copy
import warnings

# LLVM
from numba import jit, njit

# Maths
import numpy as np
import scipy.interpolate
import scipy.signal
import astropy.stats as stats
import sklearn.cluster

# Graphics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# Astropy
from astropy.io import fits

# Pychell modules
import pychell.maths as pcmath
import pychell.data.spectraldata as pcspecdata

class OrderTracer:
    """Base class for Order tracing algorithms.
    """
    
    @staticmethod
    def gen_image(orders_list, ny, nx, xrange=None, poly_mask_top=None, poly_mask_bottom=None):
        """Generates a synthetic echellogram where the value of each pixel is the label of that trace.

        Args:
            orders_list (list): The list of dictionaries for each trace.
            shape (tuple): The dimensions of the image (ny, nx).
            xrange (list, optional): The bounding left and right pixels. Defaults to [0, nx-1].
            poly_mask_top (list, optional): A list of points (each point a list of length 2 containing an (x, y) pair) that define the bounding polynomial at the top of the image. Defaults to None.
            poly_mask_bottom (list, optional): Same but for the bottom bounding polynomial. Defaults to None.

        Returns:
            np.ndarray: The image.
        """
        
        # Initiate order image
        order_image = np.full((ny, nx), np.nan)

        # Helpful arr
        xarr = np.arange(nx)

        # Xrange
        if xrange is None:
            xrange = [0, nx - 1]
        
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
        
        for o in range(len(orders_list)):
            order_center = np.polyval(orders_list[o]['pcoeffs'], xarr)
            for x in range(xrange[0], xrange[1] + 1):
                ymid = order_center[x]
                y_low = int(np.floor(ymid - orders_list[o]['height'] / 2))
                y_high = int(np.ceil(ymid + orders_list[o]['height'] / 2))
                if y_low < poly_bottom[x] or y_high > poly_top[x]:
                    continue
                order_image[y_low:y_high + 1, x] = orders_list[o]['label']

        return order_image

# class DensityClusterTracer(OrderTracer):
    
#     ###############################
#     #### CONSTRUCTOR + HELPERS ####
#     ###############################

#     def __init__(self, n_orders, poly_order=2, poly_mask_bottom=None, poly_mask_top=None, order_spacing=10, order_heights=10, downsample=4):
#         """Construct a Density cluster tracer

#         Args:
#             n_orders ([type]): [description]
#             poly_order (int, optional): [description]. Defaults to 2.
#             poly_mask_bottom ([type], optional): [description]. Defaults to None.
#             poly_mask_top ([type], optional): [description]. Defaults to None.
#             order_spacing (int, optional): [description]. Defaults to 10.
#             order_heights (int, optional): [description]. Defaults to 10.
#             downsample (int, optional): [description]. Defaults to 4.
#         """
#         self.n_orders = n_orders
#         self.poly_order = poly_order
#         self.order_spacing = order_spacing
#         self.poly_mask_bottom = poly_mask_bottom
#         self.poly_mask_top = poly_mask_top
#         try:
#             iter(order_heights)
#             self.order_heights = order_heights
#         except:
#             self.order_heights = np.full(self.n_orders, order_heights)
#         self.downsample = downsample


#     ######################
#     #### TRACE ORDERS ####
#     ######################
        
#     def trace(self, recipe, data):

#         # Image
#         image = data.parse_image()
        
#         # Fiber number
#         try:
#             fiber = int(data.spec_module.parse_fiber_nums(data))
#         except:
#             fiber = None

#         # Call function
#         orders_list = self._trace(image, self.n_orders, self.poly_order, self.order_spacing, self.order_heights, recipe.xrange, self.poly_mask_bottom, self.poly_mask_top, self.downsample, fiber, recipe.n_cores)

#         # Store result
#         data.orders_list = orders_list


#     @staticmethod
#     def _trace(image, n_orders, poly_order=2, order_spacing=10, order_heights=10, xrange=None, poly_mask_bottom=None, poly_mask_top=None, downsample=4, fiber=None, n_cores=1):

#         try:
#             iter(order_heights)
#         except:
#             order_heights = np.full(n_orders, order_heights)
    
#         # Image dimensions
#         ny, nx = image.shape

#         # xrange
#         if xrange is None:
#             xrange = [int(0.1 * nx), int(0.9 * nx)]
    
#         # Mask
#         image = np.copy(image)
#         image[:, 0:xrange[0]] = np.nan
#         image[:, xrange[1] + 1:] = np.nan

#         # Helpful arrs
#         xarr = np.arange(nx)
#         yarr = np.arange(ny)

#         # Top and bottom bounding polynomials
#         poly_top = np.polyval(poly_mask_top, xarr)
#         poly_bottom = np.polyval(poly_mask_bottom, xarr)

#         for x in range(nx):
#             bad = np.where((yarr < poly_bottom[x]) | (yarr > poly_top[x]))[0]
#             image[bad, x] = np.nan

#         # Smooth the flat.
#         image_smooth = pcmath.median_filter2d(image, width=5, preserve_nans=False)
        
#         # Normalize the flat.
#         image_smooth_norm = pcmath.normalize_image(image_smooth, np.nanmean(order_heights), order_spacing, percentile=0.99, downsample=4)
        
#         # Downsample in the horizontal direction for performance
#         nx_lr = int(nx / downsample)

#         # Only consider regions where the flux is greater than 50%
#         order_locations_lr = np.full((ny, nx_lr), np.nan)
#         image_smooth_norm_lr = image_smooth_norm[:, ::downsample]
#         good_lr = np.where(image_smooth_norm_lr > 0.5)
#         order_locations_lr[good_lr] = 1
        
#         # Store points for DBSCAN
#         good_points_lr = np.empty(shape=(good_lr[0].size, 2), dtype=float)
#         good_points_lr[:, 0], good_points_lr[:, 1] = good_lr[1], good_lr[0] # NOTE: good_lr is (y,x), good_points_lr is (x,y)
        
#         # Create the Density clustering object and run
#         density_cluster = sklearn.cluster.DBSCAN(eps=0.7*order_spacing, min_samples=5, metric='euclidean', algorithm='auto', p=None, n_jobs=n_cores)
#         db = density_cluster.fit(good_points_lr)
        
#         # Extract the labels
#         labels = db.labels_
#         good_labels = np.where(labels >= 0)[0]
#         if good_labels.size == 0:
#             raise ValueError(f"Error! The order mapping algorithm failed. No usable labels found for {order_map}.")
#         good_labels_init = labels[good_labels]
#         labels_unique_init = np.unique(good_labels_init)

#         # Create an initial order map image
#         order_locs_lr_labeled = np.full(shape=(ny, nx_lr), fill_value=np.nan)
#         for l in range(good_points_lr[:, 0].size):
#             order_locs_lr_labeled[good_lr[0][l], good_lr[1][l]] = labels[l]
        
#         # Compute the number of points per label
#         n_points_per_label = np.zeros(labels_unique_init.size, dtype=int)
#         for l in range(labels_unique_init.size):
#             inds = np.where(labels == labels_unique_init[l])
#             n_points_per_label[l] = inds[0].size

#         # Now flag the bad labels
#         ss = np.argsort(n_points_per_label)
#         good_labels = [labels_unique_init[ss[len(ss) - k - 1]] for k in range(n_orders)]
#         order_locs_lr_labeled_final = np.full_like(order_locs_lr_labeled, fill_value=np.nan)
#         y_med = np.zeros_like(good_labels)
#         for l in range(len(good_labels)):
#             inds = np.where(order_locs_lr_labeled == good_labels[l])
#             order_locs_lr_labeled_final[inds] = l + 1
#             y_med[l] = np.nanmedian(inds[0])

#         # Ensure the labels are properly sorted
#         ss = np.argsort(y_med)
#         for l in range(len(good_labels)):
#             inds = np.where(order_locs_lr_labeled == good_labels[ss[l]])
#             order_locs_lr_labeled_final[inds] = l + 1
            
#         # Overwrite good labels
#         good_labels = np.unique(order_locs_lr_labeled_final[np.where(np.isfinite(order_locs_lr_labeled_final))])

#         # Initiate the order image and orders list
#         order_image = np.full_like(image, np.nan)
#         orders_list = []
#         # Loop over good labels and fill image
#         xarr = np.arange(nx)
#         for l in range(len(good_labels)):
#             inds = np.where(order_locs_lr_labeled_final == good_labels[l])
#             if fiber is not None:
#                 label = float(str(int(good_labels[l])) + "." + str(int(fiber)))
#             else:
#                 label = int(good_labels[l])
#             pfit = np.polyfit(downsample * inds[1], inds[0], 2)
#             orders_list.append({'label': label, 'height': order_heights[l], 'pcoeffs': pfit})
            
#         return orders_list


class PeakTracer(OrderTracer):
    """Traces orders by looking at vertical cross sections in a smoothed image.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, n_orders, poly_order=2, order_spacing=10, order_heights=10, xleft=None, xright=None, n_slices=10):
        """Construct a PeakTracer object.

        Args:
            n_orders (int): The number of orders.
            poly_order (int, optional): The polynomial order to fit the positions with. A higher order polynomial can be used in extraction to further refine these positions if desired. Defaults to 2.
            order_spacing (int, optional): The minimum spacing between the edges of orders. Defaults to 10.
            order_heights (int, optional): The minimum height (in spatial direction) of the orders. Defaults to 10.
            xleft (int, optional): The left most slice. Defaults to None.
            xright (int, optional): The right most slice. Defaults to None.
            n_slices (int, optional): How many slices to use. Defaults to 10.
        """
        self.n_orders = n_orders
        self.poly_order = poly_order
        self.order_spacing = order_spacing
        self.xleft = xleft
        self.xright = xright
        self.n_slices = n_slices
        try:
            iter(order_heights)
            self.order_heights = order_heights
        except:
            self.order_heights = np.full(self.n_orders, order_heights)


    ######################
    #### TRACE ORDERS ####
    ######################
        
    def trace(self, recipe, data):
        """Trace the orders.

        Args:
            recipe (ReduceRecipe): The reduce recipe.
            data (Echellogram): The data object to trace orders from.
        """

        # Image
        image = data.parse_image()

        # dims
        ny, nx = image.shape

        # xleft and xright
        if self.xleft is None:
            xleft = int(nx / 4)
        else:
            xleft = self.xleft
        if self.xright is None:
            xright = int(3 * nx / 4)
        else:
            xright = self.xright
        
        # Fiber number
        try:
            fiber = int(data.spec_module.parse_fiber_nums(data))
        except:
            fiber = None

        # Call function
        orders_list = self._trace(image, self.n_orders, self.poly_order, self.order_spacing, self.order_heights, recipe.xrange, recipe.poly_mask_bottom, recipe.poly_mask_top, fiber, xleft, xright, self.n_slices)

        # Store result
        data.orders_list = orders_list


    @staticmethod
    def _trace(image, n_orders, poly_order=2, order_spacing=None, order_heights=None, xrange=None, poly_mask_bottom=None, poly_mask_top=None, fiber=None, xleft=None, xright=None, n_slices=10):

        try:
            iter(order_heights)
        except:
            order_heights = np.full(n_orders, order_heights)

        # Image dimensions
        ny, nx = image.shape

        # Helpful arrs
        xarr = np.arange(nx)
        yarr = np.arange(ny)

        # xrange
        if xrange is None:
            xrange = [int(0.1 * nx), int(0.9 * nx)]
    
        # Mask
        image = np.copy(image)
        image[:, 0:xrange[0]] = np.nan
        image[:, xrange[1] + 1:] = np.nan

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

        # Smooth the flat
        image_smooth = pcmath.median_filter2d(image, width=3, preserve_nans=False)

        # Slices
        xslices = np.linspace(xleft + 20, xright - 20, num=n_slices).astype(int)
        peaks = []
        for i in range(n_slices):
            x = xslices[i]
            s = np.nanmedian(image_smooth[:, x-5:x+5], axis=1)
            goody, _ = np.where(np.isfinite(image_smooth[:, x-5:x+5]))
            yi, yf = goody.min(), goody.max()
            continuum = pcmath.generalized_median_filter1d(s, width=3 * int(np.nanmean(order_spacing)), percentile=0.99)
            continuum[yi:yi+20] = np.nanmedian(continuum[yi+20:yi+40])
            continuum[yf-20:] = np.nanmedian(continuum[yf-40:])
            s /= continuum
            good = np.where(s > 0.7)
            s[good] = 1.0

            # Peak finding
            # Estimate peaks in pixel space (just indices)
            _peaks = scipy.signal.find_peaks(s, height=np.full(ny, 0.75), distance=order_spacing)[0]
            _peaks = np.sort(_peaks)
            if len(_peaks) == n_orders:
                peaks.append(_peaks)
            else:
                peaks.append(None)

        # Fit
        poly_coeffs = []
        for i in range(n_orders):
            xx = [xslices[i] for i in range(len(xslices)) if peaks[i] is not None]
            yy = [_peaks[i] for _peaks in peaks if _peaks is not None]
            pfit = np.polyfit(xx, yy, poly_order)
            poly_coeffs.append(pfit)

        # Ensure the labels are properly sorted
        y_mean = np.array([np.nanmean(np.polyval(poly_coeffs[i], xarr)) for i in range(len(poly_coeffs))], dtype=float)
        ss = np.argsort(y_mean)
        poly_coeffs = [poly_coeffs[ss[i]] for i in range(len(ss))]

        # Now build the orders list
        orders_list = []
        for i in range(len(poly_coeffs)):
            if fiber is not None:
                label = float(str(int(i + 1)) + "." + str(int(fiber)))
            else:
                label = i + 1
            orders_list.append({'label': label, 'height': order_heights[i], 'pcoeffs': poly_coeffs[i]})
        
        return orders_list