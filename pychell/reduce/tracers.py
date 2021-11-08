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
import astropy.stats as stats
import sklearn.cluster

# Plotting
import matplotlib
import matplotlib.pyplot as plt

# Astropy
from astropy.io import fits

# Pychell modules
import pychell.maths as pcmath
import pychell.data.spectraldata as pcspecdata

class OrderTracer:
    
    @staticmethod
    def gen_image(orders_list, ny, nx, mask_left=200, mask_right=200, mask_top=20, mask_bottom=20):
        order_image = np.full((ny, nx), np.nan)
        xarr = np.arange(nx)
        for o in range(len(orders_list)):
            order_center = np.polyval(orders_list[o]['pcoeffs'], xarr)
            for x in range(nx):
                ymid = order_center[x]
                y_low = int(np.floor(ymid - orders_list[o]['height'] / 2))
                y_high = int(np.ceil(ymid + orders_list[o]['height'] / 2))
                if y_low < mask_bottom or y_high > ny - mask_top - 1:
                    continue
                order_image[y_low:y_high + 1, x] = orders_list[o]['label']

        return order_image

class DensityClusterTracer(OrderTracer):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, n_orders, poly_order=2, order_spacing=10, heights=10, downsample=4, n_cores=1):
        self.n_orders = n_orders
        self.poly_order = poly_order
        self.order_spacing = order_spacing
        try:
            iter(heights)
            self.heights = heights
        except:
            self.heights = np.full(self.n_orders, heights)
        self.n_cores = n_cores
        self.downsample = downsample


    ######################
    #### TRACE ORDERS ####
    ######################
        
    def trace(self, recipe, data):

        # Image
        image = data.parse_image()
        
        # Fiber number
        try:
            fiber = int(data.spec_module.parse_fiber_nums(data))
        except:
            fiber = None

        # Call function
        orders_list = self._trace(image, self.n_orders, self.poly_order, self.order_spacing, self.heights, recipe.mask_left, recipe.mask_right, recipe.mask_top, recipe.mask_bottom, self.downsample, fiber, self.n_cores)

        # Store result
        data.orders_list = orders_list


    @staticmethod
    def _trace(image, n_orders, poly_order=2, order_spacing=10, heights=10, mask_left=200, mask_right=200, mask_top=20, mask_bottom=20, downsample=4, fiber=None, n_cores=1):

        try:
            iter(heights)
        except:
            heights = np.full(n_orders, heights)
    
        # Image dimensions
        ny, nx = image.shape
    
        # Mask
        image[0:mask_bottom, :] = np.nan
        image[ny-mask_top:, :] = np.nan
        image[:, 0:mask_left] = np.nan
        image[:, nx-mask_right:] = np.nan

        # Smooth the flat.
        image_smooth = pcmath.median_filter2d(image, width=5, preserve_nans=False)
        
        # Normalize the flat.
        image_smooth_norm = pcmath.normalize_image(image_smooth, np.nanmean(heights), order_spacing, percentile=0.99, downsample=4)
        
        # Downsample in the horizontal direction for performance
        nx_lr = int(nx / downsample)

        # Only consider regions where the flux is greater than 50%
        order_locations_lr = np.full((ny, nx_lr), np.nan)
        image_smooth_norm_lr = image_smooth_norm[:, ::downsample]
        good_lr = np.where(image_smooth_norm_lr > 0.5)
        order_locations_lr[good_lr] = 1
        
        # Store points for DBSCAN
        good_points_lr = np.empty(shape=(good_lr[0].size, 2), dtype=float)
        good_points_lr[:, 0], good_points_lr[:, 1] = good_lr[1], good_lr[0] # NOTE: good_lr is (y,x), good_points_lr is (x,y)
        
        # Create the Density clustering object and run
        density_cluster = sklearn.cluster.DBSCAN(eps=0.7*order_spacing, min_samples=5, metric='euclidean', algorithm='auto', p=None, n_jobs=n_cores)
        db = density_cluster.fit(good_points_lr)
        
        # Extract the labels
        labels = db.labels_
        good_labels = np.where(labels >= 0)[0]
        if good_labels.size == 0:
            raise ValueError(f"Error! The order mapping algorithm failed. No usable labels found for {order_map}.")
        good_labels_init = labels[good_labels]
        labels_unique_init = np.unique(good_labels_init)

        # Create an initial order map image
        order_locs_lr_labeled = np.full(shape=(ny, nx_lr), fill_value=np.nan)
        for l in range(good_points_lr[:, 0].size):
            order_locs_lr_labeled[good_lr[0][l], good_lr[1][l]] = labels[l]
        
        # Compute the number of points per label
        n_points_per_label = np.zeros(labels_unique_init.size, dtype=int)
        for l in range(labels_unique_init.size):
            inds = np.where(labels == labels_unique_init[l])
            n_points_per_label[l] = inds[0].size

        # Now flag the bad labels
        ss = np.argsort(n_points_per_label)
        good_labels = [labels_unique_init[ss[len(ss) - k - 1]] for k in range(n_orders)]
        order_locs_lr_labeled_final = np.full_like(order_locs_lr_labeled, fill_value=np.nan)
        y_med = np.zeros_like(good_labels)
        for l in range(len(good_labels)):
            inds = np.where(order_locs_lr_labeled == good_labels[l])
            order_locs_lr_labeled_final[inds] = l + 1
            y_med[l] = np.nanmedian(inds[0])

        # Ensure the labels are properly sorted
        ss = np.argsort(y_med)
        for l in range(len(good_labels)):
            inds = np.where(order_locs_lr_labeled == good_labels[ss[l]])
            order_locs_lr_labeled_final[inds] = l + 1
            
        # Overwrite good labels
        good_labels = np.unique(order_locs_lr_labeled_final[np.where(np.isfinite(order_locs_lr_labeled_final))])

        # Initiate the order image and orders list
        order_image = np.full_like(image, np.nan)
        orders_list = []
        # Loop over good labels and fill image
        xarr = np.arange(nx)
        for l in range(len(good_labels)):
            inds = np.where(order_locs_lr_labeled_final == good_labels[l])
            if fiber is not None:
                label = float(str(int(good_labels[l])) + "." + str(int(fiber)))
            else:
                label = int(good_labels[l])
            pfit = np.polyfit(downsample * inds[1], inds[0], 2)
            orders_list.append({'label': label, 'height': heights[l], 'pcoeffs': pfit})
            
        return orders_list
