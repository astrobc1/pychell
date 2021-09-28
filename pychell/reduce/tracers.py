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
import pychell.data as pcdata


class OrderTracer:
    pass
    
#class PredeterminedTrace(OrderTracer):
    
    #def __init__(self, filename):
        #self.data = pcdata.ImageMap()

class DensityClusterTracer(OrderTracer):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, n_orders=None, trace_pos_poly_order=2, refine_position=False, order_spacing=10, mask_left=200, mask_right=200, mask_top=20, mask_bottom=20, downsample=4):
        self.trace_pos_poly_order = trace_pos_poly_order
        self.refine_position = refine_position
        self.order_spacing = order_spacing
        self.mask_left = mask_left
        self.mask_right = mask_right
        self.n_orders = n_orders
        self.mask_top = mask_top
        self.mask_bottom = mask_bottom
        self.downsample = downsample

    ######################
    #### TRACE ORDERS ####
    ######################
        
    def trace(self, order_map, n_cores):
        
        print(f"Tracing orders for {order_map} ...", flush=True)
        
        # Load flat field image
        source_image = order_map.source.parse_image()

        # Feeder
        feeder = order_map.parser.spec_module.feeder

        # Fiber
        fiber = order_map.parser.parse_fiber_num(order_map.source) if feeder.lower() == "fiber" else None
    
        # Image dimensions
        ny, nx = source_image.shape
    
        # Mask
        source_image[0:self.mask_bottom, :] = np.nan
        source_image[ny-self.mask_top:, :] = np.nan
        source_image[:, 0:self.mask_left] = np.nan
        source_image[:, nx-self.mask_right:] = np.nan

        # Smooth the flat.
        source_image_smooth = pcmath.median_filter2d(source_image, width=5, preserve_nans=False)
        
        # Normalize the flat.
        source_image_smooth_norm = pcmath.normalize_image(source_image_smooth, window=self.order_spacing, n_knots=self.n_orders, percentile=0.99, downsample=8)
        
        # Downsample in the horizontal direction to save on memory
        nx_lr = int(nx / self.downsample)

        # Only consider regions where the flux is greater than 50%
        order_locations_lr = np.full((ny, nx_lr), np.nan)
        source_image_smooth_norm_lr = source_image_smooth_norm[:, ::self.downsample]
        good_lr = np.where(source_image_smooth_norm_lr > 0.5)
        order_locations_lr[good_lr] = 1
        
        # Store points for DBSCAN
        good_points_lr = np.empty(shape=(good_lr[0].size, 2), dtype=float)
        good_points_lr[:, 0], good_points_lr[:, 1] = good_lr[1], good_lr[0] # NOTE: good_lr is (y,x), good_points_lr is (x,y)
        
        # Create the Density clustering object and run
        density_cluster = sklearn.cluster.DBSCAN(eps=0.7*self.order_spacing, min_samples=5, metric='euclidean', algorithm='auto', p=None, n_jobs=n_cores)
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
        thresh = 0.8 * pcmath.weighted_median(n_points_per_label, percentile=0.9)
        good = np.where(n_points_per_label > thresh)[0]
        bad = np.where(n_points_per_label <= thresh)[0]
        good_labels = labels_unique_init[good]
        bad_labels = labels_unique_init[bad]
        order_locs_lr_labeled_final = np.full_like(order_locs_lr_labeled, fill_value=np.nan)
        y_med = np.zeros_like(good_labels)
        for l in range(len(good_labels)):
            inds = np.where(order_locs_lr_labeled == good_labels[l])
            order_locs_lr_labeled_final[inds] = l + 1
            y_med[l] = np.nanmedian(inds[0])

        # Ensure the labels are properly sorted
        ss = np.argsort(y_med)
        for l in range(len(good_labels)):
            inds = np.where(order_locs_lr_labeled == good_labels[l])
            order_locs_lr_labeled_final[inds] = ss[l] + 1
            

        # Overwrite good labels
        good_labels = np.unique(order_locs_lr_labeled_final[np.where(np.isfinite(order_locs_lr_labeled_final))])

        # Initiate the order image and orders list
        order_image = np.full_like(source_image, np.nan)
        orders_list = []

        # Loop over good labels and fill image
        xarr = np.arange(nx)
        for l in range(len(good_labels)):
            inds = np.where(order_locs_lr_labeled_final == good_labels[l])
            pfit = np.polyfit(self.downsample * inds[1], inds[0], 2)
            dys = np.full(nx_lr, np.nan)
            for i in range(len(inds[1])):
                _inds = np.where(inds[1] == i)[0]
                if len(_inds) > 3:
                    dys[i] = np.max(inds[0][_inds]) - np.min(inds[0][_inds])
            height = int(np.nanmedian(dys)) - 4
            orders_list.append([{'label': good_labels[l], 'height': height, 'pcoeffs': pfit, 'feeder': feeder, 'fiber': fiber}])
            order_center = np.polyval(pfit, xarr)
            for x in range(nx):
                ymid = order_center[x]
                y_low = int(np.floor(ymid - height / 2))
                y_high = int(np.ceil(ymid + height / 2))
                if y_low < self.mask_bottom or y_high > ny - self.mask_top - 1:
                    continue
                order_image[y_low:y_high + 1, x] = good_labels[l]

        order_map.orders_list = orders_list
        order_map.save(order_image)

