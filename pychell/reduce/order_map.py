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

# Pychell modules
import pychell.maths as pcmath
import pychell.data as pcdata

class OrderTracer:
    pass
    
class PredeterminedTrace(OrderTracer):
    pass

class DensityClusterTracer(OrderTracer):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, trace_pos_poly_order=2, mask_left=200, mask_right=200, mask_top=20, mask_bottom=20):
        self.trace_pos_poly_order = trace_pos_poly_order
        self.mask_left = mask_left
        self.mask_right = mask_right
        self.mask_top = mask_top
        self.mask_bottom = mask_bottom

    ######################
    #### TRACE ORDERS ####
    ######################

    def trace(self, order_map, reducer):
        
        print(f"Tracing orders for {order_map} ...", flush=True)
        
        # Load flat field image
        source_image = order_map.source.parse_image()
    
        # Image dimensions
        ny, nx = source_image.shape
    
        # Mask
        source_image[0:self.mask_bottom, :] = np.nan
        source_image[ny-self.mask_top:, :] = np.nan
        source_image[:, 0:self.mask_left] = np.nan
        source_image[:, nx-self.mask_right:] = np.nan

        # Smooth the flat.
        source_image_smooth = pcmath.median_filter2d(source_image, width=5, preserve_nans=False)
    
        # Do a horizontal normalization of the smoothed flat image to kind of remove the continuum
        y_ranges = np.linspace(0, ny, num=10).astype(int)
        first_x = self.mask_left
        last_x = nx - self.mask_right
        for i in range(len(y_ranges)-1):
            y_low = y_ranges[i]
            y_top = y_ranges[i+1]
            for x in range(first_x, last_x):
                source_image_smooth[y_low:y_top, x] = source_image_smooth[y_low:y_top, x] / pcmath.weighted_median(source_image_smooth[y_low:y_top, x], percentile=0.99)

        # Only consider regions where the flux is greater than 50%
        order_locations_all = np.full_like(source_image_smooth, np.nan)
        good = np.where(source_image_smooth > 0.5)
        order_locations_all[good] = 1
        
        # Perform the density cluster algorithm on a lower res grid to save time
        nx_lr = np.min([512, nx])
        ny_lr = np.min([2048, ny])
        Mx = int(nx / nx_lr)
        My = int(ny / ny_lr)
        first_x_lr, last_x_lr = int(first_x / Mx), int(last_x / Mx)
        x = np.arange(ny).astype(int)
        y = np.arange(ny).astype(int)
        x_lr = np.arange(nx_lr).astype(int)
        y_lr = np.arange(ny_lr).astype(int)
        
        # Get the low res order locations and store them in a way to be read into DBSCAN
        order_locations_lr = order_locations_all[::My, ::Mx]
        good_lr = np.where(order_locations_lr == 1)
        good_points = np.empty(shape=(good_lr[0].size, 2), dtype=float)
        good_points[:, 0], good_points[:, 1] = good_lr[1], good_lr[0]
        
        # Create the Density clustering object and run
        density_cluster = sklearn.cluster.DBSCAN(eps=10, min_samples=5, metric='euclidean', algorithm='auto', p=None, n_jobs=1)
        db = density_cluster.fit(good_points)
        
        # Extract the labels
        labels = db.labels_
        good_labels = np.where(labels >= 0)[0]
        if good_labels.size == 0:
            raise NameError('Error! The order mapping algorithm failed. No usable labels found.')
        good_labels_init = labels[good_labels]
        labels_unique_init = np.unique(good_labels_init)

        # Do a naive set from all labels
        order_locs_lr_labeled = np.full(shape=(ny_lr, nx_lr), fill_value=np.nan)
        for l in range(good_points[:, 0].size):
            order_locs_lr_labeled[good_lr[0][l], good_lr[1][l]] = labels[l]
        
        # Flag all bad labels
        bad = np.where(order_locs_lr_labeled == -1)
        if bad[0].size > 0:
            order_locs_lr_labeled[bad] = np.nan

        # Further flag labels that don't span at least half the detector
        # If not bad, then fit.
        orders_list = []
        for l in range(labels_unique_init.size):
            label_locs = np.where(order_locs_lr_labeled == labels_unique_init[l])
            rx_lr_max = np.max(label_locs[1]) - np.min(label_locs[1])
            if rx_lr_max < 0.8 * (last_x_lr - first_x_lr):
                order_locs_lr_labeled[label_locs] = np.nan
            else:
                rys_lr = np.empty(nx_lr, dtype=np.float64)
                for x in range(nx_lr):
                    colx_ylocs = np.where(label_locs[1] == x)[0]
                    if colx_ylocs.size == 0:
                        rys_lr[x] = np.nan
                    else:
                        rys_lr[x] = np.max(label_locs[0][colx_ylocs]) - np.min(label_locs[0][colx_ylocs])
                wmin = pcmath.weighted_median(rys_lr, percentile=0.05)
                wmax = pcmath.weighted_median(rys_lr, percentile=0.95)
                good = np.where(rys_lr > 0.8*wmax)[0]
                good_finite = np.where(np.isfinite(rys_lr))[0]
                rys_lr_good = rys_lr[good[0]:good[-1]]
                diffs = np.diff(rys_lr_good)
                if good.size < 0.8 * good_finite.size:
                    order_locs_lr_labeled[label_locs] = np.nan
                else:
                    pfit = np.polyfit(label_locs[1] * Mx, label_locs[0] * My, self.trace_pos_poly_order)
                    height = np.nanmedian(rys_lr) * My
                    orders_list.append([{'label': len(orders_list) + 1, 'pcoeffs': pfit, 'height': height}])

        
        # Now fill in a full frame image
        n_orders = len(orders_list)
        order_image = np.full(shape=(ny, nx), dtype=float, fill_value=np.nan)
        
        for o in range(n_orders):
            for x in range(nx):
                pmodel = np.polyval(orders_list[o][0]['pcoeffs'], x)
                ymax = int(pmodel + height / 2)
                ymin = int(pmodel - height / 2)
                if ymin > ny - 1 or ymax < 0:
                    continue
                if ymax > ny - 1 - self.mask_top:
                    continue
                if ymin < 0 + self.mask_bottom:
                    continue
                order_image[ymin:ymax, x] = int(orders_list[o][0]['label'])
                
        order_map.orders_list = orders_list
        order_map.save(order_image)

