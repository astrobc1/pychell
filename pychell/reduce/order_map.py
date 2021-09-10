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
    
class PredeterminedTrace(OrderTracer):
    pass

class DensityClusterTracer(OrderTracer):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, trace_pos_poly_order=2, refine_position=False, order_spacing=10, mask_left=200, mask_right=200, mask_top=20, mask_bottom=20, n_orders=1):
        self.trace_pos_poly_order = trace_pos_poly_order
        self.refine_position = refine_position
        self.order_spacing = order_spacing
        self.n_orders = n_orders
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
        
        
    def trace_dev(self, order_map, reducer):
        
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
        #source_image_smooth = pcmath.median_filter2d(source_image, width=5, preserve_nans=False)
        
        # Normalize the flat.
        #source_image_smooth_norm = pcmath.normalize_image(source_image_smooth, window=self.order_spacing, n_knots=self.n_orders, percentile=0.99)
        source_image_smooth_norm = fits.open("/Users/gj_876/Desktop/source_image_smooth_norm.fits")[0].data
        
        # Downsample in the horizontal direction to save on memory
        down_sample_factor = 8
        nx_lr = int(nx / down_sample_factor)

        # Only consider regions where the flux is greater than 50%
        order_locations_lr = np.full((ny, nx_lr), np.nan)
        source_image_smooth_norm_lr = source_image_smooth_norm[:, ::down_sample_factor]
        good_lr = np.where(source_image_smooth_norm_lr > 0.5)
        order_locations_lr[good] = 1
        
        # Store points for DBSCAN
        good_points_lr = np.empty(shape=(good[0].size, 2), dtype=float)
        good_points[:, 0], good_points[:, 1] = good[1], good[0]
        
        # Create the Density clustering object and run
        density_cluster = sklearn.cluster.DBSCAN(eps=0.7*self.order_spacing, min_samples=5, metric='euclidean', algorithm='auto', p=None, n_jobs=reducer.n_cores)
        db = density_cluster.fit(good_points)
        breakpoint()
        
        # Extract the labels
        labels = db.labels_
        good_labels = np.where(labels >= 0)[0]
        if good_labels.size == 0:
            raise NameError(f"Error! The order mapping algorithm failed. No usable labels found for {order_map}.")
        good_labels_init = labels[good_labels]
        labels_unique_init = np.unique(good_labels_init)
        
        # Do a naive set from all labels
        order_locs_labeled = np.full((ny, nx), np.nan)
        for l in range(good_points[:, 0].size):
            order_locs_labeled[good[0][l], good[1][l]] = labels[l]

        # Flag all bad labels
        bad = np.where(order_locs_labeled == -1)
        if bad[0].size > 0:
            order_locs_labeled[bad] = np.nan

        # Further flag labels that don't span at least half the detector
        # If not bad, then fit.
        orders_list = []
        for l in range(labels_unique_init.size):
            label_locs = np.where(order_locs_labeled == labels_unique_init[l])
            rx_max = np.max(label_locs[1]) - np.min(label_locs[1])
            rys = np.empty(nx, dtype=np.float64)
            for x in range(nx):
                colx_ylocs = np.where(label_locs[1] == x)[0]
                if colx_ylocs.size == 0:
                    rys[x] = np.nan
                else:
                    rys[x] = np.max(label_locs[0][colx_ylocs]) - np.min(label_locs[0][colx_ylocs]) + 1
            wmin = pcmath.weighted_median(rys, percentile=0.05)
            wmax = pcmath.weighted_median(rys, percentile=0.95)
            good = np.where(rys > 0.8*wmax)[0]
            if good.size < 20:
                continue
            good_finite = np.where(np.isfinite(rys))[0]
            rys_good = rys[good[0]:good[-1]]
            height = pcmath.weighted_median(rys_good, percentile=0.75)
            pfit = np.polyfit(label_locs[1], label_locs[0], self.trace_pos_poly_order)
            orders_list.append({'order': len(orders_list) + 1, 'pcoeffs': pfit, 'height': height, 'fiber': None})
            
            
        # Warn if not equal
        if n_orders != self.n_orders:
            warnings.warn(f"The number of orders [{n_orders}] does not match the expected number [{self.n_orders}]")
                
        # Now fill in a full frame image
        n_orders = len(orders_list)
        order_image = np.full(shape=(ny, nx), dtype=float, fill_value=np.nan)
            
        for o in range(n_orders):
            for x in range(nx):
                pmodel = np.polyval(orders_list[o]['pcoeffs'], x)
                ymax = int(pmodel + orders_list[o]['height'] / 2)
                ymin = int(pmodel - orders_list[o]['height'] / 2)
                if ymin > ny - 1 or ymax < 0:
                    continue
                if ymax > ny - 1 - self.mask_top:
                    continue
                if ymin < 0 + self.mask_bottom:
                    continue
                order_image[ymin:ymax, x] = int(orders_list[o]['order'])
                
        

        breakpoint()
        order_map.orders_list = orders_list
        order_map.save(order_image)