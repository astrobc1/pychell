# Base Python
import pickle
import glob
import datetime
import copy
import os
import gc

# Maths
import numpy as np
import scipy.stats

# Graphics
import matplotlib.pyplot as plt

# optimize deps
from optimize.knowledge import BoundedParameters

# Pychell deps
import pychell.spectralmodeling.rvcalc as pcrvcalc
import pychell.utils as pcutils
import pychell.maths as pcmath

#################
#### PARSING ####
#################

def parse_problem(path, order_num):
    
    print(f"Loading in Spectral RV Problem for Order {order_num}")
    fname = glob.glob(f"{path}Order{order_num}{os.sep}*spectralrvprob*.pkl")[0]
    with open(fname, 'rb') as f:
        specrvprob = pickle.load(f)
    return specrvprob
    
def parse_problems(path, do_orders):
    return [parse_problem(path, order_num) for order_num in do_orders]

def parse_fit_metrics(specrvprobs):
    n_orders = len(specrvprobs)
    fit_metrics = np.empty(shape=(n_orders, specrvprobs[0].n_spec, specrvprobs[0].n_iterations), dtype=float)
    for o in range(n_orders):
        for ispec in range(specrvprobs[0].n_spec):
            fit_metrics[o, ispec, :] = [specrvprobs[o].opt_results[ispec, k]["fbest"] for k in range(specrvprobs[0].n_iterations)]
    return fit_metrics
            
def parse_parameters(specrvprobs):
    n_orders = len(specrvprobs)
    pars = np.empty(shape=(n_orders, specrvprobs[0].n_spec, specrvprobs[0].n_iterations), dtype=BoundedParameters)
    for o in range(n_orders):
        for ispec in range(specrvprobs[0].n_spec):
            pars[o, ispec, :] = [specrvprobs[o].opt_results[ispec, k]["pbest"] for k in range(specrvprobs[0].n_iterations)]
    return pars

def parse_rvs(path, do_orders):
    
    # Define new dictionary containing rvs from all orders
    rvs_dict = {}

    # Load in a single forward model object to determine some values
    fname = glob.glob(f"{path}Order{do_orders[0]}{os.sep}RVs{os.sep}*.npz")[0]
    rvs0 = np.load(fname)
    n_spec, n_iterations = rvs0['rvsfwm'].shape
    n_orders = len(do_orders)
    n_nights = len(rvs0['n_obs_nights'])
    rvs_dict['bjds'] = rvs0['bjds']
    rvs_dict['bjds_nightly'] = rvs0['bjds_nightly']
    rvs_dict['n_obs_nights'] = rvs0['n_obs_nights']
    
    # Create arrays
    rvs_dict['rvsfwm'] = np.full(shape=(n_orders, n_spec, n_iterations), fill_value=np.nan)
    rvs_dict['rvsfwm_nightly'] = np.full(shape=(n_orders, n_nights, n_iterations), fill_value=np.nan)
    rvs_dict['uncfwm_nightly'] = np.full(shape=(n_orders, n_nights, n_iterations), fill_value=np.nan)
    rvs_dict['rvsxc'] = np.full(shape=(n_orders, n_spec, n_iterations), fill_value=np.nan)
    rvs_dict['uncxc'] = np.full(shape=(n_orders, n_spec, n_iterations), fill_value=np.nan)
    rvs_dict['rvsxc_nightly'] = np.full(shape=(n_orders, n_nights, n_iterations), fill_value=np.nan)
    rvs_dict['uncxc_nightly'] = np.full(shape=(n_orders, n_nights, n_iterations), fill_value=np.nan)
    rvs_dict['skew'] = np.full(shape=(n_orders, n_spec, n_iterations), fill_value=np.nan)

    # Load in rvs for each order
    for o in range(n_orders):
        print(f"Loading in RVs for Order {do_orders[o]}")
        fname = glob.glob(f"{path}Order{do_orders[o]}{os.sep}RVs{os.sep}*.npz")[0]
        rvfile = np.load(fname)
        rvs_dict['rvsfwm'][o, :, :] = rvfile['rvsfwm']
        rvs_dict['rvsfwm_nightly'][o, :] = rvfile['rvsfwm_nightly']
        rvs_dict['uncfwm_nightly'][o, :] = rvfile['uncfwm_nightly']
        rvs_dict['rvsxc'][o, :, :] = rvfile['rvsxc']
        rvs_dict['uncxc'][o, :, :] = rvfile['uncxc']
        rvs_dict['rvsxc_nightly'][o, :, :] = rvfile['rvsxc_nightly']
        rvs_dict['uncxc_nightly'][o, :, :] = rvfile['uncxc_nightly']
        rvs_dict['skew'][o, :, :] = rvfile['skew']

    return rvs_dict
    
def parse_residuals(self):
    
    res = []
    for o in range(self.n_orders):
        
        res.append(np.empty(self.n_spec, self.n_chunks), dtype=np.ndarray)
        
        for ispec in range(self.n_spec):
            
            for ichunk in range(self.n_chunks):
            
                for k in range(self.n_iters_opt):
            
                    # Get the best fit parameters
                    pars = self.forward_models[o][ispec].opt_results[k + parser.index_offset][ichunk]['xbest']
            
                    # Build the model
                    wave_data, model_lr = parser.forward_models[o][ispec].build_full(pars, parser.forward_models[o].templates_dict)
                    _res = self.forward_models[o][ispec].data.flux - model_lr
                    res[-1][ispec, ichunk] = _res
    
    self.residuals = res
    return res

#################
#### ACTIONS ####
#################

def combine_bis(path, specrvprobs, rvs_dict, bad_rvs_dict, iter_indices=None):
    
    # Numbers
    n_orders = len(specrvprobs)
    do_orders = [specrvprobs[o].order_num for o in range(len(specrvprobs))]
    n_spec = specrvprobs[0].n_spec
    n_nights = specrvprobs[0].n_nights
    n_iterations = specrvprobs[0].n_iterations
    n_obs_nights = rvs_dict["n_obs_nights"]
    
    # Which iterations to use for each order
    if iter_indices is None:
        iter_indices = [n_iterations - 1] * n_orders
    
    # Mask rvs from user input
    mask = gen_rv_mask(specrvprobs, rvs_dict, bad_rvs_dict)
    
    # Combine BIS
    bis_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    weights_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    for o in range(n_orders):
        bis_single_iter[o, :] = rvs_dict["bis"][o, :, iter_indices[o]]
        weights_single_iter[o, :] = mask[o, :, iter_indices[o]] # 1  / rvs_dict["bis"][o, :, iter_indices[o]] * mask[]
    result = pcrvcalc.combine_relative_rvs(bis_single_iter, weights_single_iter, n_obs_nights)

    # Add to dictionary
    rvs_dict['skew_out'] = result[0]
    rvs_dict['uncbis_out'] = result[1]
    rvs_dict['skew_nightly_out'] = result[2]
    rvs_dict['uncskew_nightly_out'] = result[3]
      
def combine_rvs_iteratively(path, specrvprobs, rvs_dict, bad_rvs_dict, iter_indices=None, n_flag_iters=10, thresh=4):
    
    # Numbers
    n_orders = len(specrvprobs)
    do_orders = [specrvprobs[o].order_num for o in range(len(specrvprobs))]
    n_spec = specrvprobs[0].n_spec
    n_rv_chunks = len(rvs_dict['bjds_nightly'])
    n_iterations = specrvprobs[0].n_iterations
    n_obs_nights = rvs_dict["n_obs_nights"]

    # Overwrite shape of nightly rvs
    rvs_dict['rvsfwm_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    rvs_dict['uncfwm_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    rvs_dict['rvsxc_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    rvs_dict['uncxc_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    
    # Mask rvs from user input
    mask = gen_rv_mask(specrvprobs, rvs_dict, bad_rvs_dict)
    
    # Parse fit metrics
    fit_metrics = parse_fit_metrics(specrvprobs)
    
    # Re-combine nightly rvs for each order with the mask
    for o in range(n_orders):
        for j in range(n_iterations):
            
            # FwM RVs
            rvsfwm = rvs_dict['rvsfwm'][o, :, j]
            weights = mask[o, :, j] / fit_metrics[o, :, j]**2
            rvs_dict['rvsfwm_nightly'][o, :, j], rvs_dict['uncfwm_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsfwm, weights, n_obs_nights)
            
            # XC RVs
            rvsxc = rvs_dict['rvsxc'][o, :, j]
            rvs_dict['rvsxc_nightly'][o, :, j], rvs_dict['uncxc_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsxc, weights, n_obs_nights)
    
    # Which iterations to use for each order
    if iter_indices is None:
        iter_indices = [n_iterations - 1] * n_orders

    # Generate weights
    weights = gen_rv_weights(specrvprobs, rvs_dict, mask, iter_indices)
    
    # Get RVs for single iteration
    rvsfwm_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    weights_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    order_offsets = np.array([np.nanmedian(rvs_dict["rvsfwm"][o, :, iter_indices[o]]) for o in range(n_orders)])
    for o in range(n_orders):
        rvsfwm_single_iter[o, :] = rvs_dict["rvsfwm"][o, :, iter_indices[o]] - order_offsets[o]
        weights_single_iter[o, :] = weights[o, :, iter_indices[o]]
        
    # Iteratively Combine and flag RVs
    for i in range(n_flag_iters):
        print(f"Combining RVs, iteration {i+1}")
        result_fwm = pcrvcalc.combine_relative_rvs(rvsfwm_single_iter, weights_single_iter, n_obs_nights)
        rvs, unc, rvsn, uncn = (*result_fwm,)
    
        # Flag bad RVs
        n_bad = 0
        for inight, f, l in pcutils.nightly_iteration(n_obs_nights):
            rr, ww = rvsfwm_single_iter[:, f:l], weights_single_iter[:, f:l]
            if np.nansum(ww) == 0:
                continue
            res_norm = (rr - rvsn[inight])
            bad = np.where(np.abs(res_norm) > thresh * np.nanstd(res_norm))
            if bad[0].size > 0:
               rvsfwm_single_iter[:, f:l][bad] = np.nan
               weights_single_iter[:, f:l][bad] = 0
               n_bad += len(bad[0])
        if n_bad == 0:
            break

    # Add to dictionary
    rvs_dict['rvsfwm_out'] = result_fwm[0]
    rvs_dict['uncfwm_out'] = result_fwm[1]
    rvs_dict['rvsfwm_nightly_out'] = result_fwm[2]
    rvs_dict['uncfwm_nightly_out'] = result_fwm[3]
    
    # Write to files for radvel
    spectrograph = specrvprobs[0].spectrograph
    try:
        star_name = specrvprobs[0].spectral_model.star.star_name.replace(' ', '_')
    except:
        star_name = specrvprobs[0].target_dict['name']
    time_tag = pcutils.gendatestr(time=False)
    telvec_single = np.full(n_spec, spectrograph, dtype='<U20')
    telvec_nightly = np.full(n_rv_chunks, spectrograph, dtype='<U20')
    
    # FwM
    fname = f"{path}rvsfwm_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsfwm_out']) & np.isfinite(rvs_dict['uncfwm_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds'][good], rvs_dict['rvsfwm_out'][good], rvs_dict['uncfwm_out'][good], telvec_single[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")
    fname = f"{path}rvsfwm_nightly_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsfwm_nightly_out']) & np.isfinite(rvs_dict['uncfwm_nightly_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds_nightly'][good], rvs_dict['rvsfwm_nightly_out'][good], rvs_dict['uncfwm_nightly_out'][good], telvec_nightly[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")
  
def combine_rvs(path, specrvprobs, rvs_dict, bad_rvs_dict, iter_indices=None):
    
    # Numbers
    n_orders = len(specrvprobs)
    do_orders = [specrvprobs[o].order_num for o in range(len(specrvprobs))]
    n_spec = specrvprobs[0].n_spec
    n_rv_chunks = len(rvs_dict['bjds_nightly'])
    n_iterations = specrvprobs[0].n_iterations
    n_obs_nights = rvs_dict["n_obs_nights"]

    # Overwrite shape of nightly rvs
    rvs_dict['rvsfwm_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    rvs_dict['uncfwm_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    rvs_dict['rvsxc_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    rvs_dict['uncxc_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    
    # Mask rvs from user input
    mask = gen_rv_mask(specrvprobs, rvs_dict, bad_rvs_dict)
    
    # Parse fit metrics
    fit_metrics = parse_fit_metrics(specrvprobs)
    
    # Re-combine nightly rvs for each order with the mask
    for o in range(n_orders):
        for j in range(n_iterations):

            if (j == 0 and specrvprobs[0].spectral_model.star.from_flat):
                continue
            
            # FwM RVs
            rvsfwm = rvs_dict['rvsfwm'][o, :, j]
            weights = mask[o, :, j] / fit_metrics[o, :, j]**2
            rvs_dict['rvsfwm_nightly'][o, :, j], rvs_dict['uncfwm_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsfwm, weights, n_obs_nights)
            
            # XC RVs
            rvsxc = rvs_dict['rvsxc'][o, :, j]
            rvs_dict['rvsxc_nightly'][o, :, j], rvs_dict['uncxc_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsxc, weights, n_obs_nights)
        
    # Summary of rvs
    print_rv_summary(rvs_dict, do_orders, iter_indices)
    
    # Which iterations to use for each order
    if iter_indices is None:
        iter_indices = [n_iterations - 1] * n_orders

    # Generate weights
    weights = gen_rv_weights(specrvprobs, rvs_dict, mask, iter_indices)
    
    # Combine RVs for NM
    rvsfwm_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    weights_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    for o in range(n_orders):
        rvsfwm_single_iter[o, :] = rvs_dict["rvsfwm"][o, :, iter_indices[o]]
        weights_single_iter[o, :] = weights[o, :, iter_indices[o]]
    result_fwm = pcrvcalc.combine_relative_rvs(rvsfwm_single_iter, weights_single_iter, n_obs_nights)

    # Add to dictionary
    rvs_dict['rvsfwm_out'] = result_fwm[0]
    rvs_dict['uncfwm_out'] = result_fwm[1]
    rvs_dict['rvsfwm_nightly_out'] = result_fwm[2]
    rvs_dict['uncfwm_nightly_out'] = result_fwm[3]

    # Combine RVs for XC
    rvsxc_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    weights_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    for o in range(n_orders):
        rvsxc_single_iter[o, :] = rvs_dict["rvsxc"][o, :, iter_indices[o]]
        weights_single_iter[o, :] = weights[o, :, iter_indices[o]]
    result_xc = pcrvcalc.combine_relative_rvs(rvsxc_single_iter, weights_single_iter, n_obs_nights)

    # Add to dictionary
    rvs_dict['rvsxc_out'] = result_xc[0]
    rvs_dict['uncxc_out'] = result_xc[1]
    rvs_dict['rvsxc_nightly_out'] = result_xc[2]
    rvs_dict['uncxc_nightly_out'] = result_xc[3]
    
    # Write to files for radvel
    spectrograph = specrvprobs[0].spectrograph
    try:
        star_name = specrvprobs[0].spectral_model.star.star_name.replace(' ', '_')
    except:
        star_name = specrvprobs[0].target_dict['name']
    time_tag = pcutils.gendatestr(time=False)
    telvec_single = np.full(n_spec, spectrograph, dtype='<U20')
    telvec_nightly = np.full(n_rv_chunks, spectrograph, dtype='<U20')
    
    # FwM
    fname = f"{path}rvsfwm_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsfwm_out']) & np.isfinite(rvs_dict['uncfwm_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds'][good], rvs_dict['rvsfwm_out'][good], rvs_dict['uncfwm_out'][good], telvec_single[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")
    fname = f"{path}rvsfwm_nightly_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsfwm_nightly_out']) & np.isfinite(rvs_dict['uncfwm_nightly_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds_nightly'][good], rvs_dict['rvsfwm_nightly_out'][good], rvs_dict['uncfwm_nightly_out'][good], telvec_nightly[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")
        
    # XC
    fname = f"{path}rvsxc_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsxc_out']) & np.isfinite(rvs_dict['uncxc_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds'][good], rvs_dict['rvsxc_out'][good], rvs_dict['uncxc_out'][good], telvec_single[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")
    fname = f"{path}rvsxc_nightly_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsxc_nightly_out']) & np.isfinite(rvs_dict['uncxc_nightly_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds_nightly'][good], rvs_dict['rvsxc_nightly_out'][good], rvs_dict['uncxc_nightly_out'][good], telvec_nightly[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")

def combine_rvs_simple(path, specrvprobs, rvs_dict, bad_rvs_dict, iter_indices=None, templates=None):

    # Numbers
    n_orders = len(specrvprobs)
    do_orders = [specrvprobs[o].order_num for o in range(len(specrvprobs))]
    n_spec = specrvprobs[0].n_spec
    n_rv_chunks = len(rvs_dict['bjds_nightly'])
    n_iterations = specrvprobs[0].n_iterations
    n_obs_nights = rvs_dict["n_obs_nights"]

    # Overwrite shape of nightly rvs
    rvs_dict['rvsfwm_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    rvs_dict['uncfwm_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    rvs_dict['rvsxc_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    rvs_dict['uncxc_nightly'] = np.full((n_orders, n_rv_chunks, n_iterations), np.nan)
    
    # Mask rvs from user input
    mask = gen_rv_mask(specrvprobs, rvs_dict, bad_rvs_dict)
    
    # Parse fit metrics
    fit_metrics = parse_fit_metrics(specrvprobs)
    
    # Re-combine nightly rvs for each order with the mask
    for o in range(n_orders):
        for j in range(n_iterations):

            if (j == 0 and specrvprobs[0].spectral_model.star.from_flat):
                continue
            
            # FwM RVs
            rvsfwm = rvs_dict['rvsfwm'][o, :, j]
            weights = mask[o, :, j] / fit_metrics[o, :, j]**2
            rvs_dict['rvsfwm_nightly'][o, :, j], rvs_dict['uncfwm_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsfwm, weights, n_obs_nights)
            
            # XC RVs
            rvsxc = rvs_dict['rvsxc'][o, :, j]
            rvs_dict['rvsxc_nightly'][o, :, j], rvs_dict['uncxc_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsxc, weights, n_obs_nights)
        
    # Summary of rvs
    print_rv_summary(rvs_dict, do_orders, iter_indices)
    
    # Which iterations to use for each order
    if iter_indices is None:
        iter_indices = [n_iterations - 1] * n_orders

    # Generate weights
    weights = gen_rv_weights(specrvprobs, rvs_dict, mask, iter_indices)
    
    # Combine RVs for NM
    rvsfwm_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    weights_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    for o in range(n_orders):
        rvsfwm_single_iter[o, :] = rvs_dict["rvsfwm"][o, :, iter_indices[o]] - pcmath.weighted_mean(rvs_dict["rvsfwm"][o, :, iter_indices[o]], weights[o, :, iter_indices[o]])
        weights_single_iter[o, :] = weights[o, :, iter_indices[o]]
    result_fwm = pcrvcalc.combine_rvs_weighted_mean(rvsfwm_single_iter, weights_single_iter, n_obs_nights)

    # Add to dictionary
    rvs_dict['rvsfwm_out'] = result_fwm[0]
    rvs_dict['uncfwm_out'] = result_fwm[1]
    rvs_dict['rvsfwm_nightly_out'] = result_fwm[2]
    rvs_dict['uncfwm_nightly_out'] = result_fwm[3]

    # Combine RVs for XC
    rvsxc_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    weights_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    for o in range(n_orders):
        rvsxc_single_iter[o, :] = rvs_dict["rvsxc"][o, :, iter_indices[o]] - pcmath.weighted_mean(rvs_dict["rvsxc"][o, :, iter_indices[o]], weights[o, :, iter_indices[o]])
        weights_single_iter[o, :] = weights[o, :, iter_indices[o]]
    result_xc = pcrvcalc.combine_rvs_weighted_mean(rvsxc_single_iter, weights_single_iter, n_obs_nights)

    # Add to dictionary
    rvs_dict['rvsxc_out'] = result_xc[0]
    rvs_dict['uncxc_out'] = result_xc[1]
    rvs_dict['rvsxc_nightly_out'] = result_xc[2]
    rvs_dict['uncxc_nightly_out'] = result_xc[3]
    
    # Write to files for radvel
    spectrograph = specrvprobs[0].spectrograph
    star_name = specrvprobs[0].spectral_model.star.star_name.replace(' ', '_')
    time_tag = pcutils.gendatestr(time=False)
    telvec_single = np.full(n_spec, spectrograph, dtype='<U20')
    telvec_nightly = np.full(n_rv_chunks, spectrograph, dtype='<U20')
    
    # FwM
    fname = f"{path}rvsfwm_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsfwm_out']) & np.isfinite(rvs_dict['uncfwm_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds'][good], rvs_dict['rvsfwm_out'][good], rvs_dict['uncfwm_out'][good], telvec_single[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")
    fname = f"{path}rvsfwm_nightly_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsfwm_nightly_out']) & np.isfinite(rvs_dict['uncfwm_nightly_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds_nightly'][good], rvs_dict['rvsfwm_nightly_out'][good], rvs_dict['uncfwm_nightly_out'][good], telvec_nightly[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")
        
    # XC
    fname = f"{path}rvsxc_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsxc_out']) & np.isfinite(rvs_dict['uncxc_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds'][good], rvs_dict['rvsxc_out'][good], rvs_dict['uncxc_out'][good], telvec_single[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")
    fname = f"{path}rvsxc_nightly_{spectrograph}_{star_name}_{time_tag}.txt"
    good = np.where(np.isfinite(rvs_dict['rvsxc_nightly_out']) & np.isfinite(rvs_dict['uncxc_nightly_out']))[0]
    t, rvs, unc, telvec = rvs_dict['bjds_nightly'][good], rvs_dict['rvsxc_nightly_out'][good], rvs_dict['uncxc_nightly_out'][good], telvec_nightly[good]
    with open(fname, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([t, rvs, unc, telvec], dtype=object).T, fmt="%f,%f,%f,%s")

def plot_final_rvs(path, specrvprobs, rvs_dict, which="fwm", show=False, time_offset=2450000, figsize=(8, 4), dpi=200):
        
    # Unpack rvs
    bjds, bjds_nightly = rvs_dict['bjds'], rvs_dict['bjds_nightly']
    if which == 'fwm':
        rvs_single, unc_single, rvs_nightly, unc_nightly = rvs_dict['rvsfwm_out'], rvs_dict['uncfwm_out'], rvs_dict['rvsfwm_nightly_out'], rvs_dict['uncfwm_nightly_out']
    else:
        rvs_single, unc_single, rvs_nightly, unc_nightly = rvs_dict['rvsxc_out'], rvs_dict['uncxc_out'], rvs_dict['rvsxc_nightly_out'], rvs_dict['uncxc_nightly_out']
        
    # Figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Single rvs
    plt.errorbar(bjds - time_offset, rvs_single,
                 yerr=unc_single,
                 linewidth=0, elinewidth=1, marker='o', markersize=4, markerfacecolor=pcutils.COLORS_HEX_GADFLY[2],
                 color=pcutils.COLORS_HEX_GADFLY[2], mec='black', alpha=0.7, label="Single exposure")

    # Nightly RVs
    plt.errorbar(bjds_nightly - time_offset, rvs_nightly,
                 yerr=unc_nightly,
                 linewidth=0, elinewidth=2, marker='o', markersize=8, markerfacecolor=pcutils.COLORS_HEX_GADFLY[0],
                 color='black', mec='black', alpha=0.9, label="Co-added")
    
    # Title
    plt.title(f"{specrvprobs[0].spectral_model.star.star_name.replace('_', ' ')}, {specrvprobs[0].spectrograph} Relative RVs")
    
    # Labels
    plt.xlabel(f"BJD - {time_offset}", fontsize=16)
    plt.ylabel('RV [m/s]', fontsize=16)

    # Legend
    plt.legend()

    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    plt.savefig(f"{path}rvs_{specrvprobs[0].spectrograph.lower().replace(' ', '_')}_{specrvprobs[0].spectral_model.star.star_name.lower().replace(' ', '_')}.png")
    if show:
        plt.show()
    else:
        return fig

def parameter_corrs(path, specrvprobs, rvs_dict, n_cols=4):
    
    n_orders = len(specrvprobs)
    n_iterations = specrvprobs[0].n_iterations
    n_spec = specrvprobs[0].n_spec
    
    # Loop over orders and chunks
    for o in range(n_orders):
        
        # Get varied parameters
        pars_first_obs = specrvprobs[o].opt_results[0, -1]["pbest"]
        pars_first_obs_numpy = pars_first_obs.unpack()
        varied_inds = np.where(pars_first_obs_numpy["vary"])[0]
        n_vary = len(varied_inds)
        pars = np.empty(shape=(n_spec, n_iterations, n_vary), dtype=object)
        par_vals = np.full(shape=(n_spec, n_iterations, n_vary), dtype=float, fill_value=np.nan)
        par_names_vary = [pars_first_obs_numpy["name"][i] for i in range(len(pars_first_obs)) if pars_first_obs_numpy["vary"][i]]
        for ispec in range(n_spec):
            for j in range(n_iterations):
                for k in range(n_vary):
                    pars[ispec, j, k] = specrvprobs[o].opt_results[ispec, j]['pbest'][par_names_vary[k]]
                    par_vals[ispec, j, k] = pars[ispec, j, k].value
        
        n_rows = int(np.ceil(n_vary / n_cols))
        fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10), dpi=400)
        axarr = np.atleast_2d(axarr)
        
        for row in range(n_rows):
            for col in range(n_cols):
                
                # The par index
                k = n_cols * row + col
                if k + 1 > n_vary:
                    axarr[row, col].set_visible(False)
                    continue
                
                if not specrvprobs[0].spectral_model.star.from_flat:
                    rvs0 = rvs_dict['rvsfwm'][o, :, 0]
                    pars0 = par_vals[:, 0, k]
                    good0 = np.where(np.isfinite(rvs0) & np.isfinite(pars0))[0]
                    axarr[row, col].scatter(rvs0[good0], pars0[good0], marker='o', s=1, c='red', alpha=0.8)

                rvslast = rvs_dict['rvsfwm'][o, :, -1]
                parslast = par_vals[:, -1, k]
                goodlast = np.where(np.isfinite(rvslast) & np.isfinite(parslast))[0]
                axarr[row, col].scatter(rvslast[goodlast], parslast[goodlast], marker='o', s=4, c='black', alpha=0.8)
                axarr[row, col].set_xlabel('RV [m/s]', fontsize=16)
                axarr[row, col].set_ylabel(par_names_vary[k].replace('_', ' '), fontsize=16)
                axarr[row, col].tick_params(axis='both', which='major', labelsize=16)
                axarr[row, col].text(np.max(rvslast[goodlast]), np.min(parslast[goodlast]), f"pcc {n_iterations}={round(scipy.stats.pearsonr(rvslast[goodlast], parslast[goodlast])[0], 2)}", horizontalalignment="right")
                
        plt.tight_layout()
        fname = f"{path}Order{specrvprobs[o].order_num}{os.sep}parameter_corrs_ord{specrvprobs[o].order_num}.png"
        plt.savefig(fname)
        plt.close()

def rvprec_vs_snr(path, specrvprobs, rvs_dict, bad_rvs_dict, show=False):

    # Numbers
    n_iterations = specrvprobs[0].n_iterations
    n_spec = specrvprobs[0].n_spec
    n_orders = len(specrvprobs)

    # Single exposures RMS
    rmss = parse_fit_metrics(specrvprobs)

    # Single exposures snr
    snrs_single = 1 / rmss

    # Nightly SNR
    snrs_nightly = compute_nightly_snrs(specrvprobs, rvs_dict)

    # Convert to resolution elt
    ang_per_pix = np.zeros(len(specrvprobs), dtype=float)
    fwhms = np.zeros(len(specrvprobs), dtype=float)
    for o in range(n_orders):
        pbest = specrvprobs[o].opt_results[0, -1]["pbest"]
        fwhms[o] = pcmath.sigmatofwhm(pbest["hermite_lsf_width"].value)
        wave_data, _ = specrvprobs[o].spectral_model.build(pbest)
        good = np.where(np.isfinite(wave_data))[0]
        ang_per_pix[o] = (wave_data[-1] - wave_data[0]) / (good[-1] - good[0])
        snrs_single[o, :, :] = snrs_single[o, :, :] * np.sqrt(fwhms[o] / ang_per_pix[o])
        snrs_nightly[o, :, :] = snrs_nightly[o, :, :] * np.sqrt(fwhms[o] / ang_per_pix[o])

    # Average over orders
    snrs_nightly = np.nanmean(snrs_nightly[:, :, -1], axis=0)

    # Average over orders
    snrs_single = np.nanmean(snrs_single[:, :, -1], axis=0)

    # Combine RVs
    combine_rvs(path, specrvprobs, rvs_dict, bad_rvs_dict, iter_indices=n_orders * [n_iterations - 1])

    # Uncertainty of single exposure rvs
    unc_single = rvs_dict["uncfwm_out"]

    # Uncertainty of nightly rvs
    unc_nightly = rvs_dict["uncfwm_nightly_out"]

    return snrs_single, unc_single, snrs_nightly, unc_nightly
    
def compute_empirical_rvprec(path, specrvprobs, rvs_dict, templates=None, iter_indices=None, inject_blaze=True):

    # Numbers
    n_orders = len(specrvprobs)
    n_spec = specrvprobs[0].n_spec
    n_nights = len(rvs_dict["n_obs_nights"])

    # Compute rv contents for each night
    rvcontents_theo_single_exposures = compute_rv_contents(specrvprobs, rvs_dict, inject_blaze=inject_blaze)
    
    # Really want to compare within a night
    rvcontents_theo = np.zeros(shape=(n_orders, n_nights), dtype=float)
    rvcontents_emp = np.zeros(shape=(n_orders, n_nights), dtype=float)

    # Extract the desired iteration
    for o in range(n_orders):
        for i, f, l in pcutils.nightly_iteration(rvs_dict['n_obs_nights']):
            v = np.nansum(1 / rvcontents_theo_single_exposures[o, f:l, iter_indices[o]]**2)**-0.5
            if np.isfinite(v):
                rvcontents_theo[o, i] = v
            else:
                rvcontents_theo[o, i] = np.nan

            # Empirical are just the per-order error bars, co-added across exposures
            rvcontents_emp[o, i] = rvs_dict["uncfwm_nightly"][o, i, iter_indices[o]]

    # Average theoretical and empirical across nights
    rvcontents_theo = np.nanmean(rvcontents_theo, axis=1)
    rvcontents_emp = np.nanmean(rvcontents_emp, axis=1)
    
    # Mean wavelength of each order
    mean_waves = np.zeros(n_orders, dtype=float)

    for o in range(n_orders):

        # Find first good observation
        k = 0
        for data in specrvprobs[o].data:
            if data.is_good:
                k = data.spec_num - 1
                break

        # Build the best fit model for the kth observation to find the mean wavelength
        pbest = specrvprobs[o].opt_results[k, iter_indices[o]]["pbest"]
        specrvprobs[o].spectral_model.initialize(pbest, specrvprobs[o].data[k], iter_indices[o], specrvprobs[o].stellar_templates)
        wave_data = specrvprobs[o].spectral_model.wls.build(pbest)
        mean_waves[o] = np.nanmean(wave_data)

    # Sort according to wavelength
    ss = np.argsort(mean_waves)
    mean_waves = mean_waves[ss]
    rvcontents_emp = rvcontents_emp[ss]
    rvcontents_theo = rvcontents_theo[ss]

    return mean_waves, rvcontents_emp, rvcontents_theo


###############
#### MISC. ####
###############

def spec_indices_from_night(night_index, n_obs_night):
    if night_index == 0:
        return np.arange(n_obs_night[0]).astype(int)
    else:
        s = np.cumsum(n_obs_night)[night_index - 1]
        return np.arange(s, s + n_obs_night[night_index]).astype(int)
     
def gen_rv_mask(specrvprobs, rvs_dict, bad_rvs_dict):
    
    # Numbers
    n_orders = len(specrvprobs)
    n_spec = specrvprobs[0].n_spec
    n_iterations = specrvprobs[0].n_iterations
    n_obs_nights = rvs_dict["n_obs_nights"]
    
    # Initialize a mask
    mask = np.ones(shape=(n_orders, n_spec, n_iterations), dtype=float)
    
    # Mask all spectra for a given night
    if 'bad_nights' in bad_rvs_dict:
        for inight in bad_rvs_dict['bad_nights']:
            inds = spec_indices_from_night(inight, n_obs_nights)
            mask[:, inds, :] = 0
            rvs_dict['rvsfwm'][:, inds, :] = np.nan
            rvs_dict['rvsxc'][:, inds, :] = np.nan
            rvs_dict['skew'][:, inds, :] = np.nan
    
    # Mask individual spectra
    if 'bad_spec' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_spec']:
            mask[:, i, :] = 0
            rvs_dict['rvsfwm'][:, i, :] = np.nan
            rvs_dict['rvsxc'][:, i, :] = np.nan
            rvs_dict['skew'][:, i, :] = np.nan
        
    return mask

def gen_rv_weights(specrvprobs, rvs_dict, mask, iter_indices):
    
    # Numbers
    n_orders = len(specrvprobs)
    n_spec = specrvprobs[0].n_spec
    n_iterations = specrvprobs[0].n_iterations

    # RV Content
    rvcontents = compute_rv_contents(specrvprobs, rvs_dict, inject_blaze=False)
    
    # RV content weights
    #weights = np.zeros((n_orders, n_spec, n_iterations))
    weights = mask / rvcontents**2
    #rvsxc_unc = rvs_dict['uncxc']
    #for o in range(n_orders):
        #for j in range(n_iterations):
            #weights[o, :, j] = 1 / rvsxc_unc[o, :, j]**2
    
    # Mask weights
    #weights *= mask

    return weights

def compute_rv_contents(specrvprobs, rvs_dict, inject_blaze=False):

    # SNR0
    snr0 = 100

    # Numbers
    n_orders = len(specrvprobs)
    n_spec = specrvprobs[0].n_spec
    n_nights = len(rvs_dict["n_obs_nights"])
    n_iterations = specrvprobs[0].n_iterations

    # The RV contents, for each iteration (lower is "better")
    rvcontents = np.zeros((n_orders, n_spec, n_iterations), dtype=float)

    # The mean wavelength of each order
    mean_waves = np.zeros(n_orders, dtype=float)

    # S/N
    snrs_single = 1 / parse_fit_metrics(specrvprobs)

    # Compute RVC for each order and iteration
    for o in range(n_orders):

        # Find first good observation
        k = 0
        for data in specrvprobs[o].data:
            if data.is_good:
                k = data.spec_num - 1
                break

        # Use parameters for the kth osbervation, last iter - Doesn't so much matter here.
        pars = specrvprobs[o].opt_results[k, -1]['pbest']

        # Initialize
        specrvprobs[o].spectral_model.initialize(pars, specrvprobs[o].data[k], -1, specrvprobs[o].stellar_templates)

        # Data wave grid
        data_wave_k = specrvprobs[o].spectral_model.wls.build(pars)

        # Mean wavelength
        mean_waves[o] = np.nanmean(data_wave_k)
        
        # Original templates
        templates_dictcp = copy.deepcopy(specrvprobs[o].spectral_model.templates_dict)
        
        # Compute RVC for this iteration
        for j in range(n_iterations):

            # Skip first iteration if starting from flat
            if specrvprobs[0].spectral_model.star.from_flat and j == 0:
                continue

            # Best fit pars for the kth observation
            pars = specrvprobs[o].opt_results[k, j]["pbest"]

            # Initialize the kth observation
            specrvprobs[o].spectral_model.initialize(pars, specrvprobs[o].data[k], j, specrvprobs[o].stellar_templates)

            # Data wave grid
            data_wave = specrvprobs[o].spectral_model.wls.build(pars)
            
            # Set the star in the templates dict
            specrvprobs[o].spectral_model.templates_dict["star"] = np.copy(specrvprobs[o].stellar_templates[j])
        
            # Alias the model wave grid
            model_wave = specrvprobs[o].spectral_model.model_wave

            # Formula: derivative comes from gas cell / stellar flux, but overall flux further accounts for tellurics and blaze
            _, rvcontents[o, :, j] = pcrvcalc.compute_rv_content(pars, specrvprobs[o].spectral_model, snr=snr0)

            # Scale to S/N
            rvcontents[o, :, j] *= (snr0 / snrs_single[o, :, j])
            
        # Reset templates dict
        specrvprobs[o].spectral_model.templates_dict = templates_dictcp
        
    # Return
    return rvcontents

def compute_nightly_snrs(specrvprobs, rvs_dict):
    
    # Numbers
    n_orders = len(specrvprobs)
    n_spec = specrvprobs[0].n_spec
    n_nights = len(rvs_dict["n_obs_nights"])
    n_iterations = specrvprobs[0].n_iterations
    n_obs_nights = rvs_dict["n_obs_nights"]

    # Parse the rms
    rms = parse_fit_metrics(specrvprobs)
    nightly_snrs = np.zeros((n_orders, n_nights, n_iterations))

    # Loop over orders, iterations, and observations
    for o in range(n_orders):
        for i, f, l in pcutils.nightly_iteration(n_obs_nights):
            for j in range(n_iterations):
                nightly_snrs[o, i, j] = np.nansum((1 / rms[o, f:l, j])**2)**0.5
                
    return nightly_snrs

def print_rv_summary(rvs_dict, do_orders, iter_indices):
    
    # Numbers
    n_orders, n_spec, n_iterations = rvs_dict["rvsfwm"].shape
    
    for o in range(n_orders):
        print(f"Order {do_orders[o]}")
        for k in range(n_iterations):
            stddev = np.nanstd(rvs_dict['rvsfwm_nightly'][o, :, k])
            stddevxc = np.nanstd(rvs_dict['rvsxc_nightly'][o, :, k])
                
            if k == iter_indices[o]:
                print(f" ** Iteration {k + 1} [FwM]: {round(stddev, 4)} m/s")
                print(f" ** Iteration {k + 1} [XC]: {round(stddevxc, 4)} m/s")
            else:
                print(f"    Iteration {k + 1} [FwM]: {round(stddev, 4)} m/s")
                print(f"    Iteration {k + 1} [XC]: {round(stddevxc, 4)} m/s")