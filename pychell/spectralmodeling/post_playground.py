# Base Python
import pickle
import glob
import datetime
import copy
import os

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
import pychell.maths as pcmaths

#################
#### PARSING ####
#################

def parse_problem(path, order_num):
    
    print(f"Loading in Spectral RV Problem for Order {order_num}")
    fname = glob.glob(f"{path}Order{order_num}{os.sep}*spectralrvprob*.pkl")[0]
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
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
    rvs_dict['rvsxc_nightly'] = np.full(shape=(n_orders, n_nights, n_iterations), fill_value=np.nan)
    rvs_dict['uncxc_nightly'] = np.full(shape=(n_orders, n_nights, n_iterations), fill_value=np.nan)
    rvs_dict['bis'] = np.full(shape=(n_orders, n_spec, n_iterations), fill_value=np.nan)

    # Load in rvs for each order
    for o in range(n_orders):
        print(f"Loading in RVs for Order {do_orders[o]}")
        fname = glob.glob(f"{path}Order{do_orders[o]}{os.sep}RVs{os.sep}*.npz")[0]
        rvfile = np.load(fname)
        rvs_dict['rvsfwm'][o, :, :] = rvfile['rvsfwm']
        rvs_dict['rvsfwm_nightly'][o, :] = rvfile['rvsfwm_nightly']
        rvs_dict['uncfwm_nightly'][o, :] = rvfile['uncfwm_nightly']
        rvs_dict['rvsxc'][o, :, :] = rvfile['rvsxc']
        rvs_dict['rvsxc_nightly'][o, :, :] = rvfile['rvsxc_nightly']
        rvs_dict['uncxc_nightly'][o, :, :] = rvfile['uncxc_nightly']
        rvs_dict['bis'][o, :, :] = rvfile['bis']

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
    rvs_dict['bis_out'] = result[0]
    rvs_dict['uncbis_out'] = result[1]
    rvs_dict['bis_nightly_out'] = result[2]
    rvs_dict['uncbis_nightly_out'] = result[3]
    
    
    
def rvs_quicklook(path, rvs_dict, do_orders, iter_indices=None):
    
    # Numbers
    n_orders, n_spec, n_iterations = rvs_dict["rvsfwm"].shape
    n_obs_nights = rvs_dict["n_obs_nights"]
    n_nights = len(n_obs_nights)
    
    # Which iterations to use for each order
    if iter_indices is None:
        iter_indices = [n_iterations - 1] * n_orders
        
    # BJDs nightly
    bjds_nightly = rvs_dict["bjds_nightly"]
    
    # Combine RVs
    
    # Plot each order
    for o in range(n_orders):
        breakpoint()
        rvs = rvs_dict["rvsfwm_nightly"][o, :, iter_indices[o]] - np.nanmedian(rvs_dict["rvsfwm_nightly"][o, :, iter_indices[o]])
        rverr = rvs_dict["rvserrfwm_nightly"][o, :, iter_indices[o]]
        plt.errorbar(bjds_nightly, rvs, yerr=rverr, marker='o', lw=0, elinewidth=0.5, label=f"Order {do_orders[o]}")
    
    # Coadded
    plt.errorbar(bjds_nightly, rvs, yerr=rverr, marker='o', lw=0, elinewidth=0.5, label=f"Order {do_orders[o]}")
  
  
def combine_rvs2(path, specrvprobs, rvs_dict, bad_rvs_dict, iter_indices=None, templates=None, n_flag_iters=3):
    
    # Numbers
    n_orders = len(specrvprobs)
    do_orders = [specrvprobs[o].order_num for o in range(len(specrvprobs))]
    n_spec = specrvprobs[0].n_spec
    n_nights = specrvprobs[0].n_nights
    n_iterations = specrvprobs[0].n_iterations
    n_obs_nights = rvs_dict["n_obs_nights"]
    
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
    weights = gen_rv_weights(specrvprobs, rvs_dict, mask, iter_indices, templates=templates)
    
    # Get RVs for single iteration
    rvsfwm_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    weights_single_iter = np.full(shape=(n_orders, n_spec), fill_value=np.nan)
    for o in range(n_orders):
        rvsfwm_single_iter[o, :] = rvs_dict["rvsfwm"][o, :, iter_indices[o]] - np.nanmedian(rvs_dict["rvsfwm"][o, :, iter_indices[o]])
        weights_single_iter[o, :] = weights[o, :, iter_indices[o]]
        
    # Iteratively Combine and flag RVs
    for i in range(n_flag_iters):
        result_fwm = pcrvcalc.combine_relative_rvs(rvsfwm_single_iter, weights_single_iter, n_obs_nights)
        rvs, unc, rvsn, uncn = (*result_fwm,)
        
        plt.errorbar(rvs_dict["bjds_nightly"], rvsn - np.nanmedian(rvsn), yerr=uncn, marker='o', lw=0, elinewidth=0.5, label="Iter 1, FwM")
        for o in range(n_orders):
            plt.plot(rvs_dict["bjds"], rvsfwm_single_iter[o, :], marker='.', lw=0, label=f"Order {do_orders[o]}")
        plt.show()
    
        # Flag bad RVs
        for inight, f, l in pcutils.nightly_iteration(n_obs_nights):
            wstddev = pcmaths.weighted_stddev(rvsfwm_single_iter[:, f:l].flatten(), weights_single_iter[:, f:l].flatten())
            bad = np.where(np.abs(rvsfwm_single_iter[:, f:l] - rvsn[inight]) > 4 * wstddev)
            if bad[0].size > 0:
                rvsfwm_single_iter[:, f:l][bad] = np.nan
                weights_single_iter[:, f:l][bad] = 0

    # Add to dictionary
    rvs_dict['rvsfwm_out'] = result_fwm[0]
    rvs_dict['uncfwm_out'] = result_fwm[1]
    rvs_dict['rvsfwm_nightly_out'] = result_fwm[2]
    rvs_dict['uncfwm_nightly_out'] = result_fwm[3]
    
    # Write to files for radvel
    spectrograph = specrvprobs[0].spectrograph
    star_name = specrvprobs[0].target_dict["name"].replace(' ', '_')
    time_tag = datetime.date.today().strftime('%d%m%Y')
    telvec_single = np.full(n_spec, spectrograph, dtype='<U20')
    telvec_nightly = np.full(n_nights, spectrograph, dtype='<U20')
    
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
  
def combine_rvs(path, specrvprobs, rvs_dict, bad_rvs_dict, iter_indices=None, templates=None):
    
    # Numbers
    n_orders = len(specrvprobs)
    do_orders = [specrvprobs[o].order_num for o in range(len(specrvprobs))]
    n_spec = specrvprobs[0].n_spec
    n_nights = specrvprobs[0].n_nights
    n_iterations = specrvprobs[0].n_iterations
    n_obs_nights = rvs_dict["n_obs_nights"]
    
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
        
    # Summary of rvs
    print_rv_summary(rvs_dict, do_orders, iter_indices)
    
    # Which iterations to use for each order
    if iter_indices is None:
        iter_indices = [n_iterations - 1] * n_orders

    # Generate weights
    weights = gen_rv_weights(specrvprobs, rvs_dict, mask, iter_indices, templates=templates)
    
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
    star_name = specrvprobs[0].target_dict["name"].replace(' ', '_')
    time_tag = datetime.date.today().strftime('%d%m%Y')
    telvec_single = np.full(n_spec, spectrograph, dtype='<U20')
    telvec_nightly = np.full(n_nights, spectrograph, dtype='<U20')
    
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
        
def plot_final_rvs(path, specrvprobs, rvs_dict, which="xc"):
        
    # Unpack rvs
    bjds, bjds_nightly = rvs_dict['bjds'], rvs_dict['bjds_nightly']
    if which == 'fwm':
        rvs_single, unc_single, rvs_nightly, unc_nightly = rvs_dict['rvsfwm_out'], rvs_dict['uncfwm_out'], rvs_dict['rvsfwm_nightly_out'], rvs_dict['uncfwm_nightly_out']
    else:
        rvs_single, unc_single, rvs_nightly, unc_nightly = rvs_dict['rvsxc_out'], rvs_dict['uncxc_out'], rvs_dict['rvsxc_nightly_out'], rvs_dict['uncxc_nightly_out']
        
    # Figure
    fig = plt.figure(figsize=(8, 4), dpi=100)
    
    # Single rvs
    plt.errorbar(bjds - 2450000, rvs_single-np.nanmedian(rvs_single),
                 yerr=unc_single,
                 linewidth=0, elinewidth=1, marker='.', markersize=10, markerfacecolor='pink',
                 color='green', alpha=0.8)

    # Nightly RVs
    plt.errorbar(bjds_nightly - 2450000, rvs_nightly-np.nanmedian(rvs_nightly),
                 yerr=unc_nightly,
                 linewidth=0, elinewidth=2, marker='o', markersize=10, markerfacecolor='blue',
                 color='grey', alpha=0.9)
    
    # Title
    plt.title(f"{specrvprobs[0].target_dict['name'].replace('_', ' ')}, {specrvprobs[0].spectrograph} Relative RVs")
    
    # Labels
    plt.xlabel("BJD - 2450000")
    plt.ylabel('RV [m/s]')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    plt.savefig(f"{path}rvs_{specrvprobs[0].spectrograph.lower().replace(' ', '_')}_{specrvprobs[0].target_dict['name'].lower().replace(' ', '_')}.png")
    plt.show()

def parameter_corrs(path, specrvprobs, rvs_dict):
    
    n_orders = len(specrvprobs)
    n_iterations = specrvprobs[0].n_iterations
    n_spec = specrvprobs[0].n_spec
    
    # Loop over orders and chunks
    for o in range(n_orders):
        
        # Get varied parameters
        pars_first_order = specrvprobs[o].opt_results[0, -1]["pbest"]
        pars_first_order_numpy = pars_first_order.unpack()
        varied_inds = np.where(pars_first_order_numpy["vary"])[0]
        n_vary = len(varied_inds)
        pars = np.empty(shape=(n_spec, n_iterations, n_vary), dtype=object)
        par_vals = np.full(shape=(n_spec, n_iterations, n_vary), dtype=float, fill_value=np.nan)
        par_names_vary = [pars_first_order_numpy["name"][i] for i in range(len(pars_first_order)) if pars_first_order_numpy["vary"][i]]
        for ispec in range(n_spec):
            for j in range(n_iterations):
                for k in range(n_vary):
                    pars[ispec, j, k] = specrvprobs[o].opt_results[ispec, j]['pbest'][par_names_vary[k]]
                    par_vals[ispec, j, k] = pars[ispec, j, k].value
        
        n_cols = 5
        n_rows = int(np.ceil(n_vary / n_cols))
        fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 15), dpi=400)
        
        for row in range(n_rows):
            for col in range(n_cols):
                
                # The par index
                k = n_cols * row + col
                if k + 1 > n_vary:
                    axarr[row, col].set_visible(False)
                    continue
                
                # Views to arrays
                rvs0, rvslast = rvs_dict['rvsfwm'][o, :, 0], rvs_dict['rvsfwm'][o, :, -1]
                pars0, parslast = par_vals[:, 0, k], par_vals[:, -1, k]
                
                axarr[row, col].scatter(rvs0, pars0, marker='o', s=1, c='red', alpha=0.7)
                axarr[row, col].scatter(rvslast, parslast, marker='o', s=1, c='black', alpha=0.7)
                axarr[row, col].set_xlabel('RV [m/s]', fontsize=4)
                axarr[row, col].set_ylabel(par_names_vary[k].replace('_', ' '), fontsize=4)
                axarr[row, col].tick_params(axis='both', which='major', labelsize=4)
                good0 = np.where(np.isfinite(rvs0) & np.isfinite(pars0))[0]
                goodlast = np.where(np.isfinite(rvslast) & np.isfinite(parslast))[0]
                axarr[row, col].text(np.max(rvs0[good0]), np.max(pars0[good0]), f"pcc 1={round(scipy.stats.pearsonr(rvs0[good0], pars0[good0])[0], 2)}", horizontalalignment="right")
                axarr[row, col].text(np.max(rvs0[good0]), np.min(pars0[good0]), f"pcc {n_iterations}={round(scipy.stats.pearsonr(rvslast[goodlast], parslast[goodlast])[0], 2)}", horizontalalignment="right")
                
        plt.tight_layout()
        fname = f"{path}Order{specrvprobs[o].order_num}{os.sep}parameter_corrs_ord{specrvprobs[o].order_num}.png"
        plt.savefig(fname)
        plt.close()

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
            rvs_dict['bis'][:, inds, :] = np.nan
    
    # Mask individual spectra
    if 'bad_spec' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_spec']:
            mask[:, i, :] = 0
            rvs_dict['rvsfwm'][:, i, :] = np.nan
            rvs_dict['rvsxc'][:, i, :] = np.nan
            rvs_dict['bis'][:, i, :] = np.nan
        
    return mask

def gen_rv_weights(specrvprobs, rvs_dict, mask, iter_indices, templates):
    
    # Numbers
    n_orders = len(specrvprobs)
    n_spec = specrvprobs[0].n_spec
    n_iterations = specrvprobs[0].n_iterations
    
    # RMS weights
    fit_metrics = parse_fit_metrics(specrvprobs)
    weights_fit = 1 / fit_metrics**2
    bad = np.where(weights_fit < 100)
    if bad[0].size > 0:
        weights_fit[bad] = 0
        
    # RV content weights
    weights_rvcont = np.zeros((n_orders, n_spec, n_iterations))
    rvconts = compute_rv_contents(specrvprobs, templates)
    for o in range(n_orders):
        for j in range(n_iterations):
            weights_rvcont[o, :, j] = 1 / rvconts[o, j]**2
    
    # Combine weights, multiplicatively
    weights = weights_rvcont * weights_fit * mask

    return weights

def compute_rv_contents(specrvprobs, templates=None):
    
    if templates is None:
        templates = ["star"]
        
    # Numbers
    n_orders = len(specrvprobs)
    n_spec = specrvprobs[0].n_spec
    n_iterations = specrvprobs[0].n_iterations
            
    # The RV contents, for each iteration (lower is "better")
    rvcs = np.zeros((n_orders, n_iterations))
    
    # The nightly S/N, for each iteration
    nightly_snrs = compute_nightly_snrs(specrvprobs)
    
    # Compute RVC for each order and iteration
    for o in range(n_orders):
        
        # Original templates
        templates_dictcp = copy.deepcopy(specrvprobs[o].spectral_model.templates_dict)
        
        # Compute RVC for this iteration
        for j in range(n_iterations):
        
            # Use parameters for the first osbervation - Doesn't so much matter here.
            pars = specrvprobs[o].opt_results[0, j]['pbest']
            
            # Set the star in the templates dict
            specrvprobs[o].spectral_model.templates_dict["star"] = np.copy(specrvprobs[o].stellar_templates[j])
        
            # Alias the model wave grid
            model_wave = specrvprobs[o].spectral_model.model_wave
        
            # Data wave grid
            data_wave = specrvprobs[o].spectral_model.wavelength_solution.build(pars)
        
            # LSF
            if specrvprobs[o].spectral_model.lsf is not None:
                lsf = specrvprobs[o].spectral_model.lsf.build(pars)
        
            # RV Content for each template individually
            rvcs_per_template = np.zeros(len(templates))
        
            # Loop over templates
            for i, template_key in enumerate(templates):
            
                # Build the high res template
                template_flux = getattr(specrvprobs[o].spectral_model, template_key).build(pars, specrvprobs[o].spectral_model.templates_dict[template_key], model_wave)
            
                # The S/N for this observation
                snr = np.nanmedian(nightly_snrs[o, :, j])
            
                # Compute content for this template
                _, rvcs_per_template[i] = pcrvcalc.compute_rv_content(model_wave, template_flux, snr=snr, blaze=True, ron=0, wave_to_sample=data_wave)
           
            # Add in quadrature
            rvcs[o, j] = np.sqrt(np.nansum(rvcs_per_template**2))
            
        # Reset templates dict
        specrvprobs[o].spectral_model.templates_dict = templates_dictcp
        
    # Return
    return rvcs

def compute_nightly_snrs(specrvprobs):
    
    # Numbers
    n_orders = len(specrvprobs)
    n_spec = specrvprobs[0].n_spec
    n_nights = specrvprobs[0].n_nights
    n_iterations = specrvprobs[0].n_iterations
    n_obs_nights = specrvprobs[0].rvs_dict["n_obs_nights"]

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