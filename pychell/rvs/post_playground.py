import pychell.rvs.post_parser as parser
import pychell.rvs.rvcalc as pcrvcalc
import numpy as np
import matplotlib.pyplot as plt
import pychell
import pychell.rvs.forward_models as pcfoward_models
import os
import scipy.signal
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")
import datetime
from pdb import set_trace as stop

def combine_rvs(output_path_root, bad_rvs_dict=None, do_orders=None, iter_index=None, templates=False, method=None, use_rms=False, debug=False, xcorr=False):
    """Combines RVs across orders through a weighted TFA scheme.

    Args:
        output_path_root (str): The full output path for this run.
        bad_rvs_dict (dict, optional): A bad rvs dictionary. Possible keys are 1. 'bad_spec' with an item being a list of bad bad spectra. These spectra for all orders are flagged. 2. 'bad_nights' where all observations on that night are flagged. Defaults to None.
        do_orders (list, optional): A list of which orders to work with. Defaults to None, including all orders.
        iter_index (int or str, optional): Which iteration index to use. Use 'best' for  the iteration with the lowest long term stddev. Defaults to the last index.
        templates (bool, optional): Whether or not to compute the rv content from the stellar template and consider that for weights. Defaults to None.
        method (str, optional): Which method in rvcalc to call. Defaults to combine_orders.
        use_rms (bool, optional): Whether or not to consider the rms of the fits as weights. Defaults to False.
        debug (bool, optional): If True, the code stops using pdb.set_trace() before exiting this function. Defaults to False.
        xcorr (bool, optional): Whether or not to use the xcorr RVs instead of the NM RVs. Defaults to False.
    Returns:
        tuple: The results returned by the call to method.
    """
    # Get the orders
    if do_orders is None:
        do_orders = parser.get_orders(output_path_root)
    n_orders = len(do_orders)
    
    # The method to combine rvs with
    if method is None:
        rv_method = getattr(pcrvcalc, 'combine_orders_fast')
    else:
        rv_method = getattr(pcrvcalc, method)
    
    # Get the tag for this run
    fwm_temp = parser.parse_forward_model(output_path_root, do_orders[0], 1)
    tag = fwm_temp.tag + '_' + datetime.date.today().strftime("%d%m%Y")
    index_offset = int(not fwm_temp.models_dict['star'].from_synthetic)
    
    # Parse the RVs
    rvs_dict = parser.parse_rvs(output_path_root, do_orders=do_orders)
    
    # Number of spectra and nights
    n_spec = np.sum(rvs_dict['n_obs_nights'])
    n_nights = len(rvs_dict['n_obs_nights'])
    n_iters = rvs_dict['rvs'].shape[2]
    
    # Determine which iteration to use
    if iter_index is None:
        iter_indexes = np.zeros(n_orders).astype(int) + n_iters - 1
    elif iter_index == 'best':
        _, iter_indexes = get_best_iterations(rvs_dict, xcorr)
    else:
        iter_indexes = np.zeros(n_orders).astype(int) + iter_index
        
    # Summary of rvs
    print_rv_summary(rvs_dict, do_orders, iter_indexes, xcorr)

    # Parse the RMS and rvs, single iteration
    if use_rms:
        rms_all = parser.parse_rms(output_path_root, do_orders=do_orders)
        rms = np.zeros((n_orders, n_spec))
        for o in range(n_orders):
            rms[o, :] = rms_all[o, :, iter_indexes[o] + index_offset]
    else:
        rms = None
            
    rvs = np.zeros((n_orders, n_spec))
    unc_nightly = np.zeros((n_orders, n_nights))
    rvs_nightly = np.zeros((n_orders, n_nights))
    for o in range(n_orders):
        if xcorr:
            rvs[o, :] = rvs_dict['rvsx'][o, :, iter_indexes[o]]
            rvs_nightly[o, :] = rvs_dict['rvsx_nightly'][o, :, iter_indexes[o]]
            unc_nightly[o, :] = rvs_dict['uncx_nightly'][o, :, iter_indexes[o]]
        else:
            rvs[o, :] = rvs_dict['rvs'][o, :, iter_indexes[o]]
            rvs_nightly[o, :] = rvs_dict['rvs_nightly'][o, :, iter_indexes[o]]
            unc_nightly[o, :] = rvs_dict['unc_nightly'][o, :, iter_indexes[o]]
        
    if templates is not None:
        stellar_templates = parser.parse_stellar_templates(output_path_root, do_orders=do_orders, iter_indexes=iter_indexes)
        rvcs = np.zeros(n_orders)
        for o in range(n_orders):
            _, rvcs[o] = pcrvcalc.compute_rv_content(stellar_templates[o][:, 0], stellar_templates[o][:, 1], snr=100, blaze=True, ron=0, R=80000)
    else:
        rvcs = np.zeros(n_orders) + np.nanmedian(unc_nightly)

    # Generate weights
    weights = gen_rv_weights(n_orders, bad_rvs_dict, rvs_dict['n_obs_nights'], rms=rms, rvcs=None)
    
    # Combine the orders via tfa, sort of
    rvs_out = rv_method(rvs, rvs_nightly, unc_nightly, weights, rvs_dict['n_obs_nights'])
    
    # Plot the final rvs
    fname = output_path_root + tag + '_final_rvs.png'
    plot_final_rvs(rvs_dict['BJDS'], rvs_dict['BJDS_nightly'], *rvs_out, phase_to=None, show=True, fname=None)
        
    # Save to a text file
    fname = output_path_root + tag + '_final_rvs.txt'
    np.savetxt(fname, np.array([rvs_dict['BJDS'], rvs_out[0], rvs_out[1]]).T, delimiter=',')
    fname = output_path_root + tag + '_final_rvs_nightly.txt'
    np.savetxt(fname, np.array([rvs_dict['BJDS_nightly'], rvs_out[2], rvs_out[3]]).T, delimiter=',')
    
    if debug:
        stop()
        
    return rvs_out
    
def lsperiodogram(t, rvs, pmin=1.3, pmax=None):
    """Computes a Lomb-Scargle periodogram.

    Args:
        t (np.ndarray): The independent variable.
        rvs (np.ndarray): The dependent variable.
        pmin (float, optional): . Defaults to 1.3.
        pmax (float, optional): The max period to consider. Defaults to 1.5 * time_baseline
    Returns:
        np.ndarray: The periods.
        np.ndarray: The LS periodogram
    """
    dt = np.nanmax(t) - np.nanmin(t)
    good = np.where(np.isfinite(rvs))[0]
    tp = np.arange(pmin, 1.5*dt, .001)
    af = 2 * np.pi / tp
    pgram = scipy.signal.lombscargle(t, rvs, af)
    return tp, pgram
    
def gen_rv_mask_single_order(bad_rvs_dict, n_obs_nights):
    n_nights = len(n_obs_nights)
    n_spec = np.sum(n_obs_nights)
    mask = np.ones(n_spec, dtype=float)
    if 'bad_nights' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_nights']:
            mask[pcfoward_models.ForwardModel.get_all_spec_indices_from_night(i, n_obs_nights)] = 0
    
    if 'bad_spec' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_spec']:
            mask[i] = 0
    
    return mask

def print_rv_summary(rvs_dict, do_orders, iter_indexes, xcorr):
    
    n_ord, _, n_iters = rvs_dict['rvs'].shape
    
    for o in range(n_ord):
        print('Order ' + str(do_orders[o]))
        for k in range(n_iters):
            if xcorr:
                stddev = np.nanstd(rvs_dict['rvsx_nightly'][o, :, k])
            else:
                stddev = np.nanstd(rvs_dict['rvs_nightly'][o, :, k])
            if k == iter_indexes[o]:
                print(' ** Iteration ' +  str(k + 1) + ': ' + str(round(stddev, 4)) + ' m/s')
            else:
                print('    Iteration ' +  str(k + 1) + ': ' + str(round(stddev, 4)) + ' m/s')
            

def get_best_iterations(rvs_dict, xcorr):
    
    n_iters = rvs_dict['rvs'].shape[2]
    n_orders = rvs_dict['rvs'].shape[0]
    best_iters = np.zeros(n_orders, dtype=int)
    best_stddevs = np.zeros(n_orders, dtype=int)
    for o in range(n_orders):
        stddevs = np.zeros(n_iters) + np.nan
        for k in range(n_iters):
            if xcorr:
                stddevs[k] = np.nanstd(rvs_dict['rvsx_nightly'][o, :, k])
            else:
                stddevs[k] = np.nanstd(rvs_dict['rvs_nightly'][o, :, k])
        best_iters[o] = np.nanargmin(stddevs)
        best_stddevs[o] = stddevs[best_iters[o]]
    return stddevs, best_iters

def gen_rv_mask(n_orders, bad_rvs_dict, n_obs_nights):
    n_nights = len(n_obs_nights)
    n_spec = np.sum(n_obs_nights)
    mask = np.ones(shape=(n_orders, n_spec), dtype=float)
    if 'bad_nights' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_nights']:
            mask[:, pcfoward_models.ForwardModel.get_all_spec_indices_from_night(i, n_obs_nights)] = 0
    
    if 'bad_spec' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_spec']:
            mask[:, i] = 0
            
    return mask
            
def gen_rv_weights(n_orders, bad_rvs_dict, n_obs_nights, rms=None, rvcs=None):
    
    # The number of spectra
    n_spec = np.sum(n_obs_nights)
    
    # Generate mask
    mask = gen_rv_mask(n_orders, bad_rvs_dict, n_obs_nights)
    
    # RMS weights
    if rms is not None:
        weights_rms = 1 / rms**2
        weights_rms *= mask
    else:
        weights_rms = np.ones_like(mask)
    weights_rms /= np.nansum(weights_rms)
        
    # RV content weights
    if rvcs is not None:
        weights_rvcont = np.outer(1 / rvcs**2, np.ones(n_spec))
    else:
        weights_rvcont = np.copy(mask)
    
    # Combine weights
    weights = weights_rvcont * weights_rms
    
    # Normalize
    weights /= np.nansum(weights)

    return weights

def plot_final_rvs(bjds, bjds_nightly, rvs_single, unc_single, rvs_nightly, unc_nightly, phase_to=None, show=True, fname=None):
    
    if phase_to is None:
        phase_to = 1E20
    
    # Single rvs
    plt.errorbar(bjds%phase_to, rvs_single-np.nanmedian(rvs_single), yerr=unc_single, linewidth=0, elinewidth=1, marker='.', markersize=10, markerfacecolor='pink', color='green', alpha=0.8)

    # Nightly RVs
    plt.errorbar(bjds_nightly%phase_to, rvs_nightly-np.nanmedian(rvs_nightly), yerr=unc_nightly, linewidth=0, elinewidth=3, marker='o', markersize=10, markerfacecolor='blue', color='grey', alpha=0.9)
    
    if show:
        plt.show()
    else:
        if fname is not None:
            plt.savefig(fname)