import pychell.rvs.post_parser as parser
import pychell.rvs.rvcalc as pcrvcalc
import numpy as np
import matplotlib.pyplot as plt
import pychell
import pychell.rvs.forward_models as pcfoward_models
import os
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")
import datetime
from pdb import set_trace as stop

def combine_rvs(output_path_root, bad_rvs_dict=None, do_orders=None, iter_num=None, templates=None, method=None, use_rms=False):
    
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
    if iter_num is None:
        iter_nums = np.zeros(n_orders).astype(int) + n_iters - 1
    elif iter_num == 'best':
        _, iter_nums = get_best_iterations(rvs_dict, rvs_dict['n_obs_nights'])
    else:
        iter_nums = np.zeros(n_orders).astype(int) + iter_num

    # Parse the RMS and rvs, single iteration
    if use_rms:
        rms_all = parser.parse_rms(output_path_root, do_orders=do_orders)
        rms = np.zeros((n_orders, n_spec))
        for o in range(n_orders):
            rms[o, :] = rms_all[o, :, iter_nums[o] + index_offset]
    else:
        rms = None
            
    rvs = np.zeros((n_orders, n_spec))
    unc_nightly = np.zeros((n_orders, n_nights))
    rvs_nightly = np.zeros((n_orders, n_nights))
    for o in range(n_orders):
        rvs[o, :] = rvs_dict['rvs'][o, :, iter_nums[o]]
        rvs_nightly[o, :] = rvs_dict['rvs_nightly'][o, :, iter_nums[o]]
        unc_nightly[o, :] = rvs_dict['unc_nightly'][o, :, iter_nums[o]]
        
        
    if templates is not None:
        stellar_templates = parser.parse_stellar_templates(output_path_root, do_orders=do_orders, iter_nums=iter_nums)
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


def get_best_iterations(rvs_dict, n_obs_nights):
    
    n_iters = rvs_dict['rvs'].shape[2]
    n_orders = rvs_dict['rvs'].shape[0]
    best_iters = np.zeros(n_orders, dtype=int)
    best_stddevs = np.zeros(n_orders, dtype=int)
    for o in range(n_orders):
        stddevs = np.zeros(n_iters) + np.nan
        for i in range(n_iters):
            stddevs[i] = np.nanstd(rvs_dict['rvs'][o, :, i])
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