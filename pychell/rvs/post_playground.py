import pychell.rvs.post_parser as parser
import pychell.rvs.rvcalc as pcrvcalc
import numpy as np
import matplotlib.pyplot as plt
import pychell
import pychell.rvs.forward_models as pcfoward_models
import pychell.maths as pcmath
import os
import scipy.constants as cs
import copy
import scipy.signal
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")
import datetime
from pdb import set_trace as stop

def combine_stellar_templates(output_path_root, do_orders=None, iter_index=None):
    
    if do_orders is None:
        do_orders = parser.get_orders(output_path_root)
        
    n_orders = len(do_orders)
    
    stellar_templates = parser.parse_stellar_templates(output_path_root, do_orders=do_orders, iter_indexes=[iter_index]*n_orders)
    nxs = np.array([stellar_templates[o][:, 0].size for o in range(n_orders)])
    wave_min, wave_max = np.nanmin(stellar_templates[0][:, 0]), np.nanmax(stellar_templates[-1][:, 0])
    nx_master = int(np.average(nxs) * n_orders)
    master_template_wave = np.linspace(wave_min, wave_max, num=nx_master)
    
    stellar_templates_interp = np.zeros((nx_master, n_orders))
    for o in range(n_orders):
        good = np.where(np.isfinite(stellar_templates[o][:, 0]) & np.isfinite(stellar_templates[o][:, 1]))[0]
        stellar_templates_interp[:, o] = scipy.interpolate.CubicSpline(stellar_templates[o][good, 0], stellar_templates[o][good, 1], extrapolate=False)(master_template_wave)
        
    master_template_flux = np.nanmean(stellar_templates_interp, axis=1)
    
    np.savetxt(output_path_root + 'master_stellar_template.txt', np.array([master_template_wave, master_template_flux]).T, delimiter=',')
        

def combine_rvs(output_path_root, bad_rvs_dict=None, do_orders=None, iter_index=None, templates=False, method=None, use_rms=False, debug=False, xcorr=False, phase_to=None, tc=None, kamp=None, forward_models=None, detrend=False, bis_thresh=None):
    """Combines RVs across orders.

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
        phase_to (float, optional): The period to phase to for plotting the RVs.
        tc (float, optional): The tc to phase to for plotting the RVs.
    Returns:
        tuple: The results returned by the call to method.
    """
    # Get the orders
    if do_orders is None:
        do_orders = parser.get_orders(output_path_root)
    n_orders = len(do_orders)
    
    # The method to combine rvs with
    if method is None:
        rv_method = getattr(pcrvcalc, 'compute_relative_rvs_from_all')
    else:
        rv_method = getattr(pcrvcalc, method)
    
    # Parse forward models
    if forward_models is None:
        forward_models = parser.parse_forward_models(output_path_root, do_orders=do_orders)
        
    tag = forward_models[0, 0].tag + '_' + datetime.date.today().strftime("%d%m%Y")
    index_offset = int(not forward_models[0, 0].models_dict['star'].from_synthetic)
    star_name = forward_models[0, 0].star_name
    spectrograph = forward_models[0, 0].spectrograph
    
    # Parse the RVs
    rvs_dict = parser.parse_rvs(output_path_root, do_orders=do_orders)
    
    # Number of spectra for each night
    n_obs_nights = rvs_dict['n_obs_nights']
    
    # Mask rvs from user input
    rvs_dict, mask = gen_rv_mask(rvs_dict, bad_rvs_dict)
    
    # Numbers
    n_spec = np.sum(rvs_dict['n_obs_nights'])
    n_nights = len(rvs_dict['n_obs_nights'])
    n_iters_rvs = rvs_dict['rvs'].shape[2]
    n_iters_pars = n_iters_rvs + index_offset
    
    # Detrend RVs if applicable
    if detrend:
        for o in range(n_orders):
            for j in range(n_iters_rvs):
                good = np.where(np.isfinite(rvs_dict['rvsx'][o, :, j]) & np.isfinite(rvs_dict['BIS'][o, :, j]))[0]
                bad = np.where(~np.isfinite(rvs_dict['rvsx'][o, :, j]) | ~np.isfinite(rvs_dict['BIS'][o, :, j]))[0]
                if good.size == 0:
                    continue
                rvs_dict['rvsx'][o, good, j] = pcrvcalc.detrend_rvs(do_orders[o], rvs_dict['rvsx'][o, good, j], rvs_dict['BIS'][o, good, j], thresh=bis_thresh)
                        

    # Parse the RMS and rvs, single iteration
    rms_all = np.zeros((n_orders, n_spec, n_iters_pars))
    for o in range(n_orders):
        for i in range(n_spec):
            for j in range(n_iters_pars):
                rms_all[o, i, j] = forward_models[o, i].opt[j][0]
    
    # Regenerate nightly rvs
    for o in range(n_orders):
        for j in range(n_iters_rvs):
            
            # NM RVs
            rvs_dict['rvs_nightly'][o, :, j], rvs_dict['unc_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvs_dict['rvs'][o, :, j], 1 / rms_all[o, :, j + index_offset]**2, n_obs_nights, flag_outliers=True)
            
            # xcorr RVs
            if rvs_dict['do_xcorr']:
                rvs_dict['rvsx_nightly'][o, :, j], rvs_dict['uncx_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvs_dict['rvsx'][o, :, j], 1 / rms_all[o, :, j + index_offset]**2, n_obs_nights, flag_outliers=True)
            
    # Determine which iteration to use
    if iter_index is None:
        iter_indexes = np.zeros(n_orders).astype(int) + n_iters - 1
    elif iter_index == 'best':
        _, iter_indexes = get_best_iterations(rvs_dict, xcorr)
    else:
        iter_indexes = np.zeros(n_orders).astype(int) + iter_index
    
    # Get for RVs for the desired iterations
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
        
    # Summary of rvs
    print_rv_summary(rvs_dict, bad_rvs_dict, do_orders, iter_indexes, xcorr)
        
    # Get rms for all orders x spectra
    rms = np.zeros((n_orders, n_spec))
    for o in range(n_orders):
        rms[o, :] = rms_all[o, :, iter_indexes[o] + index_offset]
        
    # S / N
    snrs = np.nanmedian(1 / rms, axis=1)
    nightly_snrs = np.full(shape=(n_orders, n_nights), fill_value=np.nan)
    for o in range(n_orders):
        f, l = 0, n_obs_nights[0]
        for i in range(n_nights):
            nightly_snrs[o, i] = np.nansum((1 / rms[o, f:l])**2)**0.5
            if i < n_nights - 1:
                f += n_obs_nights[i]
                l += n_obs_nights[i+1]
            
    # Median nightly S / N for each order to compare against the photon limit
    nightly_snrs = np.nanmedian(nightly_snrs, axis=1)
    
    # Compute RV content of each order if set
    if templates is not None and len(templates) > 0:
        rvcs = np.zeros(n_orders)
        stellar_templates = parser.parse_stellar_templates(output_path_root, do_orders=do_orders, iter_indexes=iter_indexes)
        for o in range(n_orders):
            bad = np.where(forward_models[o, 0].data.badpix == 0)[0]
            wave = forward_models[o, 0].models_dict['wavelength_solution'].build(forward_models[o, 0].initial_parameters)
            wave[bad] = np.nan
            rvc = np.zeros(len(templates))
            for i, t in enumerate(templates):
                if t == 'star':
                    _, rvc[i] = pcrvcalc.compute_rv_content(stellar_templates[o][:, 0], stellar_templates[o][:, 1], snr=nightly_snrs[o], blaze=True, ron=0,width=forward_models[o, 0].initial_parameters[forward_models[o, 0].models_dict['lsf'].par_names[0]].value, sampling=None, wave_to_sample=wave)
                else:
                    _, rvc[i] = pcrvcalc.compute_rv_content(forward_models[o, 0].templates_dict[t][:, 0], forward_models[o, 0].templates_dict[t][:, 1], snr=nightly_snrs[o], blaze=True, ron=0,width=forward_models[o, 0].initial_parameters[forward_models[o, 0].models_dict['lsf'].par_names[0]].value, sampling=None, wave_to_sample=wave)
            rvcs[o] = np.nansum(rvc**2)**0.5
    else:
        rvcs = np.zeros(n_orders) + np.nanmedian(unc_nightly)

    # Generate weights
    rvs_dict, weights = gen_rv_weights(rvs_dict, bad_rvs_dict, rms=rms, rvcs=rvcs)
    
    # Combine the orders via tfa, sort of
    rvs_out = rv_method(rvs, rvs_nightly, unc_nightly, weights, rvs_dict['n_obs_nights'])
    
    # Plot the final rvs
    fname = output_path_root + tag + '_final_rvs.png'
    rvs_single_out, unc_single_out, rvs_nightly_out, unc_nightly_out = rvs_out[0], rvs_out[1], rvs_out[2], rvs_out[3]
    plot_final_rvs(star_name, spectrograph, rvs_dict['BJDS'], rvs_dict['BJDS_nightly'], rvs_single_out, unc_single_out, rvs_nightly_out, unc_nightly_out, phase_to=phase_to, show=True, fname=None, tc=tc, kamp=kamp)

    # Save to a text file
    fname = output_path_root + tag + '_final_rvs.txt'
    np.savetxt(fname, np.array([rvs_dict['BJDS'], rvs_single_out, unc_single_out]).T, delimiter=',')
    fname = output_path_root + tag + '_final_rvs_nightly.txt'
    np.savetxt(fname, np.array([rvs_dict['BJDS_nightly'], rvs_nightly_out, unc_nightly_out]).T, delimiter=',')
    
    # redchi2s, redchi2s_nightly = compute_redchi2s(rvs_out[0], rvs_out[1], rvs_out[2], rvs_out[3], rvs_dict['n_obs_nights'])
    
    # Plot the RV contents and error bars
    if templates is not None:
        plt.plot(do_orders, rvcs, label='Photon noise limit', lw=2, marker='X', markersize=12, mfc='hotpink', c='black')
        plt.title(star_name.replace('_', ' ') + ' ' + spectrograph + ' RV Precision')
        plt.plot(do_orders, np.nanmedian(unc_nightly, axis=1), label='Median Nightly RV uncertainties', lw=2, marker='X', markersize=12, mfc='green')
        plt.xticks(do_orders)
        plt.legend()
        plt.ylabel('$\sigma_{RV}$')
        plt.xlabel('Order')
        plt.show()
    
    #rvs_out += (redchi2s,)
    
    if debug:
        stop()
        
    return forward_models
    
def compute_redchi2s(rvs_single, unc_single, rvs_nightly, unc_nightly, n_obs_nights):
    n_nights = len(rvs_nightly)
    f, l = 0, n_obs_nights[0]
    redchi2s = np.zeros(n_nights)
    redchi2s_nightly = np.zeros(n_nights)
    redchi2s_nightly[:] = np.nan
    redchi2s[:] = np.nan
    for inight in range(n_nights):
        ng = np.where(np.isfinite(rvs_single[f:l]))[0].size
        if ng == 1:
            continue
        redchi2s[inight] = np.nansum(((rvs_single[f:l] - rvs_nightly[inight]) / unc_single[f:l])**2) / (ng - 1)
        if inight < n_nights - 1:
            f += n_obs_nights[inight]
            l += n_obs_nights[inight+1]
    return redchi2s, redchi2s_nightly
    
def lsperiodogram(t, rvs, pmin=1.3, pmax=None, dp=0.01):
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
    good = np.where(np.isfinite(rvs))[0]
    dt = np.nanmax(t[good]) - np.nanmin(t[good])
    tp = np.arange(pmin, 1.5*dt, dp)
    af = 2 * np.pi / tp
    pgram = scipy.signal.lombscargle(t[good], rvs[good] - np.median(rvs[good]), af)
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

def print_rv_summary(rvs_dict, bad_rvs_dict, do_orders, iter_indexes, xcorr):
    
    n_ord, _, n_iters = rvs_dict['rvs'].shape
    n_obs_nights = rvs_dict['n_obs_nights']
    
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
        stddevs = np.full(n_iters, fill_value=np.nan)
        for k in range(n_iters):
            if xcorr:
                stddevs[k] = np.nanstd(rvs_dict['rvsx_nightly'][o, :, k])
            else:
                stddevs[k] = np.nanstd(rvs_dict['rvs_nightly'][o, :, k])
        best_iters[o] = np.nanargmin(stddevs)
        best_stddevs[o] = stddevs[best_iters[o]]
    return stddevs, best_iters


def gen_rv_mask(rvs_dict, bad_rvs_dict):
    
    # Copy the dictionary
    rvs_dict_out = copy.deepcopy(rvs_dict)
    
    # Some numbers
    n_orders, n_spec, n_iters = rvs_dict['rvs'].shape
    n_obs_nights = rvs_dict['n_obs_nights']
    n_nights = len(n_obs_nights)
    
    # Initialize a mask
    mask = np.ones(shape=(n_orders, n_spec), dtype=float)
    
    # Mask all spectra for a given night
    if 'bad_nights' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_nights']:
            inds = pcfoward_models.ForwardModel.get_all_spec_indices_from_night(i, n_obs_nights)
            mask[:, inds] = 0
            rvs_dict_out['rvs'][:, inds, :] = np.nan
            if rvs_dict_out['do_xcorr']:
                rvs_dict_out['rvsx'][:, inds, :] = np.nan
                rvs_dict_out['BIS'][:, inds, :] = np.nan
    
    # Mask individual spectra
    if 'bad_spec' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_spec']:
            mask[:, i] = 0
            rvs_dict_out['rvs'][:, i, :] = np.nan
            if rvs_dict_out['do_xcorr']:
                rvs_dict_out['rvsx'][:, i, :] = np.nan
                rvs_dict_out['BIS'][:, i, :] = np.nan
            
    return rvs_dict_out, mask
            
def gen_rv_weights(rvs_dict, bad_rvs_dict, rms=None, rvcs=None):
    
    # Numbers
    n_orders, n_spec, n_iters = rvs_dict['rvs'].shape
    n_obs_nights =  rvs_dict['n_obs_nights']
    n_nights = len(n_obs_nights)
    
    rvs_dict_out = copy.deepcopy(rvs_dict)
    
    # Generate mask
    rvs_dict_out, mask = gen_rv_mask(rvs_dict_out, bad_rvs_dict)
    
    # RMS weights
    if rms is not None:
        weights_rms = 1 / rms**2
    else:
        weights_rms = np.ones_like(mask)
    weights_rms /= np.nansum(weights_rms)
        
    # RV content weights
    if rvcs is not None:
        weights_rvcont = np.outer(1 / rvcs**2, np.ones(n_spec))
    else:
        weights_rvcont = np.ones_like(mask)
    weights_rvcont /= np.nansum(weights_rvcont)
    
    # Combine weights.
    # NOTE: For multiplicative weights, the scaling of the individual weights does not matter.
    weights = weights_rvcont * weights_rms * mask
    
    # Normalize
    weights /= np.nansum(weights)

    return rvs_dict_out, weights



def parameter_corrs(output_path_root, bad_rvs_dict=None, do_orders=None, iter_index=None, templates=False, debug=False, xcorr=False, forward_models=None):
    
    # Get the orders
    if do_orders is None:
        do_orders = parser.get_orders(output_path_root)
    n_orders = len(do_orders)
    
    # Get the tag for this run
    if forward_models is None:
        forward_models = parser.parse_forward_models(output_path_root, do_orders=do_orders)
    tag = forward_models[0, 0].tag + '_' + datetime.date.today().strftime("%d%m%Y")
    index_offset = int(not forward_models[0, 0].models_dict['star'].from_synthetic)
    star_name = forward_models[0, 0].star_name
    spectrograph = forward_models[0, 0].spectrograph
    
    # Parse the RVs
    rvs_dict = parser.parse_rvs(output_path_root, do_orders=do_orders)
    
    rvs_dict, _ = gen_rv_mask(rvs_dict, bad_rvs_dict)
    
    # Number of spectra and nights
    n_spec = np.sum(rvs_dict['n_obs_nights'])
    n_nights = len(rvs_dict['n_obs_nights'])
    n_iters_rvs = rvs_dict['rvs'].shape[2]
    n_iters_pars = n_iters_rvs + index_offset
    
    # Determine which iteration to use
    if iter_index is None:
        iter_indexes = np.zeros(n_orders).astype(int) + n_iters_rvs - 1
    elif iter_index == 'best':
        _, iter_indexes = get_best_iterations(rvs_dict, xcorr)
    else:
        iter_indexes = np.zeros(n_orders).astype(int) + iter_index
    
    for o in range(n_orders):
        
        vp = forward_models[o, 0].best_fit_pars[iter_indexes[o] + index_offset].unpack(keys='varies')['varies']
        vpi = np.where(vp)[0]
        nv = vpi.size
        
        pars = np.empty(shape=(n_spec, n_iters_pars, nv), dtype=object)
        par_vals = np.full(shape=(n_spec, n_iters_pars, nv), dtype=float, fill_value=np.nan)
        par_names = list(forward_models[o, 0].best_fit_pars[iter_indexes[o] + index_offset].keys())
        par_names = [par_names[v] for v in vpi]
        for ispec in range(n_spec):
            for j in range(n_iters_pars):
                for k in range(nv):
                    pars[ispec, j, k] = forward_models[o, ispec].best_fit_pars[j][par_names[k]]
                    par_vals[ispec, j, k] = pars[ispec, j, k].value
        
        n_cols = 5
        n_rows = int(np.ceil(nv / n_cols))
        fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 12), dpi=400)
        
        for row in range(n_rows):
            for col in range(n_cols):
                
                # The par index
                k = n_cols * row + col
                
                if k + 1 > nv:
                    axarr[row, col].set_visible(False)
                    continue
                    
                n_iters_plot = np.min([10, n_iters_pars])
                for ispec in range(n_spec):
                    axarr[row, col].plot(rvs_dict['rvs'][o, ispec, -n_iters_plot:], par_vals[ispec, -n_iters_plot:, k], alpha=0.7, c='powderblue', lw=0.7)
                
                axarr[row, col].plot(rvs_dict['rvs'][o, :, iter_indexes[o]], par_vals[:, iter_indexes[o] + index_offset, k], marker='.', lw=0, c='black', markersize=8)
                axarr[row, col].set_xlabel('RV [m/s]', fontsize=4)
                axarr[row, col].set_ylabel(par_names[k].replace('_', ' '), fontsize=4)
                axarr[row, col].tick_params(axis='both', which='major', labelsize=4)
                axarr[row, col].grid(None)
        fig.suptitle(star_name.replace('_', ' ') + ' Parameter Correlations Order ' + str(do_orders[o]), fontsize=10)
        fname = output_path_root + 'Order' + str(do_orders[o]) + os.sep + tag + '_ord' + str(do_orders[o]) + '_parameter_corrs.png'
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.5)
        plt.savefig(fname)
        plt.close()
        
    if debug:
        stop()
        
    
def plot_final_rvs(star_name, spectrograph, bjds, bjds_nightly, rvs_single, unc_single, rvs_nightly, unc_nightly, phase_to=None, tc=None, kamp=None, show=True, fname=None):
    
    if phase_to is None:
        _phase_to = 1E20
    else:
        _phase_to = phase_to
        modelx = np.linspace(0, _phase_to, num=300)
        modely = kamp * np.sin(2 * np.pi * modelx / _phase_to)
        
    if tc is None:
        alpha = 0
    else:
        alpha = tc - _phase_to / 2
    
    # Single rvs
    plt.errorbar((bjds - alpha)%_phase_to, rvs_single-np.nanmedian(rvs_single), yerr=unc_single, linewidth=0, elinewidth=1, marker='.', markersize=10, markerfacecolor='pink', color='green', alpha=0.8)

    # Nightly RVs
    plt.errorbar((bjds_nightly - alpha)%_phase_to, rvs_nightly-np.nanmedian(rvs_nightly), yerr=unc_nightly, linewidth=0, elinewidth=3, marker='o', markersize=10, markerfacecolor='blue', color='grey', alpha=0.9)
    
    plt.title(star_name.replace('_', ' ') + ', ' + spectrograph + ' Relative RVs')
    if phase_to is None:
        plt.xlabel('BJD - BJD$_{0}$')
    else:
        plt.xlabel('Phase [days, P = ' +  str(round(_phase_to, 3)) + ']')
        plt.plot(modelx, modely, label='K = ' + str(kamp) + ' m/s')
    plt.ylabel('RV [m/s]')
    
    if show:
        plt.show()
    else:
        if fname is not None:
            plt.savefig(fname)
            
def plot_stellar_templates_single_iter(output_path_root, star_name, stellar_templates, do_orders, iter_indexes, unit='Ang'):
    
    if unit == 'microns':
        factor = 1E-4
    elif unit == 'Ang':
        factor = 1
    else:
        factor = 1E-1
    
    n_orders = len(do_orders)
    
    fig, axarr = plt.subplots(nrows=n_orders, ncols=1, figsize=(20, 16), dpi=250)
    axarr = np.atleast_1d(axarr)
    
    for o in range(n_orders):
        axarr[o].plot(stellar_templates[o][:, 0] * factor, stellar_templates[o][:, 1])
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indexes[o] + 1), fontsize=8)
        axarr[o].tick_params(axis='both', labelsize=10)
    axarr[-1].set_xlabel('Wavelength [' + unit + ']')
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.95, wspace=None, hspace=0.5)
    fig.text(0.03, 0.5, 'Norm. flux', rotation=90, verticalalignment='center', horizontalalignment='center', fontsize=20)
    fig.text(0.5, 0.97, star_name, fontsize=20, verticalalignment='center', horizontalalignment='center')
    plt.savefig(output_path_root + 'stellar_templates.png')
    
    plt.close()
            

def residual_coherence(output_path_root, do_orders, bad_rvs_dict, iter_indexes, frame='star', templates=[], unit='Ang', debug=False, star_name=None, nsample=1):
    
    n_orders = len(do_orders)
    
    forward_models = parser.parse_forward_models(output_path_root, do_orders=do_orders)
    
    stellar_templates = parser.parse_stellar_templates(output_path_root, do_orders=do_orders, iter_indexes=iter_indexes)
    
    if star_name is None:
        star_name = forward_models[0, 0].star_name
    
    if unit == 'microns':
        factor = 1E-4
    elif unit == 'Ang':
        factor = 1
    else:
        factor = 1E-1
    
    n_orders, n_spec = forward_models.shape
    nxhr = forward_models[0, 0].templates_dict['star'][:, 0].size
    plot_height_single = 4
    fig, axarr = plt.subplots(nrows=n_orders, ncols=1, figsize=(10, int(plot_height_single*n_orders)), dpi=250)
    axarr = np.atleast_1d(axarr)
    
    for o in range(n_orders):
        
        star_wave = forward_models[o, 0].templates_dict['star'][:, 0]
        res = np.full(shape=(n_spec, nxhr), fill_value=np.nan)
        
        # Residuals
        for i in range(0, n_spec, nsample):
            
            if frame == 'star':
                
                if iter_indexes[o] == 0 and not forward_models[o, i].models_dict['star'].from_synthetic:
                    vel = forward_models[o, i].data.bc_vel
                else:
                    vel = -1 * forward_models[o, i].best_fit_pars[iter_indexes[o]][forward_models[o, i].models_dict['star'].par_names[0]].value
                    
                # Shift the residuals
                wave_shifted = forward_models[o, i].wavelength_solutions[iter_indexes[o]] * np.exp(vel / cs.c)
            
                # Interpolate for sanity / consistency
                good = np.where(np.isfinite(forward_models[o, i].residuals[iter_indexes[o]]) & np.isfinite(wave_shifted))[0]
                residuals_shifted = scipy.interpolate.CubicSpline(wave_shifted[good], forward_models[o, i].residuals[iter_indexes[o]][good], extrapolate=False)(star_wave)
                res[i, :] = residuals_shifted
                
                # Plot
                axarr[o].plot(wave_shifted * factor, forward_models[o, i].residuals[iter_indexes[o]], alpha=0.7)
                
            else:
                
                good = np.where(np.isfinite(forward_models[o, i].residuals[iter_indexes[o]]) & np.isfinite(forward_models[o, i].wavelength_solutions[iter_indexes[o]]))[0]
                residuals_interp = scipy.interpolate.CubicSpline(forward_models[o, i].wavelength_solutions[iter_indexes[o]][good], forward_models[o, i].residuals[iter_indexes[o]][good], extrapolate=False)(star_wave)
                
                res[i, :] = residuals_interp

                # Plot
                axarr[o].plot(forward_models[o, i].wavelength_solutions[iter_indexes[o]] * factor, forward_models[o, i].residuals[iter_indexes[o]], alpha=0.7)
        
        for t in templates:
            if type(forward_models[o, i].templates_dict[t]) is dict:
                for tt in forward_models[o, 0].templates_dict[t]:
                    w, f = forward_models[o, 0].templates_dict[t][tt][:, 0], forward_models[o, i].templates_dict[t][tt][:, 1]
                    ww = np.linspace(np.nanmin(w), np.nanmax(w), num=w.size)
                    ff = np.interp(ww, w, f, left=np.nan, right=np.nan)
                    fc = forward_models[o, 0].models_dict['lsf'].convolve_flux(ff, pars=forward_models[o, i].initial_parameters)
                    axarr[o].plot(ww, fc - np.nanmin(fc) + 0.2, alpha=0.8, label=tt)
            else:
                w, f = forward_models[o, 0].templates_dict[t][:, 0], forward_models[o, 0].templates_dict[t][:, 1]
                ww = np.linspace(np.nanmin(w), np.nanmax(w), num=w.size)
                ff = np.interp(ww, w, f, left=np.nan, right=np.nan)
                fc = forward_models[o, 0].models_dict['lsf'].convolve_flux(ff, pars=forward_models[o, 0].initial_parameters)
                axarr[o].plot(ww, fc - np.nanmin(fc) + 0.2, alpha=0.8, label=t)
                
            if frame == 'star' and t == 'star':
                w, f = forward_models[o, 0].templates_dict[t][:, 0], forward_models[o, 0].templates_dict[t][:, 1]
                fc = forward_models[o, 0].models_dict['lsf'].convolve_flux(f, pars=forward_models[o, 0].initial_parameters)
                axarr[o].plot(ww, fc - np.nanmin(fc) + 0.2, alpha=0.8, label=t)
                
                
        axarr[o].plot(star_wave * factor, np.nanmedian(res, axis=0), c='black')
            
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indexes[o] + 1), fontsize=8)
        axarr[o].tick_params(axis='both', labelsize=10)
        axarr[o].legend()
    axarr[-1].set_xlabel('Wavelength [' + unit + ']')
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.95, wspace=None, hspace=0.5)
    plt.tight_layout()
    fig.text(0.03, 0.5, 'Norm. flux', rotation=90, verticalalignment='center', horizontalalignment='center', fontsize=10)
    fig.text(0.5, 0.97, star_name, fontsize=10, verticalalignment='center', horizontalalignment='center')
    plt.savefig(output_path_root + 'residuals.png')
    
    if debug:
        plt.show()
        stop()
    
    plt.close()
    
    
    
def inspect_lsf(output_path_root, do_orders, bad_rvs_dict, iter_indexes, debug=False, star_name=None, forward_models=None):
    
    # Load forward models
    if forward_models is None:
        forward_models = parser.parse_forward_models(output_path_root, do_orders=do_orders)
    
    if star_name is None:
        star_name = forward_models[0, 0].star_name
    
    n_orders, n_spec = forward_models.shape
    plot_height_single = 4
    fig, axarr = plt.subplots(nrows=n_orders, ncols=1, figsize=(10, int(plot_height_single*n_orders)), dpi=250)
    axarr = np.atleast_1d(axarr)
    
    for o in range(n_orders):
        
        xdefault = forward_models[o, 0].models_dict['lsf'].x
        nxlsf = xdefault.size
        lsfs = np.zeros((nxlsf, n_spec))
        
        for i in range(n_spec):
            
            # Build LSF
            x = forward_models[o, i].models_dict['lsf'].x
            lsf = forward_models[o, i].models_dict['lsf'].build(pars=forward_models[o, i].best_fit_pars[iter_indexes[o]])
            
            # Plot
            axarr[o].plot(x, lsf, alpha=0.7)
            
            # Interpolate
            lsfs[:, i] = scipy.interpolate.CubicSpline(x, lsf, extrapolate=False)(xdefault)
            
        axarr[o].plot(xdefault, np.nanmedian(lsfs, axis=1), c='black')
        good = np.where(lsfs[:, 0] / np.nanmax(lsfs[:, 0]) > 1E-6)[0]
        f, l = x[good[0]], x[good[-1]]
            
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indexes[o] + 1), fontsize=8)
        axarr[o].tick_params(axis='both', labelsize=10)
        axarr[o].set_xlabel('Wavelength [Ang], Mean $\lambda=$' + str(round(np.nanmean(forward_models[o, 0].templates_dict['star'][:, 0]), 3)))
        axarr[o].set_xlim(f, l)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.95, wspace=None, hspace=0.5)
    plt.tight_layout()
    fig.text(0.03, 0.5, 'Normalized LSF profile', rotation=90, verticalalignment='center', horizontalalignment='center', fontsize=10)
    fig.text(0.5, 0.97, star_name, fontsize=10, verticalalignment='center', horizontalalignment='center')
    plt.savefig(output_path_root + 'lsf_profiles.png')
    
    if debug:
        plt.show()
    
    plt.close()
    
    return forward_models
    
    
def inspect_wls(output_path_root, do_orders, bad_rvs_dict, iter_indexes, star_name=None, debug=False, forward_models=None):
    
    # Load forward models
    if forward_models is None:
        forward_models = parser.parse_forward_models(output_path_root, do_orders=do_orders)
    
    if star_name is None:
        star_name = forward_models[0, 0].star_name
    
    n_orders, n_spec = forward_models.shape
    plot_height_single = 4
    fig, axarr = plt.subplots(nrows=n_orders, ncols=1, figsize=(10, int(plot_height_single*n_orders)), dpi=250)
    axarr = np.atleast_1d(axarr)
    
    for o in range(n_orders):
        
        npix = forward_models[o, 0].data.flux.size
        pix = np.arange(npix)
        wlss = np.zeros((npix, n_spec))
        
        for i in range(n_spec):
            
            # Build wls
            wls = forward_models[o, i].models_dict['wavelength_solution'].build(forward_models[o, i].best_fit_pars[iter_indexes[o]])
            wlss[:, i] = wls
            
        mwls = np.nanmedian(wlss, axis=1)
        
        for i in range(n_spec):
            
            # Plot
            axarr[o].plot(pix, mwls - wlss[:, i], alpha=0.7)
            
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indexes[o] + 1), fontsize=8)
        axarr[o].tick_params(axis='both', labelsize=10)
    axarr[-1].set_xlabel('Detector Pixels')
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.95, wspace=None, hspace=0.5)
    plt.tight_layout()
    fig.text(0.03, 0.5, 'Wavelength Solution', rotation=90, verticalalignment='center', horizontalalignment='center', fontsize=10)
    fig.text(0.5, 0.97, star_name, fontsize=10, verticalalignment='center', horizontalalignment='center')
    plt.savefig(output_path_root + 'wavelength_solutions.png')
    
    if debug:
        plt.show()
    
    plt.close()
    
    return forward_models
    
def inspect_blaze(output_path_root, do_orders, bad_rvs_dict, iter_indexes, star_name=None, debug=False, forward_models=None):
    
    # Load forward models
    if forward_models is None:
        forward_models = parser.parse_forward_models(output_path_root, do_orders=do_orders)
    
    if star_name is None:
        star_name = forward_models[0, 0].star_name
    
    n_orders, n_spec = forward_models.shape
    plot_height_single = 4
    fig, axarr = plt.subplots(nrows=n_orders, ncols=1, figsize=(10, int(plot_height_single*n_orders)), dpi=250)
    axarr = np.atleast_1d(axarr)
    
    for o in range(n_orders):
        
        x = forward_models[o, 0].models_dict['wavelength_solution'].build(forward_models[o, 0].best_fit_pars[iter_indexes[o]])
        nx = x.size
        
        blazes = np.zeros((nx, n_spec))
        
        for i in range(n_spec):
            
            # Build blaze
            blaze = forward_models[o, i].models_dict['blaze'].build(forward_models[o, i].best_fit_pars[iter_indexes[o]], x)
            
            # Plot
            axarr[o].plot(x, blaze, alpha=0.7)
            
            blazes[:, i] = blaze
            
        axarr[o].plot(x, np.nanmedian(blazes, axis=1), c='black')
            
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indexes[o] + 1), fontsize=8)
        axarr[o].tick_params(axis='both', labelsize=10)
    axarr[-1].set_xlabel('Wavelength [Ang]')
    plt.subplots_adjust(left=0.15, bottom=0.08, right=0.97, top=0.95, wspace=None, hspace=0.5)
    plt.tight_layout()
    fig.text(0.5, 0.97, star_name, fontsize=10, verticalalignment='center', horizontalalignment='center')
    plt.savefig(output_path_root + 'continuums.png')
    
    if debug:
        plt.show()
    
    plt.close()
    
    return forward_models


            
def rvs_quicklook(output_path_root, do_orders, bad_rvs_dict, iter_index, xcorr=False, flag=False, phase_to=None, debug=False, tc=None, thresh=5):
    
    if phase_to is None:
        _phase_to = 1E20
    else:
        _phase_to = phase_to
        
    if tc is None:
        alpha = 0
    else:
        alpha = tc - phase_to / 2
    
    # Parse RVs
    rvs_dict = parser.parse_rvs(output_path_root, do_orders)
    
    # Numbers
    n_orders, n_spec, n_iters = rvs_dict['rvs'].shape
    n_obs_nights = rvs_dict['n_obs_nights']
    bjds, bjdsn = rvs_dict['BJDS'], rvs_dict['BJDS_nightly']
    n_nights = len(n_obs_nights)
    
    iter_indexes = np.zeros(n_spec)
    print_rv_summary(rvs_dict, {}, do_orders, iter_indexes, xcorr=xcorr)
    
    # Generate mask
    rvs_dict, mask = gen_rv_mask(rvs_dict, bad_rvs_dict)
    
    # Get RVs
    rvs = np.zeros((n_orders, n_spec))
    rvsn = np.zeros((n_orders, n_nights))
    uncn = np.zeros((n_orders, n_nights))
    weights = np.zeros((n_orders, n_spec))
    for o in range(n_orders):
        if xcorr:
            rvs[o, :] = rvs_dict['rvsx'][o, :, iter_index] - np.nanmedian(rvs_dict['rvsx'][o, :, iter_index])
            rvsn[o, :] = rvs_dict['rvsx_nightly'][o, :, iter_index] - np.nanmedian(rvs_dict['rvsx_nightly'][o, :, iter_index])
            uncn[o, :] = rvs_dict['uncx_nightly'][o, :, iter_index]
        else:
            rvs[o, :] = rvs_dict['rvs'][o, :, iter_index] - np.nanmedian(rvs_dict['rvs'][o, :, iter_index])
            rvsn[o, :] = rvs_dict['rvs_nightly'][o, :, iter_index] - np.nanmedian(rvs_dict['rvs_nightly'][o, :, iter_index])
            uncn[o, :] = rvs_dict['unc_nightly'][o, :, iter_index]
        
        f, l = 0, n_obs_nights[0]
        for inight in range(n_nights):
            weights[o, f:l] = mask[o, f:l] * 1 / uncn[o, inight]**2
            if inight < n_nights - 1:
                f += n_obs_nights[inight]
                l += n_obs_nights[inight+1]
        
    # Combine
    rvs_nightly, unc_nightly = pcrvcalc.compute_nightly_rvs_from_all(rvs, weights, n_obs_nights, flag_outliers=flag, thresh=thresh)
    
    # Plot
    for o in range(n_orders):
        plt.plot((bjds - alpha)%_phase_to, rvs[o, :] - np.nanmedian(rvs[o, :]), marker='o', markersize=6, lw=0, label='Order ' + str(do_orders[o]), alpha=0.6)
        plt.errorbar((bjdsn - alpha)%_phase_to, rvsn[o, :] - np.nanmedian(rvsn[o, :]), yerr=uncn[o, :], marker='o', markersize=6, lw=0, label='Order ' + str(do_orders[o]), alpha=0.8)
        
        
    plt.errorbar((bjdsn - alpha)%_phase_to, rvs_nightly-np.nanmedian(rvs_nightly), yerr=unc_nightly, marker='o', lw=0, elinewidth=1, label='Binned Nightly', c='black', markersize=10)
    plt.legend()
    plt.show()
    
    if debug:
        stop()
        
    