import pychell.rvs.post_parser as pcparser
import pychell.rvs.rvcalc as pcrvcalc
import numpy as np
import matplotlib.pyplot as plt
import pychell
import pychell.rvs.forward_models as pcfoward_models
import pychell.maths as pcmath
import os
import scipy.constants as cs
import copy
from numba import jit, njit
import scipy.signal
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")
import datetime
from pdb import set_trace as stop


# Super Duper RV precision analysis
def rv_precision(parsers, targets, templates=None):
    
    # Number of targets
    n_targets = len(parsers)
    
    # Which templates to consider to determine the effective photon limit
    if templates is None:
        templates = ['star']
    
    # SNR for each target, for all orders, observations, and spectra
    snrs = [1 / p.parse_rms() for p in parsers]
    
    # Parse RVs
    for p in parsers:
        p.parse_rvs()
    
    # Mean wavelengths of each order.
    mean_waves = []
    for ip, p in enumerate(parsers):
        mean_waves.append([np.nanmean(p.forward_models[o][0].models_dict['wavelength_solution'].build(p.forward_models[o][0].opt_results[-1][0])) / 10 for o in range(p.n_orders)])
    
    # Compute approx nightly snrs for all targets, orders, obs, all orders
    print('Computing nightly S/N')
    nightly_snrs = []
    for ip, p in enumerate(parsers):
        nightly_snrs.append(np.zeros(p.n_nights))
        for o in range(p.n_orders):
            f, l = 0, p.n_obs_nights[0]
            for inight in range(p.n_nights):
                nightly_snrs[ip][inight] = np.sum(snrs[ip][o, f:l, -1]**2)**0.5
                if inight < p.n_nights - 1:
                    f += p.n_obs_nights[inight]
                    l += p.n_obs_nights[inight+1]
                    
                    
    print('Computing Nightly RVs')
    for p in parsers:
        combine_rvs(p)
        
                
    # Compute RV content of each order if set
    print('Computing Effective Noise limit from S/N and Template(s)')
    rvcontents = []
    rvprecs = []
    rvprecs_onesigma = []
    for ip, p in enumerate(parsers):
        rvprecs.append(np.zeros(p.n_orders))
        rvprecs_onesigma.append(np.zeros(p.n_orders))
        for o in range(p.n_orders):
            rvprecs[ip][o] = np.nanmean(p.rvs_dict['unc_nightly'][o, :, -1])
            rvprecs_onesigma[ip][o] = np.std(p.rvs_dict['unc_nightly'][o, :, -1])
        rvcontents.append(compute_rv_contents(p)[:, -1])

    # Fit order co-added precision vs per-pixel s/n
    #snrmodels = []
    #for o in range(p[0].n_orders):
        #_rvcs, _snrs = rvprecs_perwavelength[ip][o], nightly_snrs_perwavelength[ip][o]
        #init_pars = np.array([np.nanmean(_rvcs * (_snrs - np.nanmin(_rvcs))), np.nanmin(_rvcs)])
        #solver = NelderMead(pcmath.rms_loss_creator(rvprecsnrmodel), init_pars, args_to_pass=(_snrs, _rvcs))
        #opt_result = solver.solve()
        #best_pars = opt_result[0]
        #snrmodels_perwavelength[ip][o].append(best_pars)
        #snr_hrgrid = np.arange(_snrs.min()-10, _snrs.max() + 10, 0.1)
        #_snrmodel = rvprecsnrmodel(snr_hrgrid, *best_pars)
        #snrmodels_perwavelength[ip].append(_snrmodel)
        
    # RV prec and noise limit vs snr for each target
    plt.figure(1, figsize=(12, 8), dpi=200)
    for ip, p in enumerate(parsers):
        
        # Plot rv unc
        plt.errorbar(mean_waves[0], rvprecs[ip], yerr=rvprecs_onesigma[ip], marker='o', lw=0, elinewidth=1)
        
        # Annotate
        #plt.annotate(targets[ip]['stype'], (mean_waves[0], rvcontents[ip][1]))
        
        # Plot noise limit
        plt.plot(mean_waves[0], rvcontents[ip], marker='X', lw=1)
        
    plt.title(parsers[0].spectrograph + ' PRV Performance')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('$\sigma_{RV}$')
    #plt.tight_layout()
    fname = 'rv_precisions_' + parsers[0].spectrograph.lower() + '.png'
    plt.savefig(fname)
    plt.close()
    
def combine_stellar_templates(output_path_root, do_orders=None, iter_index=None):
    
    if do_orders is None:
        do_orders = parser.get_orders(output_path_root)
        
    n_orders = len(do_orders)
    
    stellar_templates = parser.parse_stellar_templates(output_path_root, do_orders=do_orders, iter_indices=[iter_index]*n_orders)
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
        

def combine_rvs(parser, iter_indices=None):
    """Combines RVs across orders.

    Args:
        parser: A parser.
    Returns:
        tuple: The results returned by the call to method.
    """
    
    # Parse RVs
    rvs_dict = parser.parse_rvs()
    
    # Mask rvs from user input
    rv_mask = gen_rv_mask(parser)
    
    # Regenerate nightly rvs
    rms = parser.parse_rms()
    for o in range(parser.n_orders):
        for j in range(parser.n_iters_rvs):
            
            # NM RVs
            rvs = parser.rvs_dict['rvs'][o, :, j]
            weights = 1 / rms[o, :, j + parser.index_offset]**2 * rv_mask[o, :, j]
            rvs_dict['rvs_nightly'][o, :, j], rvs_dict['unc_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvs, weights, parser.n_obs_nights, flag_outliers=True)
            
            # Xcorr RVs
            if parser.do_xcorr:
                rvs = parser.rvs_dict['rvsx'][o, :, j]
                weights = 1 / rms[o, :, j + parser.index_offset]**2 * rv_mask[o, :, j]
                rvs_dict['rvsx_nightly'][o, :, j], rvs_dict['uncx_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvs, weights, parser.n_obs_nights, flag_outliers=True)
                
    # Determine indices
    if iter_indices == 'best':
        _, iter_indices = parser.get_best_iters()
    elif iter_indices is None:
        iter_indices = np.zeros(parser.n_orders).astype(int) + parser.n_iters_rvs - 1
    elif type(iter_indices) is int:
        iter_indices = np.zeros(parser.n_orders).astype(int) + iter_indices
        
    # Summary of rvs
    print_rv_summary(parser, iter_indices)

    # Generate weights
    rvcs = compute_rv_contents(parser)
    weights = gen_rv_weights(parser)
    
    # Combine RVs for NM
    rvs_unpacked = np.array([rvs_dict['rvs'][o, :, iter_indices[o]] for o in range(parser.n_orders)])
    weights_unpacked = np.array([weights[o, :, iter_indices[o]] for o in range(parser.n_orders)])
    result_nm = pcrvcalc.combine_relative_rvs(rvs_unpacked, weights_unpacked, parser.n_obs_nights)
    
    # Combine RVs for XC
    rvs_unpacked = np.array([rvs_dict['rvsx'][o, :, iter_indices[o]] for o in range(parser.n_orders)])
    weights_unpacked = np.array([weights[o, :, iter_indices[o]] for o in range(parser.n_orders)])
    result_xc = pcrvcalc.combine_relative_rvs(rvs_unpacked, weights_unpacked, parser.n_obs_nights)
    
    # Add to dictionary
    parser.rvs_dict['rvs_out'] = result_nm[0]
    parser.rvs_dict['unc_out'] = result_nm[1]
    parser.rvs_dict['rvs_nightly_out'] = result_nm[2]
    parser.rvs_dict['unc_nightly_out'] = result_nm[3]
    parser.rvs_dict['rvsx_out'] = result_xc[0]
    parser.rvs_dict['uncx_out'] = result_xc[1]
    parser.rvs_dict['rvsx_nightly_out'] = result_xc[2]
    parser.rvs_dict['uncx_nightly_out'] = result_xc[3]
    
# Detrend RVs if applicable
def detrend_rvs(parser, var='BIS', thresh=0.5):
    rvs_dict = parser.rvs_dict
    rvs_detrended = np.zeros((parser.n_orders, parser.n_spec))
    rvsx_detrended = np.zeros((parser.n_orders, parser.n_spec))
    for o in range(parser.n_orders):
        for j in range(parser.n_iters_rvs):
            good = np.where(np.isfinite(rvs_dict['rvs'][o, :, j]) & np.isfinite(rvs_dict['BIS'][o, :, j]))[0]
            if good.size == 0:
                continue
            rvs_dict['rvs'][o, good, j] = pcrvcalc.detrend_rvs(parser.do_orders[o], rvs_dict['rvs'][o, good, j], rvs_dict['BIS'][o, good, j], thresh=thresh)
            
            if rvs_dict['do_xcorr']:
                good = np.where(np.isfinite(rvs_dict['rvsx'][o, :, j]) & np.isfinite(rvs_dict['BIS'][o, :, j]))[0]
                if good.size == 0:
                    continue
                rvs_dict['rvsx'][o, good, j] = pcrvcalc.detrend_rvs(parser.do_orders[o], rvs_dict['rvsx'][o, good, j], rvs_dict['BIS'][o, good, j], thresh=thresh)
                
    # Add to dictionary
    rvs_dict['rvsx_detrended'] = rvs_detrended
    rvs_dict['rvsx_detrended'] = rvsx_detrended
    
    
def gen_rv_mask(parser):
    
    # Return if no dictionary exists
    if not hasattr(parser, 'bad_rvs_dict'):
        return parser.rvs_dict, mask
    
    rvs_dict = parser.rvs_dict
    bad_rvs_dict = parser.rvs_dict
    
    # Initialize a mask
    mask = np.ones(shape=(parser.n_orders, parser.n_spec, parser.n_iters_rvs), dtype=float)
    
    # Mask all spectra for a given night
    if 'bad_nights' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_nights']:
            inds = parser.forward_models[0][0].get_all_spec_indices_from_night(i, parser.n_obs_nights)
            mask[:, inds, :] = 0
            rvs_dict['rvs'][:, inds, :] = np.nan
            if rvs_dict['do_xcorr']:
                rvs_dict['rvsx'][:, inds, :] = np.nan
                rvs_dict['BIS'][:, inds, :] = np.nan
    
    # Mask individual spectra
    if 'bad_spec' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_spec']:
            mask[:, i, :] = 0
            rvs_dict['rvs'][:, i, :] = np.nan
            if rvs_dict['do_xcorr']:
                rvs_dict['rvsx'][:, i, :] = np.nan
                rvs_dict['BIS'][:, i, :] = np.nan
        
    return mask
    
    
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

def print_rv_summary(parser, iter_indices):
    
    rvs_dict = parser.rvs_dict
    
    for o in range(parser.n_orders):
        print('Order ' + str(parser.do_orders[o]))
        for k in range(parser.n_iters_rvs):
            stddev = np.nanstd(rvs_dict['rvs_nightly'][o, :, k])
            if rvs_dict['do_xcorr']:
                stddevxc = np.nanstd(rvs_dict['rvsx_nightly'][o, :, k])
                
            if k == iter_indices[o]:
                print(' ** Iteration ' +  str(k + 1) + ': ' + str(round(stddev, 4)) + ' m/s')
            else:
                print('    Iteration ' +  str(k + 1) + ': ' + str(round(stddev, 4)) + ' m/s')
                
            if k == iter_indices[o]:
                print(' ** Iteration ' +  str(k + 1) + ': ' + str(round(stddevxc, 4)) + ' m/s')
            else:
                print('    Iteration ' +  str(k + 1) + ': ' + str(round(stddevxc, 4)) + ' m/s')
            
def gen_rv_weights(parser):
    
    # Generate mask
    mask = gen_rv_mask(parser)
    
    # RMS weights
    rms = parser.parse_rms()
    weights_rms = 1 / rms**2
        
    # RV content weights
    rvconts = compute_rv_contents(parser)
    weights_rvcont = 1 / rvconts**2
    
    # Combine weights, multiplicatively
    weights_rv_cont_expanded = np.zeros((parser.n_orders, parser.n_spec, parser.n_iters_rvs))
    for o in range(parser.n_orders):
        for j in range(parser.n_iters_rvs):
            weights_rv_cont_expanded[o, :, j] = weights_rvcont[o, j]

    weights = weights_rv_cont_expanded * weights_rms * mask
    
    # Normalize
    weights /= np.nansum(weights)

    return weights

def compute_rv_contents(parser, templates=None):
    
    if templates is None:
        templates = []
        for t in parser.forward_models[0].templates_dict.keys():
            if t == 'star':
                templates.append('star')
            elif t == 'gas_cell':
                templates.append('gas_cell')
            
    rvcs = np.zeros((parser.n_orders, parser.n_iters_rvs))
    
    nightly_snrs = compute_nightly_snrs(parser)
    
    for o in range(parser.n_orders):
        pars = parser.forward_models[o][0].opt_results[-1][0]
        wave_data = parser.forward_models[o][0].models_dict['wavelength_solution'].build(pars)
        lsf = parser.forward_models[o][0].models_dict['lsf'].build(pars)
        _rvcs = np.zeros(len(templates))
        for itemplate, t in enumerate(templates):
            for j in range(parser.n_iters_rvs):
                template_wave, template_flux = parser.forward_models[o].templates_dict[t][:, 0], parser.forward_models[o].templates_dict[t][:, 1]
                _, _rvcs[itemplate] = pcrvcalc.compute_rv_content(template_wave, template_flux, snr=np.nanmedian(nightly_snrs[o, :, j]), blaze=True, ron=0, wave_to_sample=wave_data, lsf=lsf)
                rvcs[o, j] = np.nansum(_rvcs**2)**0.5
        
    return rvcs
        

def compute_nightly_snrs(parser):

    # Parse the rms
    rms = parser.parse_rms()
    nightly_snrs = np.zeros((parser.n_orders, parser.n_nights, parser.n_iters_rvs))

    for o in range(parser.n_orders):
        f, l = 0, parser.n_obs_nights[0]
        for i in range(parser.n_nights):
            for j in range(parser.n_iters_rvs):
                nightly_snrs[o, i] = np.nansum((1 / rms[o, f:l, j + parser.index_offset])**2)**0.5
        if i < parser.n_nights - 1:
            f += parser.n_obs_nights[i]
            l += parser.n_obs_nights[i+1]
                
    return nightly_snrs


def parameter_corrs(parser, iter_indices=None, debug=False, rvvec=False):
    
    # Parse the RVs
    rvs_dict = parser.parse_rvs(output_path_root, do_orders=do_orders)
    
    mask = gen_rv_mask(parser)
    
    # Number of spectra and nights
    n_spec = np.sum(rvs_dict['n_obs_nights'])
    n_nights = len(rvs_dict['n_obs_nights'])
    n_iters_rvs = rvs_dict['rvs'].shape[2]
    n_iters_pars = n_iters_rvs + index_offset
    
    # Determine which iteration to use
    if iter_index is None:
        iter_indices = np.zeros(n_orders).astype(int) + n_iters_rvs - 1
    elif iter_index == 'best':
        _, iter_indices = get_best_iterations(rvs_dict, xcorr)
    else:
        iter_indices = np.zeros(n_orders).astype(int) + iter_index
    
    for o in range(n_orders):
        
        vp = forward_models[o, 0].opt_results[iter_indices[o] + index_offset][0].unpack(keys='vary')['vary']
        vpi = np.where(vp)[0]
        nv = vpi.size
        
        pars = np.empty(shape=(n_spec, n_iters_pars, nv), dtype=object)
        par_vals = np.full(shape=(n_spec, n_iters_pars, nv), dtype=float, fill_value=np.nan)
        par_names = list(forward_models[o, 0].opt_results[iter_indices[o] + index_offset][0].keys())
        par_names = [par_names[v] for v in vpi]
        for ispec in range(n_spec):
            for j in range(n_iters_pars):
                for k in range(nv):
                    pars[ispec, j, k] = forward_models[o, ispec].opt_results[j][0][par_names[k]]
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
                
                axarr[row, col].plot(rvs_dict['rvs'][o, :, iter_indices[o]], par_vals[:, iter_indices[o] + index_offset, k], marker='.', lw=0, c='black', markersize=8)
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
        
    
def plot_final_rvs(parser, phase_to=None, tc=None, kamp=None):
    
    if phase_to is None:
        _phase_to = 1E20
    else:
        _phase_to = phase_to
        if kamp is not None:
            modelx = np.linspace(0, _phase_to, num=300)
            modely = kamp * np.sin(2 * np.pi * modelx / _phase_to)
        
    if tc is None:
        alpha = 0
    else:
        alpha = tc - _phase_to / 2
        
    # Unpack rvs
    bjds, bjds_nightly = parser.rvs_dict['BJDS'], parser.rvs_dict['BJDS_nightly']
    if parser.xcorr:
        rvs_single, unc_single, rvs_nightly, unc_nightly = parser.rvs_dict['rvsx_out'], parser.rvs_dict['uncx_out'], parser.rvs_dict['rvsx_nightly_out'], parser.rvs_dict['uncx_nightly_out']
    else:
        rvs_single, unc_single, rvs_nightly, unc_nightly = parser.rvs_dict['rvs_out'], parser.rvs_dict['unc_out'], parser.rvs_dict['rvs_nightly_out'], parser.rvs_dict['unc_nightly_out']
    
    # Single rvs
    plt.errorbar((bjds - alpha)%_phase_to, rvs_single-np.nanmedian(rvs_single), yerr=unc_single, linewidth=0, elinewidth=1, marker='.', markersize=10, markerfacecolor='pink', color='green', alpha=0.8)

    # Nightly RVs
    plt.errorbar((bjds_nightly - alpha)%_phase_to, rvs_nightly-np.nanmedian(rvs_nightly), yerr=unc_nightly, linewidth=0, elinewidth=3, marker='o', markersize=10, markerfacecolor='blue', color='grey', alpha=0.9)
    
    plt.title(parser.star_name + ', ' + parser.spectrograph + ' Relative RVs')
    
    if kamp is not None:
        plt.plot(modelx, modely, label='K = ' + str(kamp) + ' m/s')
    
    if phase_to is None:
        plt.xlabel('BJD - BJD$_{0}$')
    else:
        plt.xlabel('Phase [days, P = ' +  str(round(_phase_to, 3)) + ']')
    plt.ylabel('RV [m/s]')
    
    plt.savefig(parser.output_path_root + 'rvs_final_' + parser.spectrograph.lower() + parser.star_name + '.png')
    plt.show()
            
            
            
def plot_stellar_templates_single_iter(output_path_root, star_name, stellar_templates, do_orders, iter_indices, unit='Ang'):
    
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
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indices[o] + 1), fontsize=8)
        axarr[o].tick_params(axis='both', labelsize=10)
    axarr[-1].set_xlabel('Wavelength [' + unit + ']')
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.95, wspace=None, hspace=0.5)
    fig.text(0.03, 0.5, 'Norm. flux', rotation=90, verticalalignment='center', horizontalalignment='center', fontsize=20)
    fig.text(0.5, 0.97, star_name, fontsize=20, verticalalignment='center', horizontalalignment='center')
    plt.savefig(output_path_root + 'stellar_templates.png')
    
    plt.close()
            

def residual_coherence(output_path_root, do_orders, bad_rvs_dict, iter_indices, frame='star', templates=[], unit='Ang', debug=False, star_name=None, nsample=1):
    
    n_orders = len(do_orders)
    
    forward_models = parser.parse_forward_models(output_path_root, do_orders=do_orders)
    
    stellar_templates = parser.parse_stellar_templates(output_path_root, do_orders=do_orders, iter_indices=iter_indices)
    
    templates_dicts = parser.parse_templates(output_path_root, do_orders=do_orders)
    
    if star_name is None:
        star_name = forward_models[0, 0].star_name
    
    if unit == 'microns':
        factor = 1E-4
    elif unit == 'Ang':
        factor = 1
    else:
        factor = 1E-1
    
    n_orders, n_spec = forward_models.shape
    nxhr = templates_dicts[o]['star'][:, 0].size
    plot_height_single = 4
    fig, axarr = plt.subplots(nrows=n_orders, ncols=1, figsize=(10, int(plot_height_single*n_orders)), dpi=250)
    axarr = np.atleast_1d(axarr)
    
    for o in range(n_orders):
        
        star_wave = templates_dicts[o]['star'][:, 0]
        res = np.full(shape=(n_spec, nxhr), fill_value=np.nan)
        
        # Residuals
        for i in range(0, n_spec, nsample):
            
            if frame == 'star':
                
                if iter_indices[o] == 0 and not forward_models[o, i].models_dict['star'].from_synthetic:
                    vel = forward_models[o, i].data.bc_vel
                else:
                    vel = -1 * forward_models[o, i].opt_results[iter_indices[o]][0][forward_models[o, i].models_dict['star'].par_names[0]].value
                    
                # Shift the residuals
                wave_shifted = forward_models[o, i].wavelength_solutions[iter_indices[o]] * np.exp(vel / cs.c)
            
                # Interpolate for sanity / consistency
                good = np.where(np.isfinite(forward_models[o, i].residuals[iter_indices[o]]) & np.isfinite(wave_shifted))[0]
                residuals_shifted = scipy.interpolate.CubicSpline(wave_shifted[good], forward_models[o, i].residuals[iter_indices[o]][good], extrapolate=False)(star_wave)
                res[i, :] = residuals_shifted
                
                # Plot
                axarr[o].plot(wave_shifted * factor, forward_models[o, i].residuals[iter_indices[o]], alpha=0.7)
                
            else:
                
                good = np.where(np.isfinite(forward_models[o, i].residuals[iter_indices[o]]) & np.isfinite(forward_models[o, i].wavelength_solutions[iter_indices[o]]))[0]
                residuals_interp = scipy.interpolate.CubicSpline(forward_models[o, i].wavelength_solutions[iter_indices[o]][good], forward_models[o, i].residuals[iter_indices[o]][good], extrapolate=False)(star_wave)
                
                res[i, :] = residuals_interp

                # Plot
                axarr[o].plot(forward_models[o, i].wavelength_solutions[iter_indices[o]] * factor, forward_models[o, i].residuals[iter_indices[o]], alpha=0.7)
        
        for t in templates:
            if type(templates_dicts[o][t]) is dict:
                for tt in templates_dicts[o][t]:
                    w, f = templates_dicts[o][tt][:, 0], templates_dicts[o][t][tt][:, 1]
                    ww = np.linspace(np.nanmin(w), np.nanmax(w), num=w.size)
                    ff = np.interp(ww, w, f, left=np.nan, right=np.nan)
                    fc = forward_models[o, 0].models_dict['lsf'].convolve_flux(ff, pars=forward_models[o, i].initial_parameters)
                    axarr[o].plot(ww, fc - np.nanmin(fc) + 0.2, alpha=0.8, label=tt)
            else:
                w, f = templates_dicts[o][t][:, 0], templates_dicts[o][t][:, 1]
                ww = np.linspace(np.nanmin(w), np.nanmax(w), num=w.size)
                ff = np.interp(ww, w, f, left=np.nan, right=np.nan)
                fc = forward_models[o, 0].models_dict['lsf'].convolve_flux(ff, pars=forward_models[o, 0].initial_parameters)
                axarr[o].plot(ww, fc - np.nanmin(fc) + 0.2, alpha=0.8, label=t)
                
            if frame == 'star' and t == 'star':
                w, f = templates_dicts[o][t][:, 0], templates_dicts[o][t][:, 1]
                fc = forward_models[o, 0].models_dict['lsf'].convolve_flux(f, pars=forward_models[o, 0].initial_parameters)
                axarr[o].plot(ww, fc - np.nanmin(fc) + 0.2, alpha=0.8, label=t)
                
                
        axarr[o].plot(star_wave * factor, np.nanmedian(res, axis=0), c='black')
            
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indices[o] + 1), fontsize=8)
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
    
    
    
def inspect_lsf(output_path_root, do_orders, bad_rvs_dict, iter_indices, debug=False, star_name=None, forward_models=None):
    
    # Load forward models
    if forward_models is None:
        forward_models = parser.parse_forward_models(output_path_root, do_orders=do_orders)
    
    if star_name is None:
        star_name = forward_models[0, 0].star_name
        
    templates_dicts = parser.parse_templates(output_path_root, do_orders=do_orders)
    
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
            lsf = forward_models[o, i].models_dict['lsf'].build(pars=forward_models[o, i].opt_results[iter_indices[o]][0])
            
            # Plot
            axarr[o].plot(x, lsf, alpha=0.7)
            
            # Interpolate
            lsfs[:, i] = scipy.interpolate.CubicSpline(x, lsf, extrapolate=False)(xdefault)
            
        axarr[o].plot(xdefault, np.nanmedian(lsfs, axis=1), c='black')
        good = np.where(lsfs[:, 0] / np.nanmax(lsfs[:, 0]) > 1E-6)[0]
        f, l = x[good[0]], x[good[-1]]
            
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indices[o] + 1), fontsize=8)
        axarr[o].tick_params(axis='both', labelsize=10)
        axarr[o].set_xlabel('Wavelength [Ang], Mean $\lambda=$' + str(round(np.nanmean(templates_dicts[o]['star'][:, 0]), 3)))
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
    
    
def inspect_wls(output_path_root, do_orders, bad_rvs_dict, iter_indices, star_name=None, debug=False, forward_models=None):
    
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
            wls = forward_models[o, i].models_dict['wavelength_solution'].build(forward_models[o, i].opt_results[iter_indices[o]][0])
            wlss[:, i] = wls
            
        mwls = np.nanmedian(wlss, axis=1)
        
        for i in range(n_spec):
            
            # Plot
            axarr[o].plot(pix, mwls - wlss[:, i], alpha=0.7)
            
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indices[o] + 1), fontsize=8)
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
    
def inspect_blaze(output_path_root, do_orders, bad_rvs_dict, iter_indices, star_name=None, debug=False, forward_models=None):
    
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
        
        x = forward_models[o, 0].models_dict['wavelength_solution'].build(forward_models[o, 0].opt_results[iter_indices[o]][0])
        nx = x.size
        
        blazes = np.zeros((nx, n_spec))
        
        for i in range(n_spec):
            
            # Build blaze
            blaze = forward_models[o, i].models_dict['blaze'].build(forward_models[o, i].opt_results[iter_indices[o]][0], x)
            
            # Plot
            axarr[o].plot(x, blaze, alpha=0.7)
            
            blazes[:, i] = blaze
            
        axarr[o].plot(x, np.nanmedian(blazes, axis=1), c='black')
            
        axarr[o].set_title('Order ' + str(do_orders[o]) + ' iter ' + str(iter_indices[o] + 1), fontsize=8)
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


            
def rvs_quicklook(parser, bad_rvs_dict, iter_index, xcorr=False, flag=False, phase_to=None, debug=False, tc=None, thresh=None):
    
    if phase_to is None:
        _phase_to = 1E20
    else:
        _phase_to = phase_to
        
    if tc is None:
        alpha = 0
    else:
        alpha = tc - phase_to / 2
    
    # Parse RVs
    rvs_dict = parser.parse_rvs()
    
    # Numbers
    n_orders, n_spec, n_iters = rvs_dict['rvs'].shape
    n_obs_nights = rvs_dict['n_obs_nights']
    bjds, bjdsn = rvs_dict['BJDS'], rvs_dict['BJDS_nightly']
    n_nights = len(n_obs_nights)
    
    # Print summary
    print_rv_summary(parser, bad_rvs_dict, do_orders, [iter_index]*n_orders, xcorr=xcorr)
    
    # Generate mask
    mask = gen_rv_mask(rvs_dict, bad_rvs_dict)
    
    # Get single order RVs, simple median offset
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
        
    