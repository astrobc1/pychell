import pychell.rvs.post_parser as pcparser
import pychell.rvs.rvcalc as pcrvcalc
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
import pychell
import pychell.rvs.forward_models as pcforwardmodels
import pychell.maths as pcmath
from robustneldermead.neldermead import NelderMead
import os
import scipy.constants as cs
import copy
from numba import jit, njit
import scipy.signal
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")
import datetime
import pychell.utils as pcutils

# Multi Target rv precision as a function of S/N per spectral pixel, cumulative over a night
def rv_precision_snr(parsers, iter_indices=None, thresh=np.inf):
    
    # Number of targets to consider
    n_targets = len(parsers)
            
    _iter_indices = []
    for p in parsers:
        _iter_indices.append(p.resolve_iter_indices(iter_indices))
    iter_indices = _iter_indices
    
    # Parse RVs and forward models
    for p in parsers:
        p.parse_rvs()
        p.parse_forward_models()
        
    # Compute nightly S/N for each target, per spectral pixel, averaged over orders
    # Compute single spectrum S/N for each target, per spectral pixel, averaged over orders
    snrs = []
    nightly_snrs = []
    for ip, p in enumerate(parsers):
        _rms = p.parse_rms()
        _snrs = 1 / _rms
        snrs.append(np.nanmedian(_snrs[:, :, iter_indices[ip][0] + p.index_offset], axis=0).tolist()) # shape = (n_spec,)
        _nightly_snrs = compute_nightly_snrs(p)
        nightly_snrs.append(np.nanmedian(_nightly_snrs[:, :, iter_indices[ip][0]], axis=0).tolist()) # shape = (n_nights,)
        
    # Compute co-added RVs
    for ip, p in enumerate(parsers):
        combine_rvs(p, iter_indices=iter_indices[ip])
        
    # Get rv precisions from output rvs
    rvprecs = []
    nightly_rvprecs = []
    for ip, p in enumerate(parsers):
        rvprecs.append((p.rvs_dict['unc_out']).tolist()) # shape = (n_spec,)
        nightly_rvprecs.append(p.rvs_dict['unc_nightly_out'].tolist()) # shape = (n_nights,)
        
    # Effectively flatten
    rvprecs_flat = []
    nightly_rvprecs_flat = []
    snrs_flat = []
    nightly_snrs_flat = []
    for ip, p in enumerate(parsers):
        rvprecs_flat += rvprecs[ip]
        nightly_rvprecs_flat += nightly_rvprecs[ip]
        snrs_flat += snrs[ip]
        nightly_snrs_flat += nightly_snrs[ip]
        
    # Convert to arrays and sort
    snrs_all_flat = np.array(nightly_snrs_flat + snrs_flat)
    rvprecs_all_flat = np.array(nightly_rvprecs_flat + rvprecs_flat)
    inds = np.argsort(snrs_all_flat)
    rvprecs_all_flat = rvprecs_all_flat[inds]
    snrs_all_flat = snrs_all_flat[inds]
    
    # Remove outliers
    bad = np.where(rvprecs_all_flat > thresh)[0]
    if bad.size > 0:
        snrs_all_flat[bad] = np.nan
        rvprecs_all_flat[bad] = np.nan
    
    # Model
    A_guess = np.nanmedian(rvprecs_all_flat * snrs_all_flat)
    init_pars = np.array([A_guess])
    solver = NelderMead(pcmath.rms_loss_creator(rvprecmodel), init_pars, args_to_pass=(snrs_all_flat, rvprecs_all_flat))
    opt_result = solver.solve()
    best_pars = opt_result['xmin']
    snr_grid_hr = np.linspace(np.nanmax([np.nanmin(snrs_all_flat) - 10, 1]), np.nanmax(snrs_all_flat) + 10, num=1000)
    best_model = rvprecmodel(snr_grid_hr, *best_pars)
    
    # Plot
    plt.figure(1, figsize=(14, 8), dpi=200)
    plt.semilogy(snrs_all_flat, rvprecs_all_flat, marker='.', lw=0, markersize=10, markeredgewidth=0)
    plt.semilogy(snr_grid_hr, best_model, c='black', lw=3, ls=':')
    plt.axhline(y=5, c='green', ls=':', lw=2)
    plt.axhline(y=50, c='green', ls=':', lw=2)
    plt.xlabel('$S/N$ per spectral pixel', fontsize=20)
    plt.ylabel('$\sigma_{RV}$', fontsize=24)
    plt.tick_params(which='both', labelsize=20)
    plt.title(parsers[0].spectrograph + ' PRV Precision', fontsize=24)
    plt.tight_layout()
    
    # Save
    fname = 'rv_precisions_multitarget_' + datetime.date.today().strftime("%d%m%Y") + '.png'
    plt.savefig(fname)
    plt.close()
    
def rvprecmodel(SNR, A):
    return A / SNR
        

# Single Target rv precision as a function of wavelength (order)
def rv_precision_wavelength(parser, iter_indices=None):
    
    # Parse RVs and forward models
    parser.parse_rvs()
    parser.parse_forward_models()
    
    # Resolveiter indices
    iter_indices = parser.resolve_iter_indices(iter_indices)
        
    # SNR for each target, for all orders, observations, and spectra
    snrs = 1 / parser.parse_fit_metric()
    
    # Mean wavelengths of each chunk.
    mean_waves = np.zeros(shape=(parser.n_orders, parser.n_chunks))
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            mean_waves[o, ichunk] = parser.forward_models[o][0].chunk_regions[ichunk].midwave()
    
    # Compute approx nightly snrs for all targets, orders, obs, all orders
    print('Computing nightly S/N')
    nights_snrs = compute_nightly_snrs(parser)
                    
    print('Computing Nightly RVs')
    combine_rvs(parser, iter_indices=iter_indices)
        
    # Compute RV content of each order if set
    print('Computing Effective Noise limit from S/N and Template(s)')
    rvcontents_allchunks = np.zeros(shape=(parser.n_orders, parser.n_chunks))
    rvprecs_allchunks = np.zeros(shape=(parser.n_orders, parser.n_chunks))
    rvprecs_onesigma_allchunks = np.zeros(shape=(parser.n_orders, parser.n_chunks))
    _rvcontents = compute_rv_contents(parser)
    for o in range(parser.n_orders):
        rvcontents_allchunks[o] = np.nanmedian(_rvcontents[o, iter_indices[o]])
        rvprecs_allchunks[o] = np.nanmedian(parser.rvs_dict['unc_nightly'][o, :, iter_indices[o]])
        rvprecs_onesigma_allchunks[o] = np.nanstd(parser.rvs_dict['unc_nightly'][o, :, iter_indices[o]])
        
    # RV prec and noise limit vs snr for each target
    plt.figure(1, figsize=(12, 8), dpi=200)

    # Plot rv unc in nm
    plt.errorbar(mean_waves / 10, rvprecs_allchunks, yerr=rvprecs_onesigma_allchunks, marker='o', elinewidth=2, lw=0, markersize=14, label='Reported Unc.')

    # Plot noise limit in nm
    plt.plot(mean_waves / 10, rvcontents_allchunks, marker='X', lw=2.5, c='black', mfc='deeppink', markersize=14, label='Empirical Noise Limit')

    # Plot attrs
    plt.tick_params(which='both', labelsize=20)
    plt.title(parser.star_name + ' / ' + parser.spectrograph + ' PRV Precision')
    plt.xlabel('Wavelength [nm]', fontsize=20)
    plt.ylabel('$\sigma_{RV}$', fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save
    fname = parser.output_path_root + 'rv_precisions_' + parser.spectrograph.lower().replace(' ', '_') + '_' + parser.star_name.lower().replace(' ', '_') + '.png'
    plt.savefig(fname)
    plt.close()
    
    return mean_waves, rvcontents, rvprecs, rvprecs_onesigma
    
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
    fit_metric = parser.parse_fit_metric()
    for o in range(parser.n_orders):
        for j in range(parser.n_iters_rvs):
            
            # NM RVs
            rvsfwm = parser.rvs_dict['rvsfwm'][o, :, :, j] # n_spec x n_chunks
            weights = 1 / fit_metric[o, :, :, j + parser.index_offset]**2 * rv_mask[o, :, :, j]
            rvs_dict['rvsfwm_nightly'][o, :, j], rvs_dict['uncfwm_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsfwm, weights, parser.n_obs_nights, flag_outliers=True)
            
            # Xcorr RVs
            if parser.rvs_dict['do_xcorr']:
                rvsxc = parser.rvs_dict['rvsxc'][o, :, :, j]
                weights = 1 / fit_metric[o, :, :, j + parser.index_offset]**2 * rv_mask[o, :, :, j]
                rvs_dict['rvsxc_nightly'][o, :, j], rvs_dict['uncxc_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsxc, weights, parser.n_obs_nights, flag_outliers=True)
                
                # Detrended RVs
                if 'rvsxcdet' in parser.rvs_dict:
                    rvsxcdet = parser.rvs_dict['rvsxcdet'][o, :, j]
                    weights = 1 / fit_metric[o, :, :, j + parser.index_offset]**2 * rv_mask[o, :, :, j]
                    rvs_dict['rvsxcdet_nightly'][o, :, j], rvs_dict['uncxdet_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsxcdet, weights, parser.n_obs_nights, flag_outliers=True)
                    

    # Determine indices
    iter_indices = parser.resolve_iter_indices(iter_indices)
        
    # Summary of rvs
    print_rv_summary(parser, iter_indices)

    # Generate weights
    weights = gen_rv_weights(parser)
    
    # Combine RVs for NM
    rvsfwm_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    weights_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            rvsfwm_single_iter[o, :, :] = rvs_dict["rvsfwm"][o, :, :, iter_indices[o, ichunk]]
            weights_single_iter[o, :, :] = weights[o, :, :, iter_indices[o, ichunk]]
    result_nm = pcrvcalc.combine_relative_rvs(rvsfwm_single_iter, weights_single_iter, parser.n_obs_nights)

    # Combine RVs for XC
    rvsxc_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    weights_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            rvsxc_single_iter[o, :, :] = rvs_dict["rvsxc"][o, :, :, iter_indices[o, ichunk]]
            weights_single_iter[o, :, :] = weights[o, :, :, iter_indices[o, ichunk]]
    result_xc = pcrvcalc.combine_relative_rvs(rvsxc_single_iter, weights_single_iter, parser.n_obs_nights)
    
    # Combine RVs for Detrended
    if 'rvsdet' in parser.rvs_dict:
        rvsxcdet_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
        weights_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
        for o in range(parser.n_orders):
            for ichunk in range(parser.n_chunks):
                rvsxcdet_single_iter[o, :, :] = rvs_dict["rvsxcdet"][o, :, :, iter_indices[o, ichunk]]
                weights_single_iter[o, :, :] = weights[o, :, :, iter_indices[o, ichunk]]
        result_det = pcrvcalc.combine_relative_rvs(rvsxcdet_single_iter, weights_single_iter, parser.n_obs_nights)
    
    # Add to dictionary
    parser.rvs_dict['rvsfwm_out'] = result_nm['rvs']
    parser.rvs_dict['uncfwm_out'] = result_nm['unc']
    parser.rvs_dict['rvsfwm_nightly_out'] = result_nm['rvs_nightly']
    parser.rvs_dict['uncfwm_nightly_out'] = result_nm['unc_nightly']
    
    
    if 'rvsxc' in parser.rvs_dict:
        parser.rvs_dict['rvsxc_out'] = result_xc['rvs']
        parser.rvs_dict['uncxc_out'] = result_xc['unc']
        parser.rvs_dict['rvsxc_nightly_out'] = result_xc['rvs_nightly']
        parser.rvs_dict['uncxc_nightly_out'] = result_xc['unc_nightly']
    
    if 'rvsxcdet' in parser.rvs_dict:
        parser.rvs_dict['rvsxcdet_out'] = result_det['rvs']
        parser.rvs_dict['uncxcdet_out'] = result_det['unc']
        parser.rvs_dict['rvsxcdet_nightly_out'] = result_det['rvs_nightly']
        parser.rvs_dict['uncxcdet_nightly_out'] = result_det['unc_nightly']
    
    # Write to files for radvel
    fname_nightly = parser.output_path_root + 'rvs_nightly_final_' + parser.spectrograph.lower().replace(' ', '_') + '_' + parser.star_name.lower().replace(' ', '_') + '_' + datetime.date.today().strftime("%d%m%Y") + '.txt'
    fname_single = parser.output_path_root + 'rvs_final_' + parser.spectrograph.lower().replace(' ', '_') + '_' + parser.star_name.lower().replace(' ', '_') + '_' + datetime.date.today().strftime("%d%m%Y") + '.txt'
    telvec_nightly = np.array([parser.spectrograph.replace(' ', '_')] * parser.n_nights, dtype='<U20')
    telvec_single = np.array([parser.spectrograph.replace(' ', '_')] * parser.n_spec, dtype='<U20')
    if parser.rvs_out == 'xc':
        good = np.where(np.isfinite(parser.rvs_dict['rvsxc_nightly_out']))[0]
        tn, rvsn, uncn, telvec_nightly = parser.rvs_dict['bjds_nightly'][good], parser.rvs_dict['rvsxc_nightly_out'][good], parser.rvs_dict['uncxc_nightly_out'][good], telvec_nightly[good]
        good = np.where(np.isfinite(parser.rvs_dict['rvsxc_out']))[0]
        ts, rvss, uncs, telvec_single = parser.rvs_dict['bjds'][good], parser.rvs_dict['rvsxc_out'][good], parser.rvs_dict['uncxc_out'][good], telvec_single[good]
    elif parser.rvs_out == 'xcdet':
        good = np.where(np.isfinite(parser.rvs_dict['rvsxcdet_nightly_out']))[0]
        tn, rvsn, uncn, telvec_nightly = parser.rvs_dict['bjds_nightly'][good], parser.rvs_dict['rvsxcdet_nightly_out'][good], parser.rvs_dict['uncxcdet_nightly_out'][good], telvec_nightly[good]
        good = np.where(np.isfinite(parser.rvs_dict['rvsxc_out']))[0]
        ts, rvss, uncs, telvec_single = parser.rvs_dict['bjds'][good], parser.rvs_dict['rvsxcdet_out'][good], parser.rvs_dict['uncxcdet_out'][good], telvec_single[good]
    else:
        good = np.where(np.isfinite(parser.rvs_dict['rvsfwm_nightly_out']))[0]
        tn, rvsn, uncn, telvec_nightly = parser.rvs_dict['bjds_nightly'][good], parser.rvs_dict['rvsfwm_nightly_out'][good], parser.rvs_dict['uncfwm_nightly_out'][good], telvec_nightly[good]
        good = np.where(np.isfinite(parser.rvs_dict['rvsfwm_out']))[0]
        ts, rvss, uncs, telvec_single = parser.rvs_dict['bjds'][good], parser.rvs_dict['rvsfwm_out'][good], parser.rvs_dict['uncfwm_out'][good], telvec_single[good]
        
    with open(fname_nightly, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([tn, rvsn, uncn, telvec_nightly], dtype=object).T, fmt="%f,%f,%f,%s")
    with open(fname_single, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([ts, rvss, uncs, telvec_single], dtype=object).T, fmt="%f,%f,%f,%s")
    
    
def combine_rvs2(parser, iter_indices=None):
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
    fit_metric = parser.parse_fit_metric()
    for o in range(parser.n_orders):
        for j in range(parser.n_iters_rvs):
            
            # NM RVs
            rvsfwm = parser.rvs_dict['rvsfwm'][o, :, :, j] # n_spec x n_chunks
            weights = 1 / fit_metric[o, :, :, j + parser.index_offset]**2 * rv_mask[o, :, :, j]
            rvs_dict['rvsfwm_nightly'][o, :, j], rvs_dict['uncfwm_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsfwm, weights, parser.n_obs_nights, flag_outliers=True)
            
            # Xcorr RVs
            if parser.rvs_dict['do_xcorr']:
                rvsxc = parser.rvs_dict['rvsxc'][o, :, :, j]
                weights = 1 / fit_metric[o, :, :, j + parser.index_offset]**2 * rv_mask[o, :, :, j]
                rvs_dict['rvsxc_nightly'][o, :, j], rvs_dict['uncxc_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsxc, weights, parser.n_obs_nights, flag_outliers=True)
                
                # Detrended RVs
                if 'rvsxcdet' in parser.rvs_dict:
                    rvsxcdet = parser.rvs_dict['rvsxcdet'][o, :, j]
                    weights = 1 / fit_metric[o, :, :, j + parser.index_offset]**2 * rv_mask[o, :, :, j]
                    rvs_dict['rvsxcdet_nightly'][o, :, j], rvs_dict['uncxdet_nightly'][o, :, j] = pcrvcalc.compute_nightly_rvs_single_order(rvsxcdet, weights, parser.n_obs_nights, flag_outliers=True)
                    

    # Determine indices
    iter_indices = parser.resolve_iter_indices(iter_indices)
        
    # Summary of rvs
    print_rv_summary(parser, iter_indices)

    # Generate weights
    weights = gen_rv_weights(parser)
    
    # Combine RVs for NM
    rvsfwm_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    weights_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            rvsfwm_single_iter[o, :, :] = rvs_dict["rvsfwm"][o, :, :, iter_indices[o, ichunk]]
            weights_single_iter[o, :, :] = weights[o, :, :, iter_indices[o, ichunk]]
    result_nm = pcrvcalc.combine_rvs_tfa(rvsfwm_single_iter, weights_single_iter, parser.n_obs_nights)
    #plt.errorbar(parser.rvs_dict["bjds"], result_nm["rvs"] - np.nanmedian(result_nm["rvs"]), yerr=result_nm["unc"], lw=0, elinewidth=1, marker='o'); plt.show()

    # Combine RVs for XC
    rvsxc_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    weights_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            rvsxc_single_iter[o, :, :] = rvs_dict["rvsxc"][o, :, :, iter_indices[o, ichunk]]
            weights_single_iter[o, :, :] = weights[o, :, :, iter_indices[o, ichunk]]
    result_xc = pcrvcalc.combine_rvs_tfa(rvsxc_single_iter, weights_single_iter, parser.n_obs_nights)
    
    # Combine RVs for Detrended
    if 'rvsdet' in parser.rvs_dict:
        rvsxcdet_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
        weights_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
        for o in range(parser.n_orders):
            for ichunk in range(parser.n_chunks):
                rvsxcdet_single_iter[o, :, :] = rvs_dict["rvsxcdet"][o, :, :, iter_indices[o, ichunk]]
                weights_single_iter[o, :, :] = weights[o, :, :, iter_indices[o, ichunk]]
        result_det = pcrvcalc.combine_rvs_tfa(rvsxcdet_single_iter, weights_single_iter, parser.n_obs_nights)
    
    # Add to dictionary
    parser.rvs_dict['rvsfwm_out'] = result_nm['rvs']
    parser.rvs_dict['uncfwm_out'] = result_nm['unc']
    parser.rvs_dict['rvsfwm_nightly_out'] = result_nm['rvs_nightly']
    parser.rvs_dict['uncfwm_nightly_out'] = result_nm['unc_nightly']
    
    
    if 'rvsxc' in parser.rvs_dict:
        parser.rvs_dict['rvsxc_out'] = result_xc['rvs']
        parser.rvs_dict['uncxc_out'] = result_xc['unc']
        parser.rvs_dict['rvsxc_nightly_out'] = result_xc['rvs_nightly']
        parser.rvs_dict['uncxc_nightly_out'] = result_xc['unc_nightly']
    
    if 'rvsxcdet' in parser.rvs_dict:
        parser.rvs_dict['rvsxcdet_out'] = result_det['rvs']
        parser.rvs_dict['uncxcdet_out'] = result_det['unc']
        parser.rvs_dict['rvsxcdet_nightly_out'] = result_det['rvs_nightly']
        parser.rvs_dict['uncxcdet_nightly_out'] = result_det['unc_nightly']
    
    # Write to files for radvel
    fname_nightly = parser.output_path_root + 'rvs_nightly_final_' + parser.spectrograph.lower().replace(' ', '_') + '_' + parser.star_name.lower().replace(' ', '_') + '_' + datetime.date.today().strftime("%d%m%Y") + '.txt'
    fname_single = parser.output_path_root + 'rvs_final_' + parser.spectrograph.lower().replace(' ', '_') + '_' + parser.star_name.lower().replace(' ', '_') + '_' + datetime.date.today().strftime("%d%m%Y") + '.txt'
    telvec_nightly = np.array([parser.spectrograph.replace(' ', '_')] * parser.n_nights, dtype='<U20')
    telvec_single = np.array([parser.spectrograph.replace(' ', '_')] * parser.n_spec, dtype='<U20')
    if parser.rvs_out == 'xc':
        good = np.where(np.isfinite(parser.rvs_dict['rvsxc_nightly_out']))[0]
        tn, rvsn, uncn, telvec_nightly = parser.rvs_dict['bjds_nightly'][good], parser.rvs_dict['rvsxc_nightly_out'][good], parser.rvs_dict['uncxc_nightly_out'][good], telvec_nightly[good]
        good = np.where(np.isfinite(parser.rvs_dict['rvsxc_out']))[0]
        ts, rvss, uncs, telvec_single = parser.rvs_dict['bjds'][good], parser.rvs_dict['rvsxc_out'][good], parser.rvs_dict['uncxc_out'][good], telvec_single[good]
    elif parser.rvs_out == 'xcdet':
        good = np.where(np.isfinite(parser.rvs_dict['rvsxcdet_nightly_out']))[0]
        tn, rvsn, uncn, telvec_nightly = parser.rvs_dict['bjds_nightly'][good], parser.rvs_dict['rvsxcdet_nightly_out'][good], parser.rvs_dict['uncxcdet_nightly_out'][good], telvec_nightly[good]
        good = np.where(np.isfinite(parser.rvs_dict['rvsxc_out']))[0]
        ts, rvss, uncs, telvec_single = parser.rvs_dict['bjds'][good], parser.rvs_dict['rvsxcdet_out'][good], parser.rvs_dict['uncxcdet_out'][good], telvec_single[good]
    else:
        good = np.where(np.isfinite(parser.rvs_dict['rvsfwm_nightly_out']))[0]
        tn, rvsn, uncn, telvec_nightly = parser.rvs_dict['bjds_nightly'][good], parser.rvs_dict['rvsfwm_nightly_out'][good], parser.rvs_dict['uncfwm_nightly_out'][good], telvec_nightly[good]
        good = np.where(np.isfinite(parser.rvs_dict['rvsfwm_out']))[0]
        ts, rvss, uncs, telvec_single = parser.rvs_dict['bjds'][good], parser.rvs_dict['rvsfwm_out'][good], parser.rvs_dict['uncfwm_out'][good], telvec_single[good]
        
    with open(fname_nightly, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([tn, rvsn, uncn, telvec_nightly], dtype=object).T, fmt="%f,%f,%f,%s")
    with open(fname_single, 'w+') as f:
        f.write("time,mnvel,errvel,tel\n")
        np.savetxt(f, np.array([ts, rvss, uncs, telvec_single], dtype=object).T, fmt="%f,%f,%f,%s")
    
    
def inspect_bis(parser, iter_indices=None):
    """Coadds the individual BIS values across nights.

    Args:
        parser: A parser.
        
    Returns:
        tuple: The results returned by the call to method.
    """
    
    # Parse RVs
    rvs_dict = parser.parse_rvs()
    
    # Generate weights
    weights = gen_rv_weights(parser)
    
    # Determine indices
    iter_indices = parser.resolve_iter_indices(iter_indices)
    
    # For each order, chunk, and night, compute the coadded BIS values.
    parser.rvs_dict["bis_nightly"] = np.zeros(shape=(parser.n_orders, parser.n_nights, parser.n_chunks), dtype=float)
    parser.rvs_dict["bis_nightly_unc"] = np.zeros(shape=(parser.n_orders, parser.n_nights, parser.n_chunks), dtype=float)
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            bis = parser.rvs_dict['bis'][o, :, ichunk, iter_indices[o] + parser.index_offset]
            _weights = weights[o, :, ichunk, iter_indices[o] + parser.index_offset]
            rvs_dict['bis_nightly'][o, :, ichunk], rvs_dict['bis_nightly_unc'][o, :, ichunk] = pcrvcalc.compute_nightly_rvs_single_chunk(bis.flatten(), _weights.flatten(), parser.n_obs_nights, flag_outliers=True)
    
    time_stamp = datetime.date.today().strftime("%d%m%Y")
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            fname = parser.output_path_root + 'bis_nightly_final_' + parser.spectrograph.lower().replace(' ', '_') + '_' + parser.star_name.lower().replace(' ', '_') + '_ord' + str(o + 1) + '_chunk' + str(ichunk + 1) + "_" + time_stamp + '.txt'
            with open(fname, 'w+') as f:
                t = parser.rvs_dict["bjds_nightly"]
                bis = parser.rvs_dict["bis_nightly"][o, :, ichunk]
                bis_unc = parser.rvs_dict["bis_nightly_unc"][o, :, ichunk]
                f.write("time,mnvel,errvel,tel\n")
                np.savetxt(f, np.array([t, bis, bis_unc], dtype=float).T)
                
                
def plot_bis_vs_rv(parser:pcparser.PostParser, iter_indices=None):
    
    iter_indices = parser.resolve_iter_indices(iter_indices)
    
    # For each order and chunk, plot the BIS vs. RV
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            rvsxc = parser.rvs_dict["rvsxc_nightly"][o, :, ichunk]
            uncxc = parser.rvs_dict["uncxc_nightly"][o, :, ichunk]
            bis = parser.rvs_dict["bis_nightly"][o, :, ichunk]
            uncbis = parser.rvs_dict["bis_nightly_unc"][o, :, ichunk]
            # A figure
            plt.figure(1, figsize=(14, 12), dpi=250)
            plt.errorbar(rvsxc - np.nanmedian(rvsxc), bis, xerr=uncxc, yerr=uncbis, marker='o', lw=0, elinewidth=1.5, alpha=0.8, markersize=10)
            plt.xlabel("RV via XC [m/s]", fontsize=22)
            plt.ylabel("BIS [m/s]", fontsize=22)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            time_stamp = datetime.date.today().strftime("%d%m%Y")
            fname = parser.output_path_root + 'bis_vs_rvsxc' + parser.spectrograph.lower().replace(' ', '_') + '_' + parser.star_name.lower().replace(' ', '_') + '_ord' + str(parser.do_orders[o]) + '_chunk' + str(ichunk + 1) + time_stamp + '.png'
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
    
    
    
# Detrend RVs if applicable
def detrend_rvs(parser: pcparser.PostParser, vec='bis', thresh=0.5):
    
    parser.parse_rvs()
    
    rvs_dict = parser.rvs_dict
    rvsx_detrended = np.zeros((parser.n_orders, parser.n_spec, parser.n_iters_rvs))
    for o in range(parser.n_orders):
        for j in range(parser.n_iters_rvs):
            if rvs_dict['do_xcorr']:
                good = np.where(np.isfinite(rvs_dict['rvsx'][o, :, j]) & np.isfinite(rvs_dict['BIS'][o, :, j]))[0]
                if good.size == 0:
                    continue
                rvsx_detrended[o, good, j] = pcrvcalc.detrend_rvs(rvs_dict['rvsx'][o, good, j], rvs_dict['BIS'][o, good, j], thresh=thresh)
                
    # Add to dictionary
    rvs_dict['rvsx_detrended'] = rvsx_detrended
    rvs_dict['rvsx_nightly_detrended'] = np.zeros((parser.n_orders,  parser.n_nights, parser.n_iters_rvs))
    rvs_dict['uncx_nightly_detrended'] = np.zeros((parser.n_orders,  parser.n_nights, parser.n_iters_rvs))
    
def gen_rv_mask(parser : pcparser.PostParser):
    
    # Return if no dictionary exists
    if not hasattr(parser, 'bad_rvs_dict'):
        return parser.rvs_dict, mask
    
    rvs_dict = parser.rvs_dict
    bad_rvs_dict = parser.bad_rvs_dict
    
    # Initialize a mask
    mask = np.ones(shape=(parser.n_orders, parser.n_spec, parser.n_chunks, parser.n_iters_rvs), dtype=float)
    
    # Mask all spectra for a given night
    if 'bad_nights' in bad_rvs_dict:
        for inight in bad_rvs_dict['bad_nights']:
            inds = parser.forward_models[0][0].get_all_spec_indices_from_night(inight, parser.n_obs_nights)
            mask[:, inds, :, :] = 0
            rvs_dict['rvsfwm'][:, inds, :, :] = np.nan
            if rvs_dict['do_xcorr']:
                rvs_dict['rvsxc'][:, inds, :, :] = np.nan
                rvs_dict['bis'][:, inds, :, :] = np.nan
    
    # Mask individual spectra
    if 'bad_spec' in bad_rvs_dict:
        for i in bad_rvs_dict['bad_spec']:
            mask[:, i, :, :] = 0
            rvs_dict['rvsfwm'][:, i, :, :] = np.nan
            if rvs_dict['do_xcorr']:
                rvs_dict['rvsxc'][:, i, :, :] = np.nan
                rvs_dict['bis'][:, i, :, :] = np.nan
        
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
    
def lsperiodogram(t, rvs, pmin=1.3, pmax=None, dp=0.01, offset=True):
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
    if offset:
        _rvs = rvs[good] - np.median(rvs[good])
    else:
        _rvs = rvs[good]
    pgram = scipy.signal.lombscargle(t[good], _rvs, af)
    return tp, pgram

def print_rv_summary(parser, iter_indices):
    
    rvs_dict = parser.rvs_dict
    
    for o in range(parser.n_orders):
        print('Order ' + str(parser.do_orders[o]))
        for k in range(parser.n_iters_rvs):
            stddev = np.nanstd(rvs_dict['rvsfwm_nightly'][o, :, k])
            
            if rvs_dict['do_xcorr']:
                stddevxc = np.nanstd(rvs_dict['rvsxc_nightly'][o, :, k])
            else:
                stddevxc = np.nan
                
            if k == iter_indices[o, 0]:
                print(' ** Iteration ' +  str(k + 1) + ': ' + str(round(stddev, 4)) + ' m/s')
                print(' ** Iteration ' +  str(k + 1) + ': ' + str(round(stddevxc, 4)) + ' m/s')
            else:
                print('    Iteration ' +  str(k + 1) + ': ' + str(round(stddev, 4)) + ' m/s')
                print('    Iteration ' +  str(k + 1) + ': ' + str(round(stddevxc, 4)) + ' m/s')
                
def gen_rv_weights(parser):
    
    # Generate mask
    mask = gen_rv_mask(parser)
    
    # RMS weights
    fit_metrics = parser.parse_fit_metric()
    weights_fit = 1 / fit_metrics[:, :, :, parser.index_offset:]**2
    bad = np.where(weights_fit < 100)
    if bad[0].size > 0:
        weights_fit[bad] = 0
        
    # RV content weights
    rvconts = compute_rv_contents(parser)
    weights_rvcont = 1 / rvconts**2
    
    # Combine weights, multiplicatively
    weights = weights_rvcont * weights_fit * mask

    return weights

def compute_rv_contents(parser, templates=None):
    
    # Resolve templates to use
    templates = parser.resolve_rvprec_templates(templates)
            
    # The RV contents, for each iteration
    rvcs = np.zeros((parser.n_orders, parser.n_spec, parser.n_chunks, parser.n_iters_rvs))
    
    # The nightly S/N, for each iteration
    nightly_snrs = compute_nightly_snrs(parser)
    
    # Compute RVC
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            parser.forward_models[o][0].init_chunk(parser.forward_models[o].templates_dict)
            pars = parser.forward_models[o][0].opt_results[-1][ichunk]['xbest']
            wave_data = parser.forward_models[o][0].models_dict['wavelength_solution'].build(pars)
            lsf = parser.forward_models[o][0].models_dict['lsf'].build(pars)
            _rvcs = np.zeros(len(templates))
            for itemplate, t in enumerate(templates):
                for j in range(parser.n_iters_rvs):
                    template_wave, template_flux = parser.forward_models[o].templates_dict[t][:, 0], parser.forward_models[o].templates_dict[t][:, 1]
                    _, _rvcs[itemplate] = pcrvcalc.compute_rv_content(template_wave, template_flux, snr=np.nanmedian(nightly_snrs[o, :, j]), blaze=True, ron=0, wave_to_sample=wave_data, lsf=lsf)
                    rvcs[o, :, ichunk, j] = np.nansum(_rvcs**2)**0.5
    return rvcs

def compute_nightly_snrs(parser):

    # Parse the rms
    rms = parser.parse_fit_metric()
    nightly_snrs = np.zeros((parser.n_orders, parser.n_nights, parser.n_chunks, parser.n_iters_rvs))

    for o in range(parser.n_orders):
        f, l = 0, parser.n_obs_nights[0]
        for inight in range(parser.n_nights):
            for j in range(parser.n_iters_rvs):
                for ichunk in range(parser.n_chunks):
                    nightly_snrs[o, inight, ichunk, j] = np.nansum((1 / rms[o, f:l, ichunk, j + parser.index_offset])**2)**0.5
            if inight < parser.n_nights - 1:
                f += parser.n_obs_nights[inight]
                l += parser.n_obs_nights[inight + 1]
    nightly_snrs = np.nanmedian(nightly_snrs, axis=2)
    return nightly_snrs


def parameter_corrs(parser, iter_indices=None, debug=False, n_iters_plot=1, highlight=None):
    
    plt.style.use("seaborn")
    
    # Parse the RVs
    rvs_dict = parser.parse_rvs()
    
    # Generate mask
    mask = gen_rv_mask(parser)
    
    # Iterations
    iter_indices = parser.resolve_iter_indices(iter_indices)
    
    # Loop over orders and chunks
    for o in range(parser.n_orders):
        
        for ichunk in range(parser.n_chunks):
        
            vp = parser.forward_models[o][0].opt_results[iter_indices[o, ichunk] + parser.index_offset][0]['xbest'].unpack(keys='vary')['vary']
            vpi = np.where(vp)[0]
            nv = vpi.size
            
            pars = np.empty(shape=(parser.n_spec, parser.n_iters_opt, nv), dtype=object)
            par_vals = np.full(shape=(parser.n_spec, parser.n_iters_opt, nv), dtype=float, fill_value=np.nan)
            par_names = list(parser.forward_models[o][0].opt_results[iter_indices[o, ichunk] + parser.index_offset][ichunk]['xbest'].keys())
            par_names = [par_names[v] for v in vpi]
            for ispec in range(parser.n_spec):
                for j in range(parser.n_iters_opt):
                    for k in range(nv):
                        pars[ispec, j, k] = parser.forward_models[o][ispec].opt_results[j][ichunk]['xbest'][par_names[k]]
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
                    
                    # Views to arrays
                    _rvs = rvs_dict['rvsfwm'][o, :, ichunk, -n_iters_plot:] # (n_spec, n_iters_plot)
                    _pars = par_vals[:, -n_iters_plot:, k] # (n_spec, n_iters_plot)
                    par_range = np.nanmax(_pars) - np.nanmin(_pars)
                    par_low = pcmath.weighted_median(_pars, percentile=0.01)
                    par_high = pcmath.weighted_median(_pars, percentile=0.99)
                    par_range = par_high - par_low
                    par_low -= 0.1 * par_range
                    par_high += 0.1 * par_range
                    rv_low = pcmath.weighted_median(_rvs, percentile=0.1) - 20 
                    rv_high = pcmath.weighted_median(_rvs, percentile=0.9) + 20
                        
                    if n_iters_plot > 1:
                        for ispec in range(parser.n_spec):
                            axarr[row, col].plot(_rvs[ispec, :], _pars[ispec, :], alpha=0.6, c='powderblue', lw=0.7)
                    
                    axarr[row, col].plot(_rvs[:, -1], _pars[:, -1], marker='.', lw=0, c='black', markersize=5, alpha=0.8)
                    if highlight is not None:
                        axarr[row, col].plot(_rvs[highlight, -1], _pars[highlight, -1], marker='.', lw=0, c='red', markersize=5, alpha=0.8)
                    axarr[row, col].set_xlabel('RV [m/s]', fontsize=4)
                    axarr[row, col].set_ylabel(par_names[k].replace('_', ' '), fontsize=4)
                    axarr[row, col].tick_params(axis='both', which='major', labelsize=4)
                    axarr[row, col].grid(None)
                    axarr[row, col].set_xlim(rv_low, rv_high)
                    axarr[row, col].set_ylim(par_low, par_high)
            fig.suptitle(parser.star_name.replace('_', ' ') + ' Parameter Correlations Order ' + str(parser.do_orders[o]), fontsize=10)
            fname = parser.output_path_root + 'Order' + str(parser.do_orders[o]) + os.sep + 'parameter_corrs_ord' + str(parser.do_orders[o]) + '_chunk' + str(ichunk + 1) +'.png'
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.5)
            plt.savefig(fname)
            plt.close()
        
    if debug:
        breakpoint()
        
    
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
    bjds, bjds_nightly = parser.rvs_dict['bjds'], parser.rvs_dict['bjds_nightly']
    if parser.rvs_out == 'XC':
        rvs_single, unc_single, rvs_nightly, unc_nightly = parser.rvs_dict['rvsxc_out'], parser.rvs_dict['uncxc_out'], parser.rvs_dict['rvsxc_nightly_out'], parser.rvs_dict['uncxc_nightly_out']
    elif parser.rvs_out == 'XCDET':
        rvs_single, unc_single, rvs_nightly, unc_nightly = parser.rvs_dict['rvsxcdet_out'], parser.rvs_dict['uncxcdet_out'], parser.rvs_dict['rvsxcdet_nightly_out'], parser.rvs_dict['uncxcdet_nightly_out']
    else:
        rvs_single, unc_single, rvs_nightly, unc_nightly = parser.rvs_dict['rvsfwm_out'], parser.rvs_dict['uncfwm_out'], parser.rvs_dict['rvsfwm_nightly_out'], parser.rvs_dict['uncfwm_nightly_out']
    
    # Single rvs
    plt.errorbar((bjds - alpha)%_phase_to, rvs_single-np.nanmedian(rvs_nightly), yerr=unc_single, linewidth=0, elinewidth=1, marker='.', markersize=10, markerfacecolor='pink', color='green', alpha=0.8)

    # Nightly RVs
    plt.errorbar((bjds_nightly - alpha)%_phase_to, rvs_nightly-np.nanmedian(rvs_nightly), yerr=unc_nightly, linewidth=0, elinewidth=2, marker='o', markersize=10, markerfacecolor='blue', color='grey', alpha=0.9)
    
    plt.title(parser.star_name + ', ' + parser.spectrograph + ' Relative RVs')
    
    if kamp is not None:
        plt.plot(modelx, modely, label='K = ' + str(kamp) + ' m/s')
    
    if phase_to is None:
        plt.xlabel('BJD - BJD$_{0}$')
    else:
        plt.xlabel('Phase [days, P = ' +  str(round(_phase_to, 3)) + ']')
    plt.ylabel('RV [m/s]')
    plt.tight_layout()
    plt.savefig(parser.output_path_root + 'rvs_final_' + parser.spectrograph.lower().replace(' ', '_') + '_' + parser.star_name.lower().replace(' ', '_') + '.png', dpi=200, figsize=(12, 8))
    plt.show()
    
    
def stellar_template_diffs(parser:pcparser.PostParser):
    
    stellar_templates = parser.parse_stellar_templates()
    
    residuals = parser.parse_residuals()
    
    #for o in range(parser.n_orders):
        
            
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
            

def residual_coherence(parser, iter_indices=None, frame='star', nsample=1, templates=None):
    
    # Parse forward models
    parser.parse_forward_models()
    
    # Resolve the iterations to use
    iter_indices = parser.resolve_iter_indices(iter_indices)
    
    # Resolve templates to use
    templates = parser.resolve_rvprec_templates(templates)
    
    # To coadd
    res = np.full(shape=(parser.forward_models[0][0].n_model_pix_order, parser.n_spec), fill_value=np.nan)
    
    # Create a figure
    plot_height_single = 4
    fig, axarr = plt.subplots(nrows=parser.n_orders, ncols=1, figsize=(10, int(plot_height_single*parser.n_orders)), dpi=250)
    axarr = np.atleast_1d(axarr)
    
    # Loop over orders
    for o in range(parser.n_orders):
        
        # Plot residuals for each observation
        for i in range(0, parser.n_spec, nsample):
            
            # Alias
            fwm = parser.forward_models[o][i]
            
            for ichunk, sregion in enumerate(fwm.chunk_regions):
            
                # Init the chunk
                templates_dict_chunked = fwm.init_chunk(parser.forward_models[o].templates_dict, sregion)
            
                # Get the best fit parameters
                pars = fwm.opt_results[iter_indices[o, ichunk] + parser.index_offset][ichunk]['xbest']
            
                # Build the model
                wave_data, model_lr = fwm.build_full(pars, templates_dict_chunked)
            
                # Stellar rest frame
                if frame == 'star':
                    
                    # Determine the relative shift
                    if iter_indices[o, 0] == 0 and not parser.forward_models[o][i].models_dict['star'].from_synthetic:
                        vel = parser.forward_models[o][i].data.bc_vel
                    else:
                        vel = -1 * pars[fwm.models_dict['star'].par_names[0]].value
                    
                    # Shift the wavelength solution
                    wave_shifted = pcmath.doppler_shift(wave_data, vel=vel, flux=None, interp=None)
            
                    # The residuals for this iteration
                    residuals = fwm.data.flux_chunk - model_lr
                
                    # Interpolate so we don't store the unique wavelength grids
                    good = np.where(np.isfinite(residuals) & np.isfinite(wave_shifted))[0]
                    res[:, i] = scipy.interpolate.CubicSpline(wave_shifted[good], residuals[good], extrapolate=False)(templates_dict_chunked['star'][:, 0])
                    
                    # Plot the lr version
                    axarr[o].plot(wave_shifted, residuals, alpha=0.7)
                
                # Lab rest frame
                else:
                    
                    # The residuals for this iteration
                    residuals = parser.forward_models[o][i].data.flux - model_lr
                    
                    # Interpolate
                    good = np.where(np.isfinite(residuals))
                    res[:, i] = scipy.interpolate.CubicSpline(wave_shifted[good], residuals[good], extrapolate=False)(templates_dict_chunked['star'][:, 0])

                    # Plot the lr version
                    axarr[o].plot(wave_data, residuals, alpha=0.7)
        
                # Plot the template to visually determine any correlations
                for t in templates:
                    if type(templates_dict_chunked[t]) is dict:
                        for tt in templates_dict_chunked[t]:
                            w, f = templates_dict_chunked[:, 0], templates_dict_chunked[:, 1]
                            ww = np.linspace(np.nanmin(w), np.nanmax(w), num=w.size)
                            ff = np.interp(ww, w, f, left=np.nan, right=np.nan)
                            fc = parser.forward_models[o][0].models_dict['lsf'].convolve_flux(ff, pars=pars)
                            axarr[o].plot(ww, fc, alpha=0.8, label=tt)
                    else:
                        w, f = templates_dict_chunked[t][:, 0], templates_dict_chunked[t][:, 1]
                        ww = np.linspace(np.nanmin(w), np.nanmax(w), num=w.size)
                        ff = np.interp(ww, w, f, left=np.nan, right=np.nan)
                        fc = parser.forward_models[o][0].models_dict['lsf'].convolve_flux(ff, pars=pars)
                        axarr[o].plot(ww, fc, alpha=0.8, label=t)
                        
                    if frame == 'star' and t == 'star':
                        w, f = templates_dict_chunked[t][:, 0], templates_dict_chunked[t][:, 1]
                        fc = fwm.models_dict['lsf'].convolve_flux(f, pars=pars)
                        axarr[o].plot(ww, fc, alpha=0.8, label=t)
                
                
            axarr[o].plot(templates_dict_chunked['star'][:, 0], np.nanmedian(res, axis=1), c='black')
            
        axarr[o].set_title('Order ' + str(parser.do_orders[o]) + ' iter ' + str(iter_indices[o, ichunk] + 1), fontsize=8)
        axarr[o].tick_params(axis='both', labelsize=10)
        axarr[o].legend()
        for ichunk in range(1, parser.n_chunks):
            axarr[o].vlines(x=parser.forward_models[o][0].chunk_regions[ichunk].wavemin, ymin=-0.1, ymax=0.1, c='black', ls=':')
    axarr[-1].set_xlabel('Wavelength [Ang]')
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.95, wspace=None, hspace=0.5)
    plt.tight_layout()
    fig.text(0.03, 0.5, 'Norm. flux', rotation=90, verticalalignment='center', horizontalalignment='center', fontsize=10)
    fig.text(0.5, 0.97, parser.star_name, fontsize=10, verticalalignment='center', horizontalalignment='center')
    plt.savefig(parser.output_path_root + 'residuals_coherence_' + pcutils.gendatestr(time=False) + '_.png')
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
            lsf = forward_models[o, i].models_dict['lsf'].build(pars=forward_models[o, i].opt_results[iter_indices[o]]['xbest'])
            
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
            wls = forward_models[o, i].models_dict['wavelength_solution'].build(forward_models[o, i].opt_results[iter_indices[o]]['xbest'])
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
        
        x = forward_models[o, 0].models_dict['wavelength_solution'].build(forward_models[o, 0].opt_results[iter_indices[o]]['xbest'])
        nx = x.size
        
        blazes = np.zeros((nx, n_spec))
        
        for i in range(n_spec):
            
            # Build blaze
            blaze = forward_models[o, i].models_dict['blaze'].build(forward_models[o, i].opt_results[iter_indices[o]]['xbest'], x)
            
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
            
def rvs_quicklook(parser, bad_rvs_dict, iter_indices=None, phase_to=None, tc=None, thresh=None, debug=False, kamp=None):
    
    if phase_to is None:
        _phase_to = 1E20  
    else:
        _phase_to = phase_to
        
    if tc is None:
        alpha = 0
    else:
        alpha = tc - phase_to / 2
    
    # Parse RVs
    parser.parse_rvs()
    
    # Print summary
    iter_indices = parser.resolve_iter_indices(iter_indices=iter_indices)
    print_rv_summary(parser, iter_indices)
    
    mask = gen_rv_mask(parser)
    
    # Combine RVs for NM
    rvsfwm_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    rvsxc_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    weightsfwm_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    weightsxc_single_iter = np.full(shape=(parser.n_orders, parser.n_spec, parser.n_chunks), fill_value=np.nan)
    for o in range(parser.n_orders):
        for ichunk in range(parser.n_chunks):
            rvsfwm_single_iter[o, :, :] = parser.rvs_dict["rvsfwm"][o, :, :, iter_indices[o, ichunk]]
            rvsxc_single_iter[o, :, :] = parser.rvs_dict["rvsxc"][o, :, :, iter_indices[o, ichunk]]
            weightsfwm = 1 / parser.rvs_dict["uncfwm_nightly"][o, :, iter_indices[o, ichunk]]**2
            weightsxc = 1 / parser.rvs_dict["uncxc_nightly"][o, :, iter_indices[o, ichunk]]**2
            for ispec in range(parser.n_spec):
                night_index = pcforwardmodels.ForwardModel.get_night_index(ispec, parser.rvs_dict["n_obs_nights"])
                weightsfwm_single_iter[o, :, :] = weightsfwm[night_index] * mask[o, ispec, ichunk, iter_indices[o, ichunk]]
                weightsxc_single_iter[o, :, :] = weightsxc[night_index] * mask[o, ispec, ichunk, iter_indices[o, ichunk]]
    #result_nm = pcrvcalc.combine_relative_rvs(rvsfwm_single_iter, weights_single_iter, parser.n_obs_nights)
    result_nm = pcrvcalc.combine_rvs_tfa(rvsfwm_single_iter, weightsfwm_single_iter, parser.n_obs_nights)
    result_xc = pcrvcalc.combine_rvs_tfa(rvsxc_single_iter, weightsxc_single_iter, parser.n_obs_nights)
    if parser.rvs_out.lower() == "fwm":
        rvs_final = result_nm["rvs"]
        unc_final = result_nm["unc"]
        rvs_nightly_final = result_nm["rvs_nightly"]
        unc_nightly_final = result_nm["unc_nightly"]
    else:
        rvs_final = result_xc["rvs"]
        unc_final = result_xc["unc"]
        rvs_nightly_final = result_xc["rvs_nightly"]
        unc_nightly_final = result_xc["unc_nightly"]
    bjds, bjdsn = parser.rvs_dict["bjds"], parser.rvs_dict["bjds_nightly"]
    plt.errorbar((bjdsn - alpha)%_phase_to, rvs_nightly_final - np.nanmedian(rvs_nightly_final), yerr=unc_nightly_final, marker='o', lw=0, elinewidth=1, label='Binned Nightly', c='black', markersize=10)
    if kamp is not None:
        modelx = np.linspace(0, _phase_to, num=300)
        modely = kamp * np.sin(2 * np.pi * modelx / _phase_to)
        plt.plot(modelx, modely)
    plt.legend()
    plt.show()
    
    if debug:
        breakpoint()
        
    return result_nm, result_xc
        
    