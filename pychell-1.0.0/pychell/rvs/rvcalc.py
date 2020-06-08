import os
from pdb import set_trace as stop

# Multiprocessing
from joblib import Parallel, delayed

# Maths
import numpy as np
import scipy.interpolate

# Graphics
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

import pychell.maths as pcmath
import pychell.utils as pcutils
import copy


def generate_rvs(forward_models, iter_num):
    """Genreates individual and nightly (co-added) RVs after forward modeling all spectra and stores them in the ForwardModels object. If do_xcorr is True, nightly cross-correlation RVs are also computed along with the line bisector and BIS.

    Args:
        forward_models (ForwardModels): The list of forward model objects.
        iter_num (int): The iteration to generate RVs from.
    """
    
    # k1 = index for forward model array access    
    # k2 = Plot names for forward model objects
    # k3 = index for RV array access
    # k4 = RV plot names
    k1, k2, k3, k4 = forward_models[0].iteration_indices(iter_num)

    # The nightly RVs and error bars.
    rvs_nightly = np.full(forward_models.n_nights, fill_value=np.nan)
    unc_nightly = np.full(forward_models.n_nights, fill_value=np.nan)
    rvs_xcorr_nightly = np.full(forward_models.n_nights, fill_value=np.nan)
    unc_xcorr_nightly = np.full(forward_models.n_nights, fill_value=np.nan)

    # The best fit stellar RVs, remove the barycenter velocity
    rvs = np.array([forward_models[ispec].best_fit_pars[k1][forward_models[ispec].models_dict['star'].par_names[0]].value + forward_models[ispec].data.bc_vel for ispec in range(forward_models.n_spec)], dtype=np.float64)
    
    # The CC vels
    if forward_models.do_xcorr:
        rvs_xcorr = np.array([forward_models[ispec].rvs_xcorr[k3] for ispec in range(forward_models.n_spec)], dtype=np.float64)

    # The RMS from the forward model fit
    rms = np.array([forward_models[ispec].opt[k1][0] for ispec in range(forward_models.n_spec)], dtype=np.float64)

    # Co-add to get nightly RVs
    # If only one spectrum and no initial guess, no rvs!
    if forward_models.n_spec == 1 and not forward_models.models_dict['star'].from_synthetic:
        rvs[0] = np.nan
        rvs_nightly[0] = np.nan
        unc_nightly[0] = np.nan
        rvs_xcorr_nightly[0] = np.nan
        unc_xcorr_nightly[0] = np.nan
    else:
        f = 0
        l = forward_models.n_obs_nights[0]
        for inight in range(forward_models.n_nights):
            
            # Nelder Mead RVs
            rvs_single_night = rvs[f:l]
            w = 1 / rms[f:l]**2
            if forward_models.n_obs_nights[inight] > 1:
                wavg = pcmath.weighted_mean(rvs_single_night, w)
                wstddev = pcmath.weighted_stddev(rvs_single_night, w)
                good = np.where(np.abs(rvs_single_night - wavg) < 4*wstddev)[0]
                if good.size == 0:
                    good = np.arange(rvs_single_night.size).astype(int)
                rvs_nightly[inight] = pcmath.weighted_mean(rvs_single_night[good], w[good])
                unc_nightly[inight] = pcmath.weighted_stddev(rvs_single_night[good], w[good]) / np.sqrt(forward_models.n_obs_nights[inight])
            else:
                rvs_nightly[inight] = rvs_single_night[0]
                unc_nightly[inight] = 0
                
                
            # xcorr RVs
            if forward_models.do_xcorr:
                rvs_single_night = rvs_xcorr[f:l]
                if forward_models.n_obs_nights[inight] > 1:
                    wavg = pcmath.weighted_mean(rvs_single_night, w)
                    wstddev = pcmath.weighted_stddev(rvs_single_night, w)
                    good = np.where(np.abs(rvs_single_night - wavg) < 4*wstddev)[0]
                    if good.size == 0:
                        good = np.arange(rvs_single_night.size).astype(int)
                    rvs_xcorr_nightly[inight] = pcmath.weighted_mean(rvs_single_night[good], w[good])
                    unc_xcorr_nightly[inight] = pcmath.weighted_stddev(rvs_single_night[good], w[good]) / np.sqrt(forward_models.n_obs_nights[inight])
                else:
                    rvs_xcorr_nightly[inight] = rvs_single_night[0]
                    unc_xcorr_nightly[inight] = 0
                
            # Step to next night
            if inight < forward_models.n_nights - 1:
                f += forward_models.n_obs_nights[inight]
                l += forward_models.n_obs_nights[inight+1]
            
    
                
                
    # Store Outputs
    
    # Nelder Mead
    forward_models.rvs_dict['rvs'][:, k3] = rvs
    forward_models.rvs_dict['rvs_nightly'][:, k3] = rvs_nightly
    forward_models.rvs_dict['unc_nightly'][:, k3] = unc_nightly
    
    # X Corr
    if forward_models.do_xcorr:
        for ispec in range(forward_models.n_spec):
            forward_models.rvs_dict['xcorr_vels'][:, ispec, k3] = forward_models[ispec].xcorr_vels[:, k3]
            forward_models.rvs_dict['xcorrs'][:, ispec, k3] = forward_models[ispec].xcorrs[:, k3]
            forward_models.rvs_dict['line_bisectors'][:, ispec, k3] = forward_models[ispec].line_bisectors[:, k3]
            forward_models.rvs_dict['bisector_spans'][ispec, k3] = forward_models[ispec].bisector_spans[k3]
        
        # X Corr RVs
        forward_models.rvs_dict['rvs_xcorr'][:, k3] = rvs_xcorr
        forward_models.rvs_dict['rvs_xcorr_nightly'][:, k3] = rvs_xcorr_nightly
        forward_models.rvs_dict['unc_xcorr_nightly'][:, k3] = unc_xcorr_nightly

        
def plot_rvs(forward_models, iter_num):
    """Plots all RVs and cross-correlation analysis after forward modeling all spectra.

    Args:
        forward_models (ForwardModels): The list of forward model objects.
        iter_num (int): The iteration to use.
    """
    
    # k1 = index for forward model array access    
    # k2 = Plot names for forward model objects
    # k3 = index for RV array access
    # k4 = RV plot names
    k1, k2, k3, k4 = forward_models[0].iteration_indices(iter_num)
    
    # Plot the rvs, nightly rvs, xcorr rvs, xcorr nightly rvs
    plot_width, plot_height = 1800, 600
    dpi = 200
    plt.figure(num=1, figsize=(int(plot_width/dpi), int(plot_height/dpi)), dpi=200)
    
    # Alias
    rvs = forward_models.rvs_dict
    
    # Individual rvs from nelder mead fitting
    plt.plot(forward_models.BJDS - forward_models.BJDS_nightly[0],
             rvs['rvs'][:, k3] - np.nanmedian(rvs['rvs_nightly'][:, k3]),
             marker='.', linewidth=0, alpha=0.7, color=(0.1, 0.8, 0.1))
    
    # Nightly rvs from nelder mead fitting
    plt.errorbar(forward_models.BJDS_nightly - forward_models.BJDS_nightly[0],
                    rvs['rvs_nightly'][:, k3] - np.nanmedian(rvs['rvs_nightly'][:, k3]),
                    yerr=rvs['unc_nightly'][:, k3], marker='o', linewidth=0, elinewidth=1, label='Nelder Mead', color=(0, 114/255, 189/255))

    # Individual and nightly xcorr rvs
    if forward_models.do_xcorr:
        plt.plot(forward_models.BJDS - forward_models.BJDS_nightly[0],
                    rvs['rvs_xcorr'][:, k3] - np.nanmedian(rvs['rvs_xcorr'][:, k3]),
                    marker='.', linewidth=0, color='black', alpha=0.6)
        plt.errorbar(forward_models.BJDS_nightly - forward_models.BJDS_nightly[0],
                        rvs['rvs_xcorr_nightly'][:, k3] - np.nanmedian(rvs['rvs_xcorr_nightly'][:, k3]),
                        yerr=rvs['unc_xcorr_nightly'][:, k3], marker='X', linewidth=0, alpha=0.8, label='X Corr', color='darkorange')
    
    plt.title(forward_models[0].star_name + ' RVs Order ' + str(forward_models.order_num) + ', Iteration ' + str(k4), fontweight='bold')
    plt.xlabel('BJD - BJD$_{0}$', fontweight='bold')
    plt.ylabel('RV [m/s]', fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    fname = forward_models.run_output_path_rvs + forward_models.tag + '_rvs_ord' + str(forward_models.order_num) + '_iter' + str(k4) + '.png'
    plt.savefig(fname)
    plt.close()
    
    if forward_models.do_xcorr:
        # Plot the Bisector stuff
        plt.figure(1, figsize=(12, 7), dpi=200)
        for ispec in range(forward_models.n_spec):
            v0 = rvs['rvs_xcorr'][ispec, k3]
            depths = np.linspace(0, 1, num=forward_models.n_bs)
            ccf_ = rvs['xcorrs'][:, ispec, k3] - np.nanmin(rvs['xcorrs'][:, ispec, k3])
            ccf_ = ccf_ / np.nanmax(ccf_)
            plt.plot(rvs['xcorr_vels'][:, ispec, k3] - v0, ccf_)
            plt.plot(rvs['line_bisectors'][:, ispec, k3], depths)
        
        plt.title(forward_models.star_name + ' CCFs Order ' + str(forward_models.order_num) + ', Iteration ' + str(k4), fontweight='bold')
        plt.xlabel('RV$_{\star}$ [m/s]', fontweight='bold')
        plt.ylabel('CCF (RMS surface)', fontweight='bold')
        plt.xlim(-10000, 10000)
        plt.tight_layout()
        fname = fname = forward_models.run_output_path_rvs + forward_models.tag + '_ccfs_ord' + str(forward_models.order_num) + '_iter' + str(k4) + '.png'
        plt.savefig(fname)
        plt.close()
    
        # Plot the Bisector stuff
        plt.figure(1, figsize=(12, 7), dpi=200)
        plt.plot(rvs['rvs_xcorr'][:, k3], rvs['bisector_spans'][:, k3], marker='o', linewidth=0)
        plt.title(forward_models[0].star_name + ' CCF Bisector Spans Order ' + str(forward_models.order_num) + ', Iteration ' + str(k4), fontweight='bold')
        plt.xlabel('X Corr RV [m/s]', fontweight='bold')
        plt.ylabel('Bisector Span [m/s]', fontweight='bold')
        plt.tight_layout()
        fname = fname = forward_models.run_output_path_rvs + forward_models.tag + '_bisectorspans_ord' + str(forward_models.order_num) + '_iter' + str(k4) + '.png'
        plt.savefig(fname)
        plt.close()
        

def cross_correlate_all(forward_models, iter_num):
    """Cross correlation wrapper for all spectra.

    Args:
        forward_models (ForwardModels): The list of forward model objects.
        iter_num (int): The iteration to use.
    """

    # Fit in Parallel
    stopwatch = pcutils.StopWatch()
    print('Cross Correlating Spectra ... ', flush=True)

    if forward_models.n_cores > 1:

        # Construct the arguments
        iter_pass = []
        for ispec in range(forward_models.n_spec):
            iter_pass.append((forward_models[ispec], forward_models.n_spec, iter_num))

        # Cross Correlate in Parallel
        forward_models[:] = Parallel(n_jobs=forward_models.n_cores, verbose=0, batch_size=1)(delayed(cc_wrapper)(*iter_pass[ispec]) for ispec in range(forward_models.n_spec))
        
    else:
        for ispec in range(forward_models.n_spec):
            forward_models[ispec] = cc_wrapper(forward_models[ispec], forward_models.n_spec, iter_num)
            
    print('Cross Correlation Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)


def cc_wrapper(forward_model, n_spec_tot, iter_num):
    """Cross correlation wrapper for a single spectrum.

    Args:
        forward_model (ForwardModel): A single forward model object.
        n_spec_tot (int): The total number of spectra.
        iter_num (int): The iteration to use.
    """
    stopwatch = pcutils.StopWatch()
    cross_correlate_star(forward_model, iter_num)
    print('    Cross Correlated Spectrum ' + str(forward_model.data.spec_num) + ' of ' + str(n_spec_tot) + ' in ' + str(round((stopwatch.time_since()), 2)) + ' sec', flush=True)
    return forward_model


def get_nightly_jds(jds, sep=0.5):
    """Computes nightly (average) JDs (or BJDs) for a time-series observation over several nights.

    Args:
        jds (np.ndarray): An array of sorted JDs (or BJDs).
        sep (float): The minimum separation in days between two different nights of data, defaults to 0.5.
    """
    
    # Number of spectra
    n_spec = len(jds)

    prev_i = 0
    # Calculate mean JD date and number of observations per night for nightly
    # coadded RV points; assume that observations on separate nights are
    # separated by at least 0.5 days.
    jds_nightly = []
    n_obs_nights = []
    if n_spec == 1:
        jds_nightly.append(jds[0])
        n_obs_nights.append(1)
    else:
        for i in range(n_spec-1):
            if jds[i+1] - jds[i] > sep:
                jd_avg = np.average(jds[prev_i:i+1])
                n_obs_night = i - prev_i + 1
                jds_nightly.append(jd_avg)
                n_obs_nights.append(n_obs_night)
                prev_i = i + 1
        jds_nightly.append(np.average(jds[prev_i:]))
        n_obs_nights.append(n_spec - prev_i)

    jds_nightly = np.array(jds_nightly) # convert to np arrays
    n_obs_nights = np.array(n_obs_nights).astype(int)

    return jds_nightly, n_obs_nights


def cross_correlate_star(forward_model, iter_num):
    """Performs a cross-correlation via brute force RMS minimization and estimating the minumum with a quadratic.

    Args:
        forward_model (ForwardModel): A single forward model object.
        iter_num (int): The iteration to use.
        
    Returns:
        forward_model (ForwardModel): The forward model object with cross-correlation results stored in place.
    """
    
    
    # k1 = index for forward model array access
    # k2 = Plot names for forward model objects
    # k3 = index for RV array access
    # k4 = RV plot names
    k1, k2, k3, k4 = forward_model.iteration_indices(iter_num)
        
    if forward_model.crude:
        pars = copy.deepcopy(forward_model.initial_parameters)
        vels = np.arange(-250000, 250000, 500)
    else:
        pars = copy.deepcopy(forward_model.best_fit_pars[k1])
        v0 = pars[forward_model.models_dict['star'].par_names[0]].value
        vels = np.linspace(v0-forward_model.xcorr_range, v0+forward_model.xcorr_range, num=forward_model.n_xcorr_vels)

    # Stores the rms as a function of velocity
    rms = np.empty(vels.size, dtype=np.float64)
    
    # Weights for now are just bad pixels
    weights_init = np.copy(forward_model.data.badpix)
        
    for i in range(vels.size):
        
        weights = weights_init
        
        # Set the RV parameter to the current step
        pars[forward_model.models_dict['star'].par_names[0]].setv(value=vels[i])
        
        # Build the model
        _, model_lr = forward_model.build_full(pars, iter_num)
        
        # Construct the RMS
        rms[i] = np.sqrt(np.nansum((forward_model.data.flux - model_lr)**2 * weights / np.nansum(weights)))

    xcorr_star_vel = vels[np.nanargmin(rms)]
    xcorr_rv_init = xcorr_star_vel + forward_model.data.bc_vel # Actual RV is bary corr corrected
    vels_for_rv = vels + forward_model.data.bc_vel
        
    # Fit the CCF with a polynomial
    if not forward_model.crude:
        use = np.where((vels_for_rv > xcorr_rv_init - 150) & (vels_for_rv < xcorr_rv_init + 150))[0]
        pfit = np.polyfit(vels_for_rv[use], rms[use], 2)
        forward_model.rvs_xcorr[k3] = -1 * pfit[1] / (2 * pfit[0])
        forward_model.xcorr_vels[:, k3] = vels_for_rv
        forward_model.xcorrs[:, k3] = rms
        bspan_result = compute_bisector_span(forward_model.xcorr_vels[:, k3], forward_model.xcorrs[:, k3], n_bs=forward_model.n_bs)
        forward_model.line_bisectors[:, k3] = bspan_result[0]
        forward_model.bisector_spans[k3] = bspan_result[1]
    
    # If this is the first iteration, update the stellar rv
    if forward_model.crude:
        forward_model.initial_parameters[forward_model.models_dict['star'].par_names[0]].setv(value=xcorr_star_vel)
    
    return forward_model
                        

def compute_bisector_span(cc_vels, ccf, n_bs=1000):
    """Computes the bisector span of a given cross-correlation (RMS brute force) function.

    Args:
        cc_vels (np.ndarray): The velocities used for cross-correlation.
        ccf (int): The corresponding 1-dimensional RMS curve.
        n_bs (int): The number of depths to use in calculating the BIS, defaults to 1000
        
    Returns:
        line_bisectors (np.ndarray): The line bisectors of the ccf.
        bisector_span (float): The bisecor span of the line bisector (commonly referred to as the BIS).
    """



    # B(d) = (v_l(d) + v_r(d)) / 2
    # v_l = velocities located on the left side from the minimum of the CCF peak and v_r are the ones on the right side
    # Mean bisector is computed at two depth ranges:
    # d = (0.1, 0.4), (0.6, 0.85)
    # B_(0.1, 0.4) = E(B(d)) for 0.1 to 0.4
    # B_(0.6, 0.85) = E(B(d)) for 0.6 to 0.85
    # BS = B_(0.1, 0.4) - B_(0.6, 0.85) = E(B(d)) for 0.1 to 0.4 - E(B(d)) for 0.6 to 0.85
    # .. = Average(B(d)) for 0.1 to 0.4 = Average((v_l(d) + v_r(d)) / 2) for 0.1 to 0.4
    
    # The bottom "half"
    drtop = (0.1, 0.4)
    
    # The top "half"
    drbottom = (0.6, 0.8)
    
    # The depths are from 0 to 1 for the normalized CCF
    depths = np.linspace(0, 1, num=n_bs)

    # Initialize the line bisector array (a function of CCF depth)
    line_bisectors = np.empty(depths.size, dtype=np.float64)

    # First normalize the CCF function
    ccf = ccf - np.nanmin(ccf)
    cont_ = pcmath.weighted_median(ccf, med_val=0.95)
    ccf = ccf / cont_
    
    # Get the velocities and offset such that the best vel is at zero
    best_vel = cc_vels[np.nanargmin(ccf)]
    cc_vels = cc_vels - best_vel
    
    # High res version of the ccf
    cc_vels_hr = np.linspace(np.nanmin(cc_vels), np.nanmax(cc_vels), num=int(cc_vels.size*100))
    good = np.where(np.isfinite(cc_vels) & np.isfinite(ccf))[0]
    ccf_hr = scipy.interpolate.CubicSpline(cc_vels[good], ccf[good], extrapolate=False)(cc_vels_hr)
    
    # The vels on the left and right of the best vel.
    use_left = np.where(cc_vels < 0)[0]
    use_right = np.where(cc_vels > 0)[0]
    use_left_hr = np.where(cc_vels_hr < 0)[0]
    use_right_hr = np.where(cc_vels_hr > 0)[0]
    if use_left.size == 0 or use_right.size == 0:
        return np.nan, np.nan
    
    vel_max_ind_left, vel_max_ind_right = use_left[np.nanargmax(ccf[use_left])], use_right[np.nanargmax(ccf[use_right])]
    vel_max_ind_left_hr, vel_max_ind_right_hr = use_left_hr[np.nanargmax(ccf_hr[use_left_hr])], use_right_hr[np.nanargmax(ccf_hr[use_right_hr])]
    
    use_left = np.where((cc_vels > cc_vels[vel_max_ind_left]) & (cc_vels < 0))[0]
    use_right = np.where((cc_vels > 0) & (cc_vels < cc_vels[vel_max_ind_right]))[0]
    
    use_left_hr = np.where((cc_vels_hr > cc_vels_hr[vel_max_ind_left_hr]) & (cc_vels_hr < 0))[0]
    use_right_hr = np.where((cc_vels_hr > 0) & (cc_vels_hr < cc_vels_hr[vel_max_ind_right_hr]))[0]
    
    # Compute the line bisector
    for idepth in range(depths.size):
        d = depths[idepth]
        vl = cc_vels_hr[use_left_hr[pcmath.find_closest(ccf_hr[use_left_hr], d)[0]]]
        vr = cc_vels_hr[use_right_hr[pcmath.find_closest(ccf_hr[use_right_hr], d)[0]]]
        line_bisectors[idepth] = (vl + vr) / 2

    # Compute the bisector span
    top = np.where((depths > drtop[0]) & (depths < drtop[1]))[0]
    bottom = np.where((depths > drbottom[0]) & (depths < drbottom[1]))[0]
    avg_top = np.nanmean(line_bisectors[top])
    avg_bottom = np.nanmean(line_bisectors[bottom])
    
    # Store the bisector span
    bisector_span = avg_top - avg_bottom
    
    return line_bisectors, bisector_span