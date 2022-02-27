import os

import dill

import numpy as np

import pychell.utils as pcutils
import pychell.spectralmodeling.plotting

from joblib import Parallel, delayed

#####################
#### MAIN METHOD ####
#####################

def compute_rvs_iteratively(specrvprob, obj, optimizer, augmenter, output_path, tag, n_iterations, n_cores=1, do_ccf=False, verbose=False):

    # Start the main clock!
    stopwatch = pcutils.StopWatch()
    stopwatch.lap(name='ti_main')

    # Summary
    print(f"Observatory: {specrvprob.spec_module.observatory} / {specrvprob.spectrograph}", flush=True)
    print(f"Observations: {len(specrvprob)} spectra, {specrvprob.model.star.star_name}", flush=True)
    print(f"Model: {specrvprob.model}", flush=True)

    # Create output dirs
    if output_path[-1] != os.sep:
        output_path += os.sep
    output_path += specrvprob.spectrograph.lower() + "_" + tag + os.sep + specrvprob.model.sregion.label + os.sep + os.sep
    os.makedirs(output_path + "Fits", exist_ok=True)
    os.makedirs(output_path + "RVs", exist_ok=True)
    os.makedirs(output_path + "Templates", exist_ok=True)

    # Store optimization results in a list of lists of dicts: List[List[dict]]
    # The outer list indexes iteration, the inner list the observation
    opt_results = []

    # Store rvs in dict of arrays and save to npz file
    rvs = {}
    rvs["bjds"] = np.array([float(d.header["bjd"]) for d in specrvprob.data])
    rvs["rvsfwm"] = np.full((len(specrvprob), n_iterations+1), np.nan)
    if do_ccf:
        rvs["rvsxc"] = np.full((len(specrvprob), n_iterations+1), np.nan)

    # Get templates
    specrvprob.model.load_templates(specrvprob.data)

    # Get initial parameters
    p0s = [specrvprob.model.get_init_parameters(d) for d in specrvprob.data]

    # Stellar templates
    stellar_templates = np.full((len(specrvprob.model.templates["wave"]), n_iterations + 1), np.nan)
    stellar_templates[:, 0] = np.copy(specrvprob.model.templates["wave"])
    if not specrvprob.model.star.from_flat:
        stellar_templates[:, 1] = np.copy(specrvprob.model.templates["star"])
            
    # Iterate over remaining stellar template generations
    for iter_index in range(n_iterations):
        
        if iter_index == 0 and hasattr(specrvprob.model, "star") and specrvprob.model.star is not None and specrvprob.model.star.from_flat:
            
            print(f"Starting iteration {iter_index + 1} of {n_iterations} (flat template, no RVs) [{specrvprob.model.sregion.label}]", flush=True)
            stopwatch.lap(name='ti_iter')
            
            # Fit all observations
            _opt_results = optimize_all_observations(specrvprob, p0s, obj, optimizer, iter_index, output_path, n_cores, verbose)
            opt_results.append(_opt_results)
            
            print(f"Finished iteration {iter_index + 1} of {n_iterations} (flat template, no RVs) [{specrvprob.model.sregion.label}] in {round(stopwatch.time_since(name='ti_iter')/60, 2)} min", flush=True)
            
            # Augment the template
            if iter_index < n_iterations - 1:
                augmenter.augment_template(specrvprob, _opt_results, iter_index)
        
        else:
            
            print(f"Starting iteration {iter_index + 1} of {n_iterations} [{specrvprob.model.sregion.label}]", flush=True)
            stopwatch.lap(name='ti_iter')

            # Starting parameters
            if iter_index > 0:
                p0s = [opt_result["pbest"] for opt_result in opt_results[-1]]
                if iter_index == 1 and specrvprob.model.star.from_flat:
                    for ispec in range(len(specrvprob)):
                        p0s[ispec][specrvprob.model.star.par_names[0]].vary = True


            # Run the fit for all spectra and do a cross correlation analysis as well.
            _opt_results = optimize_all_observations(specrvprob, p0s, obj, optimizer, iter_index, output_path, n_cores, verbose)
            opt_results.append(_opt_results)

            rvs["rvsfwm"][:, iter_index] = np.array([_opt_results[ispec]["pbest"][specrvprob.model.star.par_names[0]].value + float(specrvprob.data[ispec].header["bc_vel"]) for ispec in range(len(specrvprob))])
        
            # Run the ccf for all spectra
            if do_ccf:
                p0s = [_opt_result["pbest"] for opt_result in _opt_results[-1]]
                rvs["rvsxc"][:, iter_index] = cross_correlate_all_observations(specrvprob, p0s, iter_index, output_path, n_cores)
        
            # Plot the rvs
            pychell.spectralmodeling.plotting.plot_rvs_single_chunk(specrvprob, rvs, iter_index, output_path, time_offset=2450000)
        
            # Save the rvs each iteration
            print(f"Finished iteration {iter_index + 1} of {n_iterations} [{specrvprob.model.sregion.label}] in {round(stopwatch.time_since(name='ti_iter')/60, 2)} min", flush=True)
            print("Saving intermediate results ...")
            specrvprob.save_to_pickle(output_path)
            save_opt_results(specrvprob, opt_results, output_path)
            save_rvs(specrvprob, rvs, output_path)
            save_stellar_templates(specrvprob, stellar_templates, output_path)

            # Print RV Diagnostics
            if len(specrvprob) >= 1:
                print(f"Stddev of all fwm RVs: {round(np.nanstd(rvs['rvsfwm'][:, iter_index]), 3)} m/s", flush=True)
                if do_ccf:
                    print(f"Stddev of all xc RVs: {round(np.nanstd(rvs['rvsxc'][:, iter_index]), 3)} m/s", flush=True)
                
            # Augment the template
            if iter_index < n_iterations - 1:
                augmenter.augment_template(specrvprob, _opt_results, iter_index)
                stellar_templates[:, iter_index+1] = np.copy(specrvprob.model.templates["star"])
            

    # Save forward model outputs
    print("Saving results ... ", flush=True)
    specrvprob.save_to_pickle(output_path)
    save_opt_results(specrvprob, opt_results, output_path)
    save_rvs(specrvprob, rvs, output_path)
    save_stellar_templates(specrvprob, stellar_templates, output_path)
    
    # End the clock!
    print(f"Completed {specrvprob.model.sregion} in {round(stopwatch.time_since(name='ti_main') / 3600, 2)} hours", flush=True)
    

###############################
#### OPTIMIZATION WRAPPERS ####
###############################

def optimize_all_observations(specrvprob, p0s, obj, optimizer, iter_index, output_path, n_cores, verbose):
        
    # Timer
    stopwatch = pcutils.StopWatch()

    # Parallel fitting
    if n_cores > 1:
        
        # Call the parallel job via joblib.
        opt_results = Parallel(n_jobs=n_cores, verbose=0, batch_size=1)(delayed(optimize_and_plot_observation)(p0s[ispec], specrvprob.data[ispec], specrvprob.model, obj, optimizer, iter_index, output_path, verbose) for ispec in range(len(specrvprob)))

    else:

        # Fit one observation at a time
        opt_results = []
        for ispec in range(len(specrvprob)):
            _opt_results = optimize_and_plot_observation(p0s[ispec], specrvprob.data[ispec], specrvprob.model, obj, optimizer, iter_index, output_path, verbose)
            opt_results.append(_opt_results)

    return opt_results
        

def optimize_and_plot_observation(p0, data, model, obj, optimizer, iter_index, output_path, verbose=False):

    # Lock parameters for lower_bound == upper_bound
    p0.sanity_lock()

    #try:

    # Time the fit
    stopwatch = pcutils.StopWatch()

    # Fit
    opt_result = optimizer.optimize(p0, lambda pars: obj.compute_obj(pars, data, model))

    # Print diagnostics
    print(f"Fit {data} in {round((stopwatch.time_since())/60, 2)} min", flush=True)
    if verbose:
        print(f" RMS = {round(opt_result['fbest'], 3)}", flush=True)
        print(f" Calls = {opt_result['fcalls']}", flush=True)
        print(f" Best Fit Parameters:\n{model.summary(opt_result['pbest'])}", flush=True)

    # Plot
    pychell.spectralmodeling.plotting.plot_spectum_fit(data, model, opt_result["pbest"], obj, iter_index, output_path)

    #except:

    # print(f"Failed to fit observation {data}")

    # # Return nan pars and set to bad
    # opt_result = dict(pbest=p0.gen_nan_pars(), fbest=np.nan, fcalls=np.nan)
    
    # Return result
    return opt_result


######################
#### CCF WRAPPERS ####
######################

def cross_correlate_all_observations(specrvprob, p0s, iter_index, output_path, n_cores):
    stopwatch = pcutils.StopWatch()
    if n_cores > 1:
        ccf_results = Parallel(n_jobs=n_cores, verbose=0, batch_size=1)(delayed(cross_correlate_observation)(p0s[ispec], specrvprob.data[ispec], specrvprob.model, iter_index) for ispec in range(len(specrvprob)))
        
    else:
        ccf_results = [cross_correlate_observation(p0s[ispec], specrvprob.model, iter_index) for ispec in range(len(specrvprob))]

    return ccf_results

def cross_correlate_observation(p0, data, model, iter_index):
    try:
        ccf_result = pcrvcalc.brute_force_ccf(p0, data, model, iter_index)
    except:
        print(f"Warning! Could not perform CCF for {data} [{model.sregion.label}]")
        ccf_result = np.nan, np.nan, np.nan
    return ccf_result



######################
#### SAVE RESULTS ####
######################

def save_rvs(specrvprob, rvs, output_path):
    fname = f"{output_path}RVs{os.sep}rvs_{specrvprob.model.sregion.label.lower()}.npz"
    np.savez(fname, **rvs)

def save_opt_results(specrvprob, opt_results, output_path):
    fname = f"{output_path}Fits{os.sep}opt_results_{specrvprob.model.sregion.label.lower()}.pkl"
    with open(fname, "wb") as f:
        dill.dump(opt_results, f)

def save_stellar_templates(specrvprob, stellar_templates, output_path):
    fname = f"{output_path}Templates{os.sep}stellar_templates_{specrvprob.model.sregion.label.lower()}.txt"
    wave = specrvprob.model.templates["wave"]
    np.savetxt(fname, stellar_templates, delimiter=',')