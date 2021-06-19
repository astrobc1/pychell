# Maths
import numpy as np

# Pychell deps
from pychell.spectralmodeling.spectralrvprob import IterativeSpectralRVProb
import pychell.utils as pcutils

# Main function
def compute_rvs_for_target(specrvprob:IterativeSpectralRVProb):
    """The main function to run for a given target to compute the RVs for iterative spectral rv problems.

    Args:
        specrvprob (IterativeSpectralRVProb): The spectral RV Problem.
    """

    # Start the main clock!
    stopwatch = pcutils.StopWatch()
    stopwatch.lap(name='ti_main')
            
    # Iterate over remaining stellar template generations
    for iter_index in range(specrvprob.n_iterations):
        
        if iter_index == 0 and specrvprob.spectral_model.models_dict['star'].from_flat:
            
            print(f"Starting Iteration {iter_index + 1} of {specrvprob.n_iterations} (flat template, no RVs) ...", flush=True)
            stopwatch.lap(name='ti_iter')
            
            # Fit all observations
            specrvprob.optimize_all_observations(0)
            
            print(f"Finished Iteration {iter_index + 1} in {round(stopwatch.time_since(name='ti_iter')/3600, 2)} hours", flush=True)
            
            # Augment the template
            if iter_index < specrvprob.n_iterations - 1:
                specrvprob.augment_templates(iter_index)
        
        else:
            
            print(f"Starting Iteration {iter_index + 1} of {specrvprob.n_iterations} ...", flush=True)
            stopwatch.lap(name='ti_iter')

            # Run the fit for all spectra and do a cross correlation analysis as well.
            specrvprob.optimize_all_observations(iter_index)
        
            # Run the ccf for all spectra
            specrvprob.cross_correlate_spectra(iter_index)
        
            # Generate the rvs for each observation
            specrvprob.gen_nightly_rvs(iter_index)
        
            # Plot the rvs
            specrvprob.plot_rvs(iter_index)
        
            # Save the rvs each iteration
            specrvprob.save_rvs()
            
            print(f"Finished Iteration {iter_index + 1} in {round(stopwatch.time_since(name='ti_iter')/3600, 2)} hours", flush=True)

            # Print RV Diagnostics
            if specrvprob.n_spec >= 1:
                rvs_std = np.nanstd(specrvprob.rvs_dict['rvsfwm'][:, iter_index])
                print(f"  Stddev of all fwm RVs: {round(rvs_std, 4)} m/s", flush=True)
                rvs_std = np.nanstd(specrvprob.rvs_dict['rvsxc'][:, iter_index])
                print(f"  Stddev of all xc RVs: {round(rvs_std, 4)} m/s", flush=True)
            if specrvprob.n_nights > 1:
                rvs_std = np.nanstd(specrvprob.rvs_dict['rvsfwm_nightly'][:, iter_index])
                print(f"  Stddev of all fwm nightly RVs: {round(rvs_std, 4)} m/s", flush=True)
                rvs_std = np.nanstd(specrvprob.rvs_dict['rvsxc_nightly'][:, iter_index])
                print(f"  Stddev of all xc nightly RVs: {round(rvs_std, 4)} m/s", flush=True)
                
            # Augment the template
            if iter_index < specrvprob.n_iterations - 1:
                specrvprob.augment_templates(iter_index)
            

    # Save forward model outputs
    print("Saving results ... ", flush=True)
    specrvprob.save_to_pickle()
    
    # End the clock!
    print(f"Completed order {specrvprob.order_num} Runtime: {round(stopwatch.time_since(name='ti_main') / 3600, 2)} hours", flush=True)
