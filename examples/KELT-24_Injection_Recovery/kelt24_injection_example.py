# Base Python libraries
import os
import sys

# Import numpy
import numpy as np

# Import pychell orbits and utils modules
import pychell.orbits as pco
import pychell.utils as pcutils

# Path to input rv file and outputs for outputs
output_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
fname = 'kelt24_rvs.txt'

# The name of the star for plots
star_name = 'KELT-24'

# Mass and uncertainty of star for mass determination (solar units)
mstar = 1.460
mstar_unc = [0.059, 0.055] # -, +

# All data in one dictionary from a radvel formatted file
data = pco.CompositeRVData.from_radvel_file(fname)

# Init parameters and planets dictionary
pars = pco.BayesianParameters()
planets_dict = {}

# Used later to set initial values
jitter_dict = {"TRES": 0, "SONG": 50}

# Define parameters for planet 1
# Other bases are available, this fits for P, TC, ECC, W, K.
planets_dict[1] = {"label": "b", "basis": pco.orbitbases.TCOrbitBasis(1)}

# Values from Rodriguez et al. 2019 for KELT-24 b
# Given the extremely precise values for P and TC from the multiple transits, there is no reason to fit for P and TC here.
# We will still fit for P and TC for demonstrative purposes.
per1 = 5.5514926
per1_unc = 0.0000081
tc1 =  2457147.0529
tc1_unc = 0.002
ecc1 = 0.077
ecc1_unc = 0.024
w1 = 55 * np.pi / 180
w1_unc = 15 * np.pi / 180

# Period
pars["per1"] = pco.BayesianParameter(value=per1, vary=True)
pars["per1"].add_prior(pco.priors.Gaussian(per1, per1_unc))

# Time of conjunction
pars["tc1"] = pco.BayesianParameter(value=tc1, vary=True)
pars["tc1"].add_prior(pco.priors.Gaussian(tc1, tc1_unc))

# Eccentricity
pars["ecc1"] = pco.BayesianParameter(value=ecc1, vary=True)
pars["ecc1"].add_prior(pco.priors.Uniform(1E-10, 1))
pars["ecc1"].add_prior(pco.priors.Gaussian(ecc1, ecc1_unc))

# Angle of periastron
pars["w1"] = pco.BayesianParameter(value=w1, vary=True)
pars["w1"].add_prior(pco.priors.Gaussian(w1, w1_unc))

# RV semi-amplitude
pars["k1"] = pco.BayesianParameter(value=462, vary=True)
pars["k1"].add_prior(pco.priors.Positive())

# The injection and recovery tests need a dummy planet to perform injections on, so we define such a planet here
planets_dict[2] = {"label": "d", "basis": pco.orbitbases.TCOrbitBasis(2)}

# The value of the period doesn't matter, as it will be changed to the injection values.
# Vary can be either true or false, depending on how strict one wants the testing to be.
# If you vary it and want to add Gaussian or Uniform priors that adjust to the injection values,
# there are arguments in the injection/recovery class that allow you to do this.
pars["per2"] = pco.BayesianParameter(value=999, vary=False)

# Similarly, the TC value doesn't matter, as it will be overwritten by the injections.
# It can also be varied or not varied depending on preference.
pars["tc2"] = pco.BayesianParameter(value=0., vary=False)

# We will restrict ourselves to injecting circular planets, so e and w stay at 0.
pars["ecc2"] = pco.BayesianParameter(value=0, vary=False)
pars["w2"] = pco.BayesianParameter(value=0, vary=False)

# The semiamplitude is one parameter that we DEFINITELY want to vary for these tests.
pars["k2"] = pco.BayesianParameter(value=10, vary=True)
pars["k2"].add_prior(pco.priors.Positive())

# Per instrument zero points
# Additional small offset is to avoid cases where the median is already subtracted off.
for instname in data:
    data[instname].y += 300
    pname = "gamma_" + instname
    pars[pname] = pco.BayesianParameter(value=np.nanmedian(data[instname].rv) + np.pi, vary=True)
    pars[pname].add_prior(pco.priors.Uniform(pars[pname].value - 200, pars[pname].value + 200))
    
# Linear and quadratic trends, fix at zero
pars["gamma_dot"] = pco.BayesianParameter(value=0, vary=False)
pars["gamma_ddot"] = pco.BayesianParameter(value=0, vary=False)

# Per-instrument jitter (only fit for SONG jitter, TRES jitter is typically sufficient.)
for instname in data:
    pname = "jitter_" + instname
    pars[pname] = pco.BayesianParameter(value=jitter_dict[instname], vary=jitter_dict[instname] > 0)
    if pars[pname].vary:
        pars[pname].add_prior(pco.priors.Uniform(1E-10, 100))

# Initialize the injection & recovery object
# Keep in mind, the scorer, likelihood, kernel, process, model, optimizer, and sampler types are all shown with their
# DEFAULT arguments.  They are just presented here to demonstrate how one would change them if needed, but it is not
# necessary to explicitly define these types when declaring an InjectionRecovery class.
injection_recovery = pco.InjectionRecovery(data=data, p0=pars, planets_dict=planets_dict,
                                           output_path=os.path.join(output_path, 'injection_runs_{}'.format(pcutils.gendatestr(True))) + os.sep,
                                           star_name=star_name, k_range=(1, 50), p_range=(1.1, 10), k_resolution=10, p_resolution=10,
                                           scorer_type=pco.RVPosterior, likelihood_type=pco.RVLikelihood, kernel_type=None,
                                           process_type=pco.RVJitter, model_type=pco.CompositeRVModel, optimizer_type=pco.IterativeNelderMead,
                                           sampler_type=pco.emceeSampler)

response = ''
while response not in ['y', 'n']:
    response = input('WARNING: This example will attempt to run 100 MCMC simulations.  This will require either a beefy CPU or '
                     'a lot of patience.  Continue? [y/n]: ')
if response == 'n':
    sys.exit()
else:
    # Now, we want to run all of the injection MCMCs at each combination of P and K to test the recovery.
    # To test false positive rates, change injection to False.  This applies to all methods hereafter.
    injection_recovery.full_mcmc_run(injection=True)

    # Similarly, we want to run maximum likelihood fits to obtain delta AICc information
    injection_recovery.full_maxlikefit_run()

    # Now we organize the data to prepare for plotting. This step is required.
    injection_recovery.organize_injection_data()

    # Finally, we plot 1D and 2D histograms and delta AICcs
    injection_recovery.plot_injection_2D_hist(xtickprecision=1, ytickprecision=1)
    injection_recovery.plot_1d_histograms()
    injection_recovery.plot_delta_aicc()