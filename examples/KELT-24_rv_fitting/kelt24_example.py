# Base Python libraries
import os

# Import numpy
import numpy as np

import pychell.orbits as pco

# Path name 
path = os.path.dirname(os.path.abspath(__file__)) + os.sep
fname = 'kelt24_rvs.txt'
star_name = 'KELT-24'
mstar = 1.460
mstar_unc = [0.059, 0.055]
jitter_dict = {'MINERVANorth': 10, 'SONG': 30, 'TRES': 0}

# All data in one dictionary
data = pco.CompositeRVData.from_radvel_file(fname, wavelengths=None)

# Init parameters and planets dictionary
pars = pco.BayesianParameters()
planets_dict = {}

# Planet 1
planets_dict[1] = {"label": "b", "basis": pco.orbitbases.TCOrbitBasis(1)}
per1 = 5.5514926
per1_unc = 0.0000081
tc1 =  2457147.0529
tc1_unc = 0.002
ecc1 = 0.077
ecc1_unc = 0.024 * 2
w1 = 55 * np.pi / 180
w1_unc = 15 * np.pi / 180
pars["per1"] = pco.BayesianParameter(value=per1, vary=False)
#pars["per1"].add_prior(pco.priors.Gaussian(per1, per1_unc))

pars["tc1"] = pco.BayesianParameter(value=tc1, vary=False)
#pars["tc1"].add_prior(pco.priors.Gaussian(tc1, tc1_unc))

pars["ecc1"] = pco.BayesianParameter(value=ecc1, vary=True)
pars["ecc1"].add_prior(pco.priors.Uniform(1E-10, 1))
pars["ecc1"].add_prior(pco.priors.Gaussian(ecc1, ecc1_unc))

pars["w1"] = pco.BayesianParameter(value=w1, vary=True)
pars["w1"].add_prior(pco.priors.Gaussian(w1, w1_unc))
pars["k1"] = pco.BayesianParameter(value=462, vary=True)
#pars["k1"].add_prior(optknow.Gaussian(8.5, 2.5))
pars["k1"].add_prior(pco.priors.Positive())

# Gamma offsets, don't include gamma dot or ddot yet
for instname in data:
    pars["gamma_" + instname] = pco.BayesianParameter(value=np.nanmedian(data[instname].rv) + np.pi, vary=True) # pi in case median is already 0.
    pars["gamma_" + instname].add_prior(pco.priors.Gaussian(pars["gamma_" + instname].value, 100))

# jitter
for instname in data:
    pname = f"jitter_{instname}"
    pars[pname] = pco.BayesianParameter(value=jitter_dict[instname], vary=jitter_dict[instname] > 0)
    if pars[pname].vary:
        pars[pname].add_prior(pco.priors.JeffreysG(1E-10, 100))
    

# Posterior
post = pco.RVPosterior()

# noise, model, and like
noise = pco.RVJitter(data=data)
model = pco.CompositeRVModel(planets_dict=planets_dict, data=data, noise_process=noise, poly_order=0)
post["rvs"] = pco.RVLikelihood(model=model)

# Exoplanet problem
rvprob = pco.RVProblem(p0=pars, output_path=path, star_name=star_name, post=post, optimizer=pco.IterativeNelderMead(), sampler=pco.ZeusSampler(), tag="EXAMPLE")
mc_result = rvprob.model_comparison()
map_result = rvprob.run_mapfit()

pbest = map_result["pbest"]

rvprob.plot_full_rvs(pbest)
rvprob.plot_phased_rvs_all(pbest)

print(pbest)
rvprob.set_pars(pbest)
mcmc_result = rvprob.run_mcmc(n_burn_steps=1000, n_taus_thresh=50, n_min_steps=5000)
rvprob.corner_plot(mcmc_result)
