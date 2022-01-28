# Base Python
import os

# Maths
import numpy as np

# pychell deps
import pychell.data.rvdata as pcrvdata
import pychell.orbits as pco

# Optimize deps
import optimize as opt

# Path name 
path = os.path.dirname(os.path.abspath(__file__)) + os.sep
fname = 'kelt24_rvs.txt'
star_name = 'KELT-24'
mstar = 1.460
mstar_unc = [0.059, 0.055]
jitter_dict = {'SONG': 30, 'TRES': 0}

# All data in one dictionary
data = pcrvdata.CompositeRVData.from_radvel_file(fname, wavelengths=None)

# Init parameters and planets dictionary
pars = pco.BayesianParameters()
planets_dict = {}

# Planet 1
planets_dict[1] = {"label": "b", "basis": pco.bases.TCOrbitBasis(1)}
per1 = 5.5514926
per1_unc = 0.0000081
tc1 =  2457147.0529
tc1_unc = 0.002
ecc1 = 0.077
ecc1_unc = 0.024 * 2
w1 = 55 * np.pi / 180
w1_unc = 15 * np.pi / 180

# Note that here the RVs will not help to constrain the period or tc because they are better constrained by the transits, but here we float them for demonstration.
pars["per1"] = pco.BayesianParameter(value=per1, vary=True)
pars["per1"].add_prior(pco.priors.Gaussian(per1, per1_unc))

pars["tc1"] = pco.BayesianParameter(value=tc1, vary=True)
pars["tc1"].add_prior(pco.priors.Gaussian(tc1, tc1_unc))

pars["ecc1"] = pco.BayesianParameter(value=ecc1, vary=True)
pars["ecc1"].add_prior(pco.priors.Uniform(1E-10, 1))
pars["ecc1"].add_prior(pco.priors.Gaussian(ecc1, ecc1_unc))

pars["w1"] = pco.BayesianParameter(value=w1, vary=True)
pars["w1"].add_prior(pco.priors.Gaussian(w1, w1_unc))

pars["k1"] = pco.BayesianParameter(value=462, vary=True)
pars["k1"].add_prior(pco.priors.Positive())

# Gamma offsets, don't include gamma dot or ddot yet
for instname in data:
    # np.pi/100 incase median is zero
    pars["gamma_" + instname] = pco.BayesianParameter(value=np.nanmedian(data[instname].rv) + np.pi/100, vary=True)
    pars["gamma_" + instname].add_prior(pco.priors.Gaussian(pars["gamma_" + instname].value, 100))

# jitter
for instname in data:
    pname = f"jitter_{instname}"
    pars[pname] = pco.BayesianParameter(value=jitter_dict[instname], vary=jitter_dict[instname] > 0)
    if pars[pname].vary:
        pars[pname].add_prior(pco.priors.JeffreysSG(1E-10, 100))

# noise, model, and like
likes = {}
noise_process = opt.WhiteNoiseProcess()
model = pco.RVModel(planets_dict=planets_dict, data=data, trend_poly_order=0)
likes["rvs"] = pco.RVLikelihood(model=model, noise_process=noise_process)

# Posterior
post = opt.Posterior(likes=likes)

# Exoplanet problem
rvprob = pco.RVProblem(p0=pars, output_path=path, star_name=star_name, post=post, tag="EXAMPLE")

# Map fit
map_result = rvprob.run_mapfit(opt.IterativeNelderMead(maximize=True))

pbest = map_result["pbest"]

rvprob.plot_full_rvs(pbest)
rvprob.plot_phased_rvs_all(pbest)

print(pbest)

mcmc_result = rvprob.run_mcmc(opt.emceeSampler(), p0=pbest, n_burn_steps=1000, n_taus_thresh=50, n_min_steps=5000)
rvprob.corner_plot(mcmc_result)
