# Base Python libraries
import os

# Import numpy
import numpy as np

# Import streamlit
import streamlit as st

# Import optimize sub modules
import optimize.knowledge as optknow
import optimize.optimizers as optimizers
import optimize.kernels as optkernels
import optimize.samplers as optsamplers

# Import pychell sub modules
import pychell.orbits.rvprob as pcrvprob
import pychell.orbits.rvdata as pcrvdata
import pychell.orbits.rvlikes as pcrvlikes
import pychell.orbits.rvmodels as pcrvmodels
import pychell.stutils as pcstutils

# Define user input here
use_st = True
path = os.path.dirname(os.path.abspath(__file__)) + os.sep
fname = 'kelt24_rvs_all_20210127.txt'
star_name = 'KELT-24'
mstar = 1.460
mstar_unc = [0.059, 0.055]
jitter_dict = {'MINERVANorth': 10, 'SONG': 30, 'TRES': 0}

# All data in one dictionary
data = pcrvdata.MixedRVData.from_radvel_file(fname)

# Use or ignore data
comps = {}
if use_st:
    pcstutils.make_title(title=star_name + " RV Optimize")
    pcstutils.DataSelector(comps, data)

# Init parameters and planets dictionary
pars = optknow.Parameters()
planets_dict = {}

# Planet 1
planets_dict[1] = {"label": "b", "basis": pcrvmodels.TCOrbitBasis(1)}
per1 = 5.5514926
per1_unc = 0.0000081
tc1 =  2457147.0529
tc1_unc = 0.002
ecc1 = 0.077
ecc1_unc = 0.024
w1 = 55 * np.pi / 180
w1_unc = 15 * np.pi / 180
pars["per1"] = optknow.Parameter(value=per1, vary=False)
#pars["per1"].add_prior(optknow.Gaussian(per1, per1_unc))

pars["tc1"] = optknow.Parameter(value=tc1, vary=False)
#pars["tc1"].add_prior(optknow.Gaussian(tc1, tc1_unc))

pars["ecc1"] = optknow.Parameter(value=ecc1, vary=True)
pars["ecc1"].add_prior(optknow.Uniform(1E-10, 1))
#pars["ecc1"].add_prior(optknow.Gaussian(ecc1, ecc1_unc))

pars["w1"] = optknow.Parameter(value=w1, vary=True)
pars["w1"].add_prior(optknow.Gaussian(w1, w1_unc))
pars["k1"] = optknow.Parameter(value=462, vary=True)
pars["k1"].add_prior(optknow.Positive())

# Gamma offsets, don't include gamma dot or ddot yet
for instname in data:
    pars["gamma_" + instname] = optknow.Parameter(value=np.nanmedian(data[instname].rv) + np.pi / 1000, vary=True) # pi in case median is already 0.
    #pars["gamma_" + instname].add_prior(optknow.Gaussian(pars["gamma_" + instname].value, 20))
    
# Gamma dot and ddot
pars["gamma_dot"] = optknow.Parameter(value=0, vary=False)
pars["gamma_ddot"] = optknow.Parameter(value=0, vary=False)

# jitter
for instname in data:
    pname = "jitter_" + instname
    pars[pname] = optknow.Parameter(value=jitter_dict[instname], vary=jitter_dict[instname] > 0)
    if pars[pname].vary:
        #pars[pname].add_prior(optknow.Gaussian(pars[pname].value, 5))
        pars[pname].add_prior(optknow.Jeffreys(1E-10, 100))

# Form the noise kernels, models, and (sub) data sets (all 1 to 1)
likes = pcrvlikes.MixedRVLikelihood()

# Kernel, model, and like
kernel = optkernels.WhiteNoise(data=data)
model = pcrvmodels.RVModel(planets_dict=planets_dict, data=data, p0=pars, kernel=kernel)
likes["rvs"] = pcrvlikes.RVLikelihood(data=data, model=model)

# Max like optimizer (iterative Nelder-Mead) and emcee MCMC sampler
optimizer = optimizers.NelderMead(scorer=likes)
sampler = optsamplers.AffInv(scorer=likes, options=None)

# Define exoplanet "problem"
optprob = pcrvprob.ExoProblem(output_path=path, star_name=star_name, p0=pars, optimizer=optimizer, sampler=sampler, data=data, likes=likes, mstar=mstar, mstar_unc=mstar_unc)

# Remaining st components
if use_st:
    pcstutils.RVActions(comps, optprob)
    if comps["max_like_button"]:
        opt_result = optprob.optimize()
        pcstutils.MaxLikeResult(comps, optprob, opt_result)
    if comps["model_comp_button"]:
        mc_result = optprob.model_comparison()
        pcstutils.ModelCompResult(comps, optprob, mc_result)
    if comps["sample_button"]:
        sampler_result = optprob.sample(n_min_steps=3000)
        pcstutils.MCMCResult(comps, optprob, sampler_result)
        pcstutils.PlanetsResults(comps, optprob, sampler_result)
    if comps["per_search_button"]:
        if  comps["persearch_kind_input"] == "Brute Force":
            periods, persearch_result = optprob.rv_period_search(pmin=float(comps["persearch_min_input"]), pmax=float(comps["persearch_max_input"]), planet_index=2, n_periods=750, n_cores=8)
            pcstutils.RVPeriodSearchResult(comps, optprob, periods, persearch_result)
        else:
            gls_result = optprob.gls_periodogram(apply_gp=comps["use_gp_input"], remove_planets=comps["remove_planet_inputs"], pmin=float(comps["persearch_min_input"]), pmax=float(comps["persearch_max_input"]))
            pcstutils.GLSResult(comps, optprob, gls_result)
    if comps["rvcolor_button"]:
        maxlike_result = optprob.optimize()
        rvcolor_result = optprob.compute_rvcolor(maxlike_result["pbest"])
        pcstutils.RVColorResult(comps, optprob, rvcolor_result)