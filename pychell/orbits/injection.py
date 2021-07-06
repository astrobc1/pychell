import copy
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import pickle
import json
import glob
import gc
import itertools
from joblib import Parallel, delayed
from numba import njit

import pychell.maths as pcmath
import pychell.orbits.planetmaths as planetmath
import pychell.data.rvdata as pcrvdata
import pychell.orbits.rvmodels as pcrvmodels
import pychell.orbits.rvnoise as pcrvnoise
import pychell.orbits.rvobjectives as pcrvobj
import pychell.orbits.rvprob as pcrvprob
import pychell.orbits.orbitbases as pcorbitbases

import optimize.data as optdata
import optimize.frameworks as optframe
import optimize.knowledge as optknow
import optimize.models as optmodels
import optimize.neldermead as optnelder
import optimize.noise as optnoise
import optimize.objectives as optobj
import optimize.optimizers as optopt
import optimize.samplers as optsamplers
import optimize.scipy_optimizers as optscipy


class SoloInjectionRecovery:

    def __init__(self, rv_problem, output_path=None, star_name=None):
        assert type(rv_problem) is pcrvprob.RVProblem
        self.rv_problem = rv_problem
        self.star_name = star_name if star_name else 'Star'

        # Alias output path
        if output_path is None:
            output_path = os.path.abspath(os.path.dirname(__file__))
        self.path = output_path

    def injection_mcmc(self, folder_name=None, injection=True, *args, **kwargs):
        """
        Runs a single injection/recovery MCMC and outputs to a sub-folder in the output directory.

        Args:
            folder_name: Name of folder to save the MCMC output to.  Defaults to an appropriate name.
            injection: bool.  Whether or not to actually inject the planet.
            *args: Arguments for the MCMC function
            **kwargs: Keyword arguments for the MCMC function

        Returns:
            results_dict: dict. Contains MCMC results and priors.
        """
        injected_planet = list(self.rv_problem.planets_dict)[-1]
        k_inj = self.rv_problem.p0["k" + str(injected_planet)].value
        p_inj = self.rv_problem.p0["per" + str(injected_planet)].value

        # Set up output folder
        if not folder_name:
            folder_name = os.path.join(self.path, '{}injection_run_{:.5f}d_{:.5f}mps'.format('non' if not injection else '', p_inj, k_inj))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Run the MCMC
        sampler_result = self.rv_problem.run_mcmc(*args, **kwargs)
        cornerplot = self.rv_problem.corner_plot(mcmc_result=sampler_result)

        # Save results to a pickle file for later use
        results_dict = {'sampler_result': sampler_result, 'priors': self.rv_problem.p0}
        with open(os.path.join(folder_name, '{}_{}injected_{}d_{}mps.pkl'.format(
            self.rv_problem.star_name.replace(' ', '_'), 'non' if not injection else '', p_inj, k_inj
        )), 'wb') as handle:
            pickle.dump(results_dict, handle)

        del sampler_result, cornerplot
        gc.collect()

        print("COMPLETED MCMC AT {}d, {}mps".format(p_inj, k_inj))
        return results_dict

    def injection_maxlikelihood(self, folder_name=None, *args, **kwargs):
        """
        Runs a maximum likelihood fit for the injected planet.
        Args:
            folder_name: str, the names of the two folders to save outputs to.  Defaults to an appropriate name.
            *args: arguments for the optimize function.
            **kwargs: keyword arguments for the optimize function.

        Returns:
            results_dict [dict], results_dict2 [dict]
            The results for both runs, in dictionary form.
        """
        injected_planet = list(self.rv_problem.planets_dict)[-1]
        k_inj = self.rv_problem.p0["k" + str(injected_planet)].value
        p_inj = self.rv_problem.p0["per" + str(injected_planet)].value

        # Set up output folder
        if not folder_name:
            folder_name = os.path.join(self.path, '{}p_likelihood_run_{:.5f}d_{:.5f}mps'.format(len(self.rv_problem.planets_dict), p_inj, k_inj))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.rv_problem.output_path = folder_name

        # Run twice: once with the injected planet and once without
        opt_result = self.rv_problem.run_mapfit(save=False, *args, **kwargs)

        results_dict = {'opt_result': opt_result, 'rvprob': self.rv_problem}
        with open(os.path.join(folder_name, '{}_{}p_likelihood_{}d_{}mps.pkl'.format(
            self.rv_problem.star_name.replace(' ', '_'), len(self.rv_problem.planets_dict), p_inj, k_inj
        )), 'wb') as handle:
            pickle.dump(results_dict, handle)

        print("COMPLETED MAXLIKEFITS AT {}d, {}mps".format(p_inj, k_inj))
        return results_dict


class InjectionRecovery:

    def __init__(self, data=None, p0=None, planets_dict=None, optimizer_type=optnelder.IterativeNelderMead, sampler_type=optsamplers.emceeSampler, scorer_type=pcrvobj.RVPosterior,
                 likelihood_type=pcrvobj.RVLikelihood, kernel_type=None, process_type=pcrvnoise.RVJitter, model_type=pcrvmodels.CompositeRVModel,
                 output_path=None, star_name=None, k_range=(1, 100), p_range=(1.1, 100), k_resolution=20, p_resolution=30, p_shift=0.12345,
                 ecc_inj=0, w_inj=np.pi, tp_inj=None, scaling='log', slurm=False,
                 gaussian_shift=None, gaussian_unc=None, uniform_shift=None, jeffreys_shift=None,
                 **internal_kwargs):
        """
        A class for running injection and recovery tests on a kernel.
        Args:
            data: pychell.orbits.RVData / CompositeRVData, the dataset to perform injections on.  Default is None.
            p0: optimize.knowledge.Parameters, the initial prior probability distributions of the model in question.  Default is None.
            planets_dict: dict, a dictionary containing information on planet labels and orbit bases, should be in the format:
                {[numerical label]: {"label": [alphabetical label], "basis": [pychell.orbits.AbstractOrbitBasis]}
                i.e. planets_dict[1] = {"label": "b", "basis": pco.TCObritBasis(1)}
                Default is None.
            optimizer_type: optimize.optimizers.Optimizer / Minimizer, the model optimizer class to be tested.  Default is
                IterativeNelderMead.
            sampler_type: optimize.samplers.Sampler, the MCMC sampler type to be tested.  Default is emceeSampler (AffInv).
            scorer_type: optimize.objectives.Posterior, the posterior distribution type.  Default is RVPosterior.
            likelihood_type: optimize.objectives.Likelihood, the likelihood type.  Default is RVLikelihood.
            kernel_type: optimize.kernels.NoiseKernel, the type of kernel to be tested.  Default is None.
            process_type: optimize.noise.NoiseProcess, the type of process to be tested.  Default is pychell.orbits.rvnoise.RVJitter.
            model_type: optimize.models.Model, the type of model to be tested.  Default is CompositeRVModel.
            output_path: str, the output directory for all files, in which subdirectories will be made.  Default is the current
                directory.
            star_name: str, the name of the star being tested.  Default is 'Star'.
            k_range: tuple, the range of semiamplitudes to be tested.  Default is (1, 100).
            p_range: tuple, the range of periods to be tested.  Default is (1.1, 100).
            k_resolution: int, the number of semiamplitudes within k_range to be tested.  Default is 20.
            p_resolution: int, the number of periods within p_range to be tested.  Default is 30.
            p_shift: float, arbitrary offset to the periods to prevent clean integer values.  Default is 0.12345.
            ecc_inj: float, the eccentricity of orbits to be injected.  Default is 0.
            w_inj: float, the argument of periastron for orbits to be injected.  Default is pi.
            tp_inj: float, the time of periastron for the orbits to be injected.  Defaults to the median of data timestamps with
                some random offset.
            scaling: str, 'log' for logarithmic or 'lin' for linear scaling in k and p.
            slurm: bool, True if one is using slurm and wishes to sort runs using a job array task ID.
            gaussian_shift: dict, keys: {"per", "k", "tc"}, option to shift the mean value of gaussian priors placed on the injected P, K, and TC away
                from said injected value.  Only applies if there are guassian priors at all.
            gaussian_unc: dict, keys: {"per", "k", "tc"}, option to scale the uncertainty of gaussian priors placed on the injected P, K, and TC by
                this amount.  Only applies if there are gaussian priors at all.
            uniform_shift: dict, keys: {"per", "k", "tc"}, option to shift the lower and upper bounds of uniform priors placed on the
                injected P, K, and TC by these amounts.  For P and TC, these values are scaled by the injected period.
            jeffreys_shift: dict, keys: {"per", "k", "tc"}, identical to uniform_shift, but for Jeffreys priors.
            **internal_kwargs: arguments for creating an already-finished injection/recovery run via a previously saved
                pickle file or some other dictionary.  Allows for definitions of all internal parameters for plotting and such.
        """
        self.data = data
        self.p0 = p0
        self.optimizer_type = optimizer_type
        self.sampler_type = sampler_type
        self.scorer_type = scorer_type
        self.likelihood_type = likelihood_type
        self.kernel_type = kernel_type
        self.process_type = process_type
        self.model_type = model_type
        self.planets_dict = planets_dict
        self.gaussian_shift = gaussian_shift
        self.gaussian_unc = gaussian_unc
        self.uniform_shift = uniform_shift
        self.jeffreys_shift = jeffreys_shift
        self.star_name = star_name if star_name else 'Star'
        array_func = np.geomspace if scaling == 'log' else np.linspace if scaling == 'lin' else None
        if not array_func:
            raise ValueError("scaling must be either \'lin\' or \'log\'")

        self.ks = array_func(k_range[0], k_range[1], k_resolution)
        self.pers = array_func(p_range[0], p_range[1], p_resolution) + p_shift
        self.kp_array = list(itertools.product(self.ks, self.pers))
        for obj in (ecc_inj, w_inj, tp_inj):
            assert (type(obj) in (int, float)) or (obj is None), "Currently only one eccentricity and angle of periastron is supported!"
        self.ecc = ecc_inj
        self.w = w_inj
        if not tp_inj and data:
            tp_inj = np.float(np.nanmedian(self.data.get_vec('t'))) + np.pi/1000
        if tp_inj:
            assert (type(tp_inj) in (int, float)), "currently only one TP for all injections is supported"
        self.tp = tp_inj

        # Alias output path
        if output_path is None:
            output_path = os.path.abspath(os.path.dirname(__file__))
        self.path = output_path

        self.slurm = slurm
        # Define instance attributes to empty dictionaries or from a previous pickle file
        self.fruntype = self.gptype = self.gpunctype = self.delta_lnL = self.delta_aicc = self.periods = self.semiamps = None

        if internal_kwargs:
            self.kbfrac = internal_kwargs['kbfrac']
            self.kbfrac_unc = internal_kwargs['kbfrac_unc']
            self.kb_rec = internal_kwargs['kb_rec']
            self.kbfrac_unc_rec = internal_kwargs['kbfrac_unc_rec']
            self.pars_sorted = internal_kwargs['pars_sorted']
            self.gp_sorted = internal_kwargs['gp_sorted']
            self.gp_unc_sorted = internal_kwargs['gp_unc_sorted']
            self.pars_like = internal_kwargs['pars_like']
            self.lnL = internal_kwargs['lnL']
            self.aicc = internal_kwargs['aicc']
            self.delta_lnL = internal_kwargs['delta_lnL']
            self.delta_aicc = internal_kwargs['delta_aicc']
            self.periods = internal_kwargs['periods']
            self.semiamps = internal_kwargs['semiamps']
            self.full_run_data, self.maxlike_results, self.maxlike_priors, self.priors, self.gp, self.gp_unc = ({} for _ in range(6))
        else:
            self.priors, self.full_run_data, self.gp, self.gp_unc, self.kbfrac, self.kbfrac_unc, self.kbfrac_unc_rec, \
            self.kb_rec, self.pars_sorted, self.gp_sorted, self.gp_unc_sorted, self.maxlike_results, self.maxlike_priors,\
            self.pars_like, self.lnL, self.aicc = ({} for _ in range(16))

    def inject_signal(self, k_inj, p_inj, tp_inj, folder_name):
        """
        Inject a signal into a copy of the data, and save it to a file.
        Args:
            k_inj: float, injected semiamplitude
            p_inj: float, injected period
            tp_inj: float, injected time of periastron
            folder_name: str, the folder to save the new RV file to.

        Returns:
            data_mod: pychell.orbits.RVData / CompositeRVData, the injected data.
        """
        # New data
        data_mod = copy.deepcopy(self.data)

        # Inject signal into the data
        for _data in data_mod.values():
            _data.y += pcrvmodels.KeplerianRVModel.planet_signal(_data.t, p_inj, tp_inj, self.ecc, self.w, k_inj)

        # Write injected RVs and diff RVs to radvel files
        data_mod.to_radvel_file(os.path.join(folder_name, '{}_{}_injected_{}d_{}mps.txt'.format(
            self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), p_inj, k_inj
        )))
        return data_mod

    def fix_priors(self, pars, injected_planet, key, value0, value1):
        """
        Edit the priors of the injected P, K, or TC to reflect the new injected values.
        Args:
            pars: pco.Parameters, the Parameters object to be edited.
            injected_planet: int, the index of the injected planet to be edited.
            key: str, the appropriate value to edit, "per" for period, "k" for semiamplitude, "tc" for time of conjunction
            value0: float, the injected value corresponding to P, K, or TC - midpoint
            value1: float, the value to scale bounds/uncertainty with

        Returns:
            pars: pco.Parameters, the edited Parameters object
        """
        gaussian_priors = [i for (i, prior) in enumerate(pars[key + str(injected_planet)].priors) if isinstance(prior, optknow.priors.Gaussian)]
        uniform_priors = [i for (i, prior) in enumerate(pars[key + str(injected_planet)].priors) if isinstance(prior, optknow.priors.Uniform)]
        jeffreys_priors = [i for (i, prior) in enumerate(pars[key + str(injected_planet)].priors) if isinstance(prior, optknow.priors.JeffreysG)]
        if gaussian_priors and (self.gaussian_shift or self.gaussian_unc):
            for index in gaussian_priors:
                if self.gaussian_shift and key in self.gaussian_shift.keys():
                    pars[key + str(injected_planet)].priors[index].mu = value0 + self.gaussian_shift[key]
                else:
                    pars[key + str(injected_planet)].priors[index].mu = value0
                if self.gaussian_unc and key in self.gaussian_unc.keys():
                    pars[key + str(injected_planet)].priors[index].sigma = value1 * self.gaussian_unc[key]
        if uniform_priors and self.uniform_shift and (key in self.uniform_shift.keys()):
            for index in uniform_priors:
                pars[key + str(injected_planet)].priors[index].minval = value0 + self.uniform_shift[key][0] * value1
                pars[key + str(injected_planet)].priors[index].maxval = value0 + self.uniform_shift[key][1] * value1
        if jeffreys_priors and self.jeffreys_shift and (key in self.jeffreys_shift.keys()):
            for index in jeffreys_priors:
                pars[key + str(injected_planet)].priors[index].minval = value0 + self.jeffreys_shift[key][0] * value1
                pars[key + str(injected_planet)].priors[index].maxval = value0 + self.jeffreys_shift[key][1] * value1
        return pars

    def create_rvproblem(self, k_inj, p_inj, tp_inj, folder_name, data, pars, remove_injected_planet=False):
        """
        Creates an RVProblem with the appropriate kernels, models, scorers, likelihoods, posteriors, etc. etc.
        for a single injection.
        Args:
            k_inj: float, injected semiamplitude.
            p_inj: float, injected period.
            folder_name: str, the folder to save outputs for the new RVProblem to.
            data: pychell.orbits.RVData / CompositeRVData, the ALREADY INJECTED data to use in the RV problem.
            pars: pychell.orbits.Parameters, a copy of the initial p0 parameters to use in the new RVProblem.
            remove_injected_planet: bool, True or False.  Used for maxlikelihood fitting to compare with models
                that do not include the injected planet.

        Returns:
            rvprobi: pychell.orbits.RVProblem, the injected RVProblem.
        """
        planets_dict = copy.deepcopy(self.planets_dict)

        # Adjust model parameters to the injected period and semiamplitude
        injected_planet = list(planets_dict.keys())[-1]
        pars["per" + str(injected_planet)].value = p_inj
        if pars["per" + str(injected_planet)].vary is False:
            print("WARNING: Injected planet's P prior was set to not vary.")
        else:
            pars = self.fix_priors(pars, injected_planet, "per", p_inj, p_inj)
        pars["k" + str(injected_planet)].value = k_inj
        if pars["k" + str(injected_planet)].vary is False:
            print("WARNING: Injected planet's K prior was set to not vary.")
        else:
            pars = self.fix_priors(pars, injected_planet, "k", k_inj, k_inj)

        tc_inj = planetmath.tp_to_tc(tp_inj, p_inj, self.ecc, self.w)
        pars["tc" + str(injected_planet)].value = tc_inj
        if pars["tc" + str(injected_planet)].vary is False:
            print("WARNING: Injected planet's TC prior was set to not vary.")
        else:
            pars = self.fix_priors(pars, injected_planet, "tc", pars["tc" + str(injected_planet)].value, p_inj)
        pars["ecc" + str(injected_planet)].value = self.ecc
        pars["w" + str(injected_planet)].value = self.w

        if remove_injected_planet:
            for key in pars.copy().keys():
                if str(injected_planet) in key:
                    pars.pop(key)
            planets_dict.pop(injected_planet)

        # Create kernels, models, scorers, and optimizers
        par_names_gp = [pname for pname in pars.keys() if "gp" in pname]
        if par_names_gp and self.kernel_type:
            kernel = self.kernel_type(data=data, par_names=par_names_gp)
            process = self.process_type(data=data, kernel=kernel, name="RVs")
        else:
            process = self.process_type(data=data)
        model = self.model_type(planets_dict=planets_dict, data=data, noise_process=process)
        scorer = self.scorer_type()
        scorer["rvs"] = self.likelihood_type(model=model)
        optimizer = self.optimizer_type()
        sampler = self.sampler_type()

        # Create the RVProblem
        rvprobi = pcrvprob.RVProblem(output_path=folder_name, star_name=self.star_name, p0=pars, optimizer=optimizer,
                                     sampler=sampler, post=scorer)
        return rvprobi

    def injection_mcmc(self, k_inj, p_inj, tp_inj, folder_name=None, injection=True, *args, **kwargs):
        """
        Runs a single injection/recovery MCMC and outputs to a sub-folder in the output directory.

        Args:
            k_inj: Injected semiamplitude
            p_inj: Injected period
            tp_inj: Injected time of periastron
            folder_name: Name of folder to save the MCMC output to.  Defaults to an appropriate name.
            injection: bool.  Whether or not to actually inject the planet.
            *args: Arguments for the MCMC function
            **kwargs: Keyword arguments for the MCMC function

        Returns:
            results_dict: dict. Contains MCMC results and priors.
        """
        # Set up output folder
        if not folder_name:
            folder_name = os.path.join(self.path, '{}injection_run_{:.5f}d_{:.5f}mps'.format('non' if not injection else '', p_inj, k_inj))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Create new RVProblem with injected data so as not to keep adding injected data in each run
        if injection:
            data = self.inject_signal(k_inj, p_inj, tp_inj, folder_name)
        else:
            data = copy.deepcopy(self.data)
        pars = copy.deepcopy(self.p0)

        rvprobi = self.create_rvproblem(k_inj, p_inj, tp_inj, folder_name + os.sep, data, pars)

        # Create SoloInjectionRun and do the MCMC
        solo_run = SoloInjectionRecovery(rv_problem=rvprobi, output_path=self.path, star_name=self.star_name)
        results_dict = solo_run.injection_mcmc(folder_name=folder_name, injection=injection, *args, **kwargs)

        return results_dict

    def injection_maxlikelihood(self, k_inj, p_inj, tp_inj, folder_names=None, *args, **kwargs):
        """
        Runs two maximum likelihood fits for the injected planet, using a model with and without the planet.
        Args:
            k_inj: float, injected semiamplitude.
            p_inj: float, injected period.
            tp_inj: float, injected time of periastron.
            folder_names: list / tuple, the names of the two folders to save outputs to.  Defaults to an appropriate name.
            *args: arguments for the optimize function.
            **kwargs: keyword arguments for the optimize function.

        Returns:
            results_dict [dict], results_dict2 [dict]
            The results for both runs, in dictionary form.
        """
        injected_planet = list(self.planets_dict)[-1]

        # Set up output folder
        if not folder_names:
            folder_names = []
            folder_names.append(os.path.join(self.path, '{}p_likelihood_run_{:.5f}d_{:.5f}mps'.format(len(self.planets_dict), p_inj, k_inj)))
            folder_names.append(os.path.join(self.path, '{}p_likelihood_run_{:.5f}d_{:.5f}mps'.format(len(self.planets_dict)-1, p_inj, k_inj)))
        for fname in folder_names:
            if not os.path.exists(fname):
                os.makedirs(fname)

        data = self.inject_signal(k_inj, p_inj, tp_inj, folder_names[0])
        pars = copy.deepcopy(self.p0)
        rvprobi = self.create_rvproblem(k_inj, p_inj, tp_inj, folder_names[0], data, pars)
        solo_run1 = SoloInjectionRecovery(rv_problem=rvprobi, output_path=self.path, star_name=self.star_name)
        results_dict = solo_run1.injection_maxlikelihood(folder_name=folder_names[0], *args, **kwargs)

        rvprobj = self.create_rvproblem(k_inj, p_inj, tp_inj, folder_names[1], data, pars, remove_injected_planet=True)
        solo_run2 = SoloInjectionRecovery(rv_problem=rvprobj, output_path=self.path, star_name=self.star_name)
        results_dict2 = solo_run2.injection_maxlikelihood(folder_name=folder_names[1], *args, **kwargs)

        return results_dict, results_dict2

    def __full_mcmc_run_parallel(self, njobs=-1, backend=None, injection=True, *args, **kwargs):
        """
        Runs all MCMCs for every combination of P and K using parallel processes with joblib.
        WARNING: VERY COMPUTATIONALLY INTESNIVE AND TIME CONSUMING DEPENDING ON YOUR SETTINGS.
        Args:
            njobs: int, Number of MCMCs to run in parallel.
            backend: str, Parallel backend for joblib.
            injection: bool, whether or not to actually inject the planets.
            *args: MCMC function args.
            **kwargs: MCMC function kwargs.

        Returns:
            None
        """
        Parallel(n_jobs=njobs, backend=backend)(delayed(self.injection_mcmc)(ki, peri, self.tp, injection=injection, *args, **kwargs) for (ki, peri) in
                                               self.kp_array)
        print("ALL DONE WITH ALL {} MCMCs".format(len(self.kp_array)))

    def __full_mcmc_run_jobarray(self, injection=True, *args, **kwargs):
        """
        Runs all MCMCs for every combination of P and K using a slurm jobarray.
        WARNING: VERY COMPUTATIONALLY INTESNIVE AND TIME CONSUMING DEPENDING ON YOUR SETTINGS.
        Args:
            injection: bool, whether or not to actually inject the planets.
            *args: MCMC function args.
            **kwargs: MCMC function kwargs.

        Returns:
            None
        """
        id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        ki, peri = self.kp_array[id - 1:id][0]
        self.injection_mcmc(ki, peri, self.tp, injection=injection, *args, **kwargs)
        print("ALL DONE WITH JOB ARRAY ID {}".format(id))

    def full_mcmc_run(self, *args, **kwargs):
        """
        Runs all MCMCs for every combination of P and K, using either joblib or slurm depending on self.slurm.
        WARNING: VERY COMPUTATIONALLY INTESNIVE AND TIME CONSUMING DEPENDING ON YOUR SETTINGS.
        Args:
            njobs: int, Number of MCMCs to run in parallel.
            backend: str, Parallel backend for joblib.
            injection: bool, whether or not to actually inject the planets.
            *args: MCMC function args.
            **kwargs: MCMC function kwargs.

        Returns:
            None
        """
        if not self.slurm:
            self.__full_mcmc_run_parallel(*args, **kwargs)
        else:
            self.__full_mcmc_run_jobarray(*args, **kwargs)

    def __full_maxlikefit_run_parallel(self, njobs=-1, backend=None, *args, **kwargs):
        """
        Runs all maxlikelihood fits for every combination of P and K using parallel processes with joblib.
        WARNING: VERY COMPUTATIONALLY INTESNIVE AND TIME CONSUMING.
        Args:
            njobs: Number of maxlikelihood fits to run in parallel.
            backend: Parallel backend for joblib
            *args: MCMC function args.
            **kwargs: MCMC function kwargs.

        Returns:
            None
        """
        Parallel(n_jobs=njobs, backend=backend)(delayed(self.injection_maxlikelihood)(ki, peri, self.tp, *args, **kwargs) for (ki, peri) in
                                               self.kp_array)
        print("ALL DONE WITH ALL {} MAXLIKEFITS".format(len(self.kp_array)))

    def __full_maxlikefit_run_jobarray(self,  *args, **kwargs):
        """
        Runs all maxlikelihood fits for every combination of P and K using a slurm jobarray
        WARNING: VERY COMPUTATIONALLY INTESNIVE AND TIME CONSUMING.
        Args:
            njobs: Number of maxlikelihood fits to run in parallel.
            backend: Parallel backend for joblib
            *args: MCMC function args.
            **kwargs: MCMC function kwargs.

        Returns:
            None
        """
        id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        ki, peri = self.kp_array[id - 1:id][0]
        self.injection_maxlikelihood(ki, peri, self.tp, *args, **kwargs)
        print("ALL DONE WITH JOB ARRAY ID {}".format(id))

    def full_maxlikefit_run(self, *args, **kwargs):
        """
        Runs all maxlikelihood fits for every combination of P and K, using either joblib or slurm depending on self.slurm.
        WARNING: VERY COMPUTATIONALLY INTESNIVE AND TIME CONSUMING DEPENDING ON YOUR SETTINGS.
        Args:
            njobs: int, Number of MCMCs to run in parallel.
            backend: str, Parallel backend for joblib.
            *args: MCMC function args.
            **kwargs: MCMC function kwargs.

        Returns:
            None
        """
        if not self.slurm:
            self.__full_maxlikefit_run_parallel(*args, **kwargs)
        else:
            self.__full_maxlikefit_run_jobarray(*args, **kwargs)

    def gather_injection_data(self, injection=True):
        """
        Gathers the saved injection/recovery data from pickle files after performing a full MCMC run.
        Args:
            injection: bool, True to gather injected data and False to gather noninjected data.

        Returns:
            priors, full_run_data, gp, gp_unc [dicts]
        """
        key = 'injection' if injection else 'noninjection'
        if key in list(self.full_run_data.keys()):
            return self.priors[key], self.full_run_data[key], self.gp[key], self.gp_unc[key]
        pickles = glob.glob(os.path.join(self.path, '**', '{}_{}injected_*d_*mps.pkl'.format(
            self.star_name.replace(' ', '_'), 'non' if not injection else '')), recursive=True)

        # Get just the first result
        result0 = self.load_data(pickles[0])
        priors0, frun_data0, frun_gp0, frun_gpunc0 = self.get_data(result0)

        # Define datatypes from the first result
        self.fruntype = [('pb_in', float), ('kb_in', float), ('pb_out', float), ('kb_out', float), ('pb_unc', object), ('kb_unc', object), ('lnL', float)]
        self.gptype = [(gp_key, float) for gp_key in frun_gp0.keys()]
        self.gpunctype = [(gp_key, object) for gp_key in frun_gpunc0.keys()]

        key = 'injection' if injection else 'noninjection'
        # Define structured arrays to store data
        self.priors[key] = np.full(shape=(len(pickles),), dtype=object, fill_value=np.nan)
        self.full_run_data[key] = np.full(shape=(len(pickles),), dtype=self.fruntype, fill_value=np.nan)
        self.gp[key] = np.full(shape=(len(pickles),), dtype=self.gptype, fill_value=np.nan)
        self.gp_unc[key] = np.full(shape=(len(pickles),), dtype=self.gpunctype, fill_value=np.nan)

        # Append the first results to the arrays
        self.priors[key][0] = priors0
        self.full_run_data[key][0] = frun_data0
        self.gp[key][0] = tuple(frun_gp0.values())
        self.gp_unc[key][0] = tuple(frun_gpunc0.values())

        # Iterate through the rest of the data and append accordingly
        for i, pkl in enumerate(pickles[1:len(pickles)]):
            result = self.load_data(pkl)
            self.priors[key][i+1], self.full_run_data[key][i+1], frun_gpi, frun_gpunci = self.get_data(result)
            self.gp[key][i+1] = tuple(frun_gpi.values())
            self.gp_unc[key][i+1] = tuple(frun_gpunci)

        return self.priors[key], self.full_run_data[key], self.gp[key], self.gp_unc[key]

    def gather_likelihood_data(self):
        """
        Gathers the saved likelihood data from pickle files after performing a full maximum likelihood run.

        Returns:
            maxlike_results (all planets), maxlike_results (all planets - 1) [dicts]
        """
        if 'high' in list(self.maxlike_results.keys()) and 'low' in list(self.maxlike_results.keys()):
            return self.maxlike_results['high'], self.maxlike_results['low']
        pickles_Xp = glob.glob(os.path.join(self.path, '**', '{}_{}p_likelihood_*d_*mps.pkl'.format(self.star_name.replace(' ', '_'),
                                                                                                    len(self.planets_dict))),
                               recursive=True)
        pickles_Yp = glob.glob(os.path.join(self.path, '**', '{}_{}p_likelihood_*d_*mps.pkl'.format(self.star_name.replace(' ', '_'),
                                                                                                    len(self.planets_dict) - 1)),
                               recursive=True)

        lnLtype = [('lnL', float), ('per', float), ('k', float)]
        self.maxlike_results['high'] = np.full(shape=(len(pickles_Xp),), dtype=lnLtype, fill_value=np.nan)
        self.maxlike_results['low'] = np.full(shape=(len(pickles_Yp),), dtype=lnLtype, fill_value=np.nan)
        self.maxlike_priors['high'] = np.full(shape=(len(pickles_Xp),), dtype=object, fill_value=np.nan)
        self.maxlike_priors['low'] = np.full(shape=(len(pickles_Yp),), dtype=object, fill_value=np.nan)

        for pickly in [pickles_Xp, pickles_Yp]:
            for i, pkl in enumerate(pickly):
                f = self.load_data(pkl)
                a = 1 if pickly == pickles_Yp else 0
                lnL = f['rvprob'].post.compute_logL(f['rvprob'].p0)
                per = f['rvprob'].p0['per' + str(len(self.planets_dict) - a)].value
                k = f['rvprob'].p0['k' + str(len(self.planets_dict) - a)].value
                if pickly == pickles_Xp:
                    self.maxlike_results['high'][i] = tuple((lnL, per, k))
                    self.maxlike_priors['high'][i] = f['rvprob'].p0
                elif pickly == pickles_Yp:
                    self.maxlike_results['low'][i] = tuple((lnL, per, k))
                    self.maxlike_priors['low'][i] = f['rvprob'].p0

        return self.maxlike_results['high'], self.maxlike_results['low']

    def organize_injection_data(self, save=True, injection=True):
        """
        Organize the collected MCMC and maxlikelihood data into 2D arrays in preparation to be plotted on a 2D histogram.
        Args:
            save: bool, True to save data to pickle & json files.
            injection: True to organize injected data, False to organize noninjected data.

        Returns:
            kbfrac: dict[ndarray], recovered semiamplitude / injected semiamplitude
            kbfrac_unc: dict[ndarray], recovered semiamplitude uncertainty / injected semiamplitude
            kb_rec: dict[ndarray], recovered semiamplitude
            kbfrac_unc_rec: dict[ndarray], recovered semiamplitude uncertainty / recovered semiamplitude
            gp_sorted: dict[ndarray], posterior GP parameters
            gp_unc_sorted: dict[ndarray], posterior GP uncertainties
            delta_lnL: ndarray, log likelihood (all planets) - log likelihood (all planets - 1)
            delta_aicc: ndarray, AiCc (all planets) - AiCc (all planets - 1)
        """
        self.gather_likelihood_data()
        self.gather_injection_data(injection=injection)
        key = 'injection' if injection else 'noninjection'
        if key in list(self.kbfrac.keys()):
            return self.kbfrac[key], self.kbfrac_unc[key], self.kb_rec[key], self.kbfrac_unc_rec[key], self.gp_sorted[key], self.gp_unc_sorted[key], \
            self.delta_lnL, self.delta_aicc
        do_likes = False if 'high' in list(self.lnL.keys()) else True

        self.periods = np.unique(self.full_run_data[key]['pb_in'])
        self.semiamps = np.unique(self.full_run_data[key]['kb_in'])

        # Alias
        pb = self.periods
        kbin = self.semiamps

        n = len(self.data.get_vec('y'))
        self.kbfrac[key] = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.kbfrac_unc[key] = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.kbfrac_unc_rec[key] = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.kb_rec[key] = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.pars_sorted[key] = np.full(shape=(len(kbin), len(pb)), dtype=object, fill_value=np.nan)
        self.gp_sorted[key] = np.full(shape=(len(kbin), len(pb)), dtype=self.gptype, fill_value=np.nan)
        self.gp_unc_sorted[key] = np.full(shape=(len(kbin), len(pb)), dtype=self.gpunctype, fill_value=np.nan)

        if do_likes:
            self.lnL['high'] = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
            self.lnL['low'] = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
            self.delta_lnL = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
            self.pars_like['high'] = np.full(shape=(len(kbin), len(pb)), dtype=object, fill_value=np.nan)
            self.pars_like['low'] = np.full(shape=(len(kbin), len(pb)), dtype=object, fill_value=np.nan)
            self.aicc['high'] = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
            self.aicc['low'] = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
            self.delta_aicc = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)

        for x in range(len(pb)):
            for y in range(len(kbin)):
                c = np.where(np.isclose(self.full_run_data[key]['pb_in'], pb[x]) & np.isclose(self.full_run_data[key]['kb_in'], kbin[y]))[0]
                eh = np.where(np.isclose(self.maxlike_results['high']['per'], pb[x]) & np.isclose(self.maxlike_results['high']['k'], kbin[y]))[0]
                # This won't work because the L maxlike results don't have an injected planet period/semiamp
                # el = np.where(np.isclose(self.maxlike_results_L['per'], pb[x]) & np.isclose(self.maxlike_results_L['k'], kbin[y]))[0]
                el = eh
                if c.size:
                    c = c[0]
                    self.kbfrac[key][y, x] = float(self.full_run_data[key]['kb_out'][c] / self.full_run_data[key]['kb_in'][c])
                    self.kbfrac_unc[key][y, x] = float(np.mean(self.full_run_data[key]['kb_unc'][c][0]) / self.full_run_data[key]['kb_in'][c])
                    self.kbfrac_unc_rec[key][y, x] = float(np.mean(self.full_run_data[key]['kb_unc'][c][0]) / self.full_run_data[key]['kb_out'][c])
                    self.kb_rec[key][y, x] = float(self.full_run_data[key]['kb_out'][c])
                    self.pars_sorted[key][y, x] = self.priors[key][c]
                    self.gp_sorted[key][y, x] = self.gp[key][c]
                    self.gp_unc_sorted[key][y, x] = self.gp_unc[key][c]
                if eh.size and do_likes:
                    eh = eh[0]
                    self.lnL['high'][y, x] = float(np.ma.masked_invalid(self.maxlike_results['high']['lnL'][eh]))
                    self.pars_like['high'][y, x] = self.maxlike_priors['high'][eh]
                if el.size and do_likes:
                    el = el[0]
                    self.lnL['low'][y, x] = float(np.ma.masked_invalid(self.maxlike_results['low']['lnL'][el]))
                    self.pars_like['low'][y, x] = self.maxlike_priors['low'][el]
                if eh.size and el.size and do_likes:
                    self.delta_lnL[y, x] = self.lnL['high'][y, x] - self.lnL['low'][y, x]
                if c.size and eh.size and el.size and do_likes:
                    kh = self.pars_like['high'][y, x].num_varied
                    kl = self.pars_like['low'][y, x].num_varied
                    self.aicc['high'][y, x] = self.get_aicc(kh, self.lnL['high'][y, x], n)
                    self.aicc['low'][y, x] = self.get_aicc(kl, self.lnL['low'][y, x], n)
                    self.delta_aicc[y, x] = self.aicc['high'][y, x] - self.aicc['low'][y, x]

        if save:
            savedict = {'kbfrac': self.kbfrac, 'kbfrac_unc': self.kbfrac_unc, 'kb_rec': self.kb_rec, 'kbfrac_unc_rec': self.kbfrac_unc_rec,
                        'lnL': self.lnL, 'delta_lnL': self.delta_lnL, 'pars_sorted': self.pars_sorted, 'pars_like': self.pars_like,
                        'aicc': self.aicc, 'delta_aicc': self.delta_aicc, 'gp_sorted': self.gp_sorted,
                        'gp_unc_sorted': self.gp_unc_sorted, 'periods': self.periods, 'semiamps': self.semiamps}
            with open(self.path + os.sep + '{}_organized_data_{}.pkl'.format(self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb') as handle:
                pickle.dump(savedict, handle)
            savedict_list = {}
            for ki, vi in savedict.items():
                v = {}
                if type(vi) is not dict:
                    savedict_list[ki] = vi.tolist()
                    continue
                if 'pars' not in ki:
                    for kii in ('injection', 'noninjection', 'high', 'low'):
                        if kii in list(vi.keys()):
                            v[kii] = vi[kii].tolist()
                else:
                    for kii in ('injection', 'noninjection', 'high', 'low'):
                        if kii in list(vi.keys()):
                            v[kii] = [vii.__repr__() for vii in vi[kii].ravel().tolist()]
                savedict_list[ki] = v
            serialized = json.dumps(savedict_list, indent=4, sort_keys=True)
            with open(self.path + os.sep + '{}_organized_data_{}.json'.format(self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), 'w') as handle:
                handle.write(serialized)

        return self.kbfrac[key], self.kbfrac_unc[key], self.kb_rec[key], self.kbfrac_unc_rec[key], self.gp_sorted[key], self.gp_unc_sorted[key], \
               self.delta_lnL, self.delta_aicc

    @staticmethod
    @njit
    def get_aicc(k, lnL, n):
        """
        Calculated the corrected Akaike injection criterion.
        Args:
            k: int, number of free parameters.
            lnL: float, log likelihood.
            n: int, number of measurements.

        Returns:
            aicc: float, corrected AiC
        """
        aic = -2.0 * lnL + 2.0 * k
        _aicc = aic
        denom = n - k - 1.0
        if denom > 0:
            _aicc += (2.0 * k * (k + 1.0)) / denom
        return _aicc

    @staticmethod
    def load_data(path):
        """
        Loads a pickle file.
        Args:
            path: str, the filepath.

        Returns:
            f: object, the contents of the pickle file.
        """
        f = pickle.load(open(path, 'rb'))
        return f

    def get_data(self, result):
        """
        Recovers the MCMC results from a single pickle file.
        Args:
            result: the loaded pickle data from self.load_data

        Returns:
            priors: pychell.orbits.Parameters, the priors of the MCMC run
            frun_data: tuple, the injected P and K, the recovered P and K, the uncertainty in recovered P and K,
                as well as the lnL from the MCMC.
            frun_gp: dict, the GP posterior parameters from the MCMC.
            frun_gp_unc: dict, the uncertainty in GP posteriors from the MCMC.
        """
        injected_planet = list(self.planets_dict.keys())[-1]
        kbkey = 'k{}'.format(injected_planet)
        pbkey = 'per{}'.format(injected_planet)
        kb_in = result['priors'][kbkey].value
        kb_out = result['sampler_result']['pmed'][kbkey].value
        kb_out_unc = result['sampler_result']['pmed'][kbkey].unc
        pb_in = result['priors'][pbkey].value
        pb_out = result['sampler_result']['pmed'][pbkey].value
        pb_out_unc = result['sampler_result']['pmed'][pbkey].unc
        lnL = result['sampler_result']['lnL']
        priors = result['priors']

        gp_keys = [key for key in result['sampler_result']['pmed'].keys() if 'gp' in key]
        gp_vals = [result['sampler_result']['pmed'][gp_key].value for gp_key in gp_keys]
        gp_uncs = [result['sampler_result']['pmed'][gp_key].unc for gp_key in gp_keys]

        frun_data = tuple((pb_in, kb_in, pb_out, kb_out, pb_out_unc, kb_out_unc, lnL))
        frun_gp = {gp_key: gp_val for gp_key, gp_val in zip(gp_keys, gp_vals)}
        frun_gp_unc = {gp_key: gp_unc for gp_key, gp_unc in zip(gp_keys, gp_uncs)}

        return priors, frun_data, frun_gp, frun_gp_unc

    def plot_injection_2D_hist(self, injection=True, vector=False, bounds_k=None, bounds_unc=None, colormap_k='RdYlGn', colormap_unc='RdYlBu',
                               xticks_k=None, xticks_unc=None, yticks_k=None, yticks_unc=None, plot_style=None, xtickprecision=0, ytickprecision=0):
        """
        Plots the 2D histogram of injection/recovery data.  The upper plot is K recovered / K injected, and the lower
        plot is K injected / sigma K recovered for injection runs and K recovered / sigma K recovered for noninjection runs.
        Args:
            injection: bool, True to plot injection data, False to plot noninjection data.
            vector: bool, True to save as eps, False to save as png.
            bounds_k: list / tuple, colorbar limits for the upper plot.
            bounds_unc: list / tuple, colorbar limits for the lower plot.
            colormap_k: matplotlib.colormap / str, the colormap to use for the upper plot.
            colormap_unc: matplotlib.colormap / str, the colormap to use for the lower plot.
            xticks_k: list / array, the upper plot x ticks.
            xticks_unc: list / array, the lower plot x ticks.
            yticks_k: list / array, the upper plot y ticks.
            yticks_unc: list / array, the lower plot y ticks.
            plot_style: str, the matplotlib stylesheet to use.  Defaults to the pychell gadfly stylesheet.

        Returns:
            None.
        """
        assert self.kbfrac is not None
        key = 'injection' if injection else 'noninjection'
        if plot_style:
            plt.style.use(plot_style)
        fig, (ax1, ax2) = plt.subplots(2)
        # norm = None if planets_model == 1 else colors.LogNorm()
        norm1 = colors.LogNorm()
        norm2 = colors.LogNorm()
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        cbar_ax = fig.add_axes([0.83, pos1.y0, 0.025, pos1.height])
        cbar2_ax = fig.add_axes([0.83, pos2.y0, 0.025, pos2.height])
        if not bounds_k:
            bounds_k = (np.nanmin(self.kbfrac[key]), np.nanmax(self.kbfrac[key]))
        plot = ax1.pcolormesh(self.periods, self.semiamps, self.kbfrac[key], cmap=colormap_k, vmin=bounds_k[0], vmax=bounds_k[1], norm=norm1,
                              shading='nearest')
        cb = fig.colorbar(plot, cax=cbar_ax, label='$K_{\\mathrm{recovered}}\\ /\\ K_{\\mathrm{injected}}$')

        unc = 1/self.kbfrac_unc_rec[key]
        if not bounds_unc:
            bounds_unc = (np.nanmin(unc), np.nanmax(unc))
        plot_b = ax2.pcolormesh(self.periods, self.semiamps, unc, cmap=colormap_unc, vmin=bounds_unc[0], vmax=bounds_unc[1], norm=norm2,
                                shading='nearest')
        istr = 'recovered'
        cb2 = fig.colorbar(plot_b, cax=cbar2_ax, label='$K_{\\mathrm{%s}}\\ /\\ \\sigma_{K}$' % istr)
        ax2.set_xlabel('Injected Period [days]')
        fig.text(0.02, 0.5, 'Injected Semiamplitude [m s$^{-1}$]', va='center', rotation='vertical')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax1.minorticks_off()
        ax2.minorticks_off()
        if not xticks_k:
            xticks_k = self.periods[::len(self.periods) // 40 + 2]
        if not xticks_unc:
            xticks_unc = self.periods[::len(self.periods) // 40 + 2]
        if not yticks_k:
            yticks_k = self.semiamps[::len(self.semiamps) // 20 + 1]
        if not yticks_unc:
            yticks_unc = self.semiamps[::len(self.semiamps) // 20 + 1]
        ax1.set_xticks(xticks_k)
        ax1.set_yticks(yticks_k)
        ax2.set_xticks(xticks_unc)
        ax2.set_yticks(yticks_unc)
        fmtstrx = '{:.' + str(xtickprecision) + 'f}'
        fmtstry = '{:.' + str(ytickprecision) + 'f}'
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: fmtstrx.format(y)))
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: fmtstry.format(y)))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: fmtstrx.format(y)))
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: fmtstry.format(y)))
        ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
        fig.subplots_adjust(right=0.8, hspace=0.3)
        ftype = 'png' if not vector else 'eps'
        plt.savefig(self.path + os.sep + '{}injection_recovery_histograms_{}_{}.{}'.format('non' if not injection else '',
                                                                       self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                                                                      ftype), bbox_extra_artist=(cb, cb2), edges='tight', dpi=300)
        plt.close()

    def plot_delta_aicc(self, vector=False, vmin=-10**6, vmax=10**6, linthresh=1, colormap=None, xticks=None, yticks=None,
                        plot_style=None, xtickprecision=0, ytickprecision=0):
        """
        Plots the 2D histogram of delta AiCc data.  Uses a symmetric logarithmic colorbar scale.
        Args:
            vector: bool, True to save as eps, False to save as png.
            vmin: float, colorbar minimum limit, must be negative.
            vmax: float, colorbar maximum limit, must be positive.
            linthresh: float, the threshold at which the colorbar becomes linear around 0.
            colormap: matplotlib.colormap / str, the colormap to use.
            xticks: list / array, the x ticks.
            yticks: list / array, the y ticks.
            plot_style: str, the plot style to use.  Defaults to pychell gadfly stylesheet.

        Returns:
            None
        """
        if not colormap:
            colors_below0 = plt.cm.PuOr(np.linspace(1, 0.5, 256))
            colors_above0 = plt.cm.RdYlGn(np.linspace(0.5, 1, 256))
            all_colors = np.vstack((colors_below0, colors_above0))
            colormap = colors.LinearSegmentedColormap.from_list('red-purple', all_colors)

        assert self.delta_aicc is not None
        if plot_style:
            plt.style.use(plot_style)
        fig, ax = plt.subplots()
        lognorm = colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)
        plot = ax.pcolormesh(self.periods, self.semiamps, self.delta_aicc, cmap=colormap, norm=lognorm, rasterized=True, shading='nearest')
        ax.set_xlabel('Injected Period [days]')
        ax.set_ylabel('Injected Semiamplitude [m s$^{-1}$]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.minorticks_off()
        if not xticks:
            xticks = self.periods[::len(self.periods) // 40 + 2]
        if not yticks:
            yticks = self.semiamps[::len(self.semiamps) // 20 + 1]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        fmtstrx = '{:.' + str(xtickprecision) + 'f}'
        fmtstry = '{:.' + str(ytickprecision) + 'f}'
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: fmtstrx.format(y)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: fmtstry.format(y)))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        cbar = fig.colorbar(plot, ax=ax, label='$\\Delta$ AICc')
        cbar.set_ticks(np.concatenate((-np.geomspace(np.abs(vmin), linthresh, int(np.log10(np.abs(vmin))+1)), [0],
                                       np.geomspace(linthresh, np.abs(vmax), int(np.log10(np.abs(vmax))+1)))))
        ftype = 'png' if not vector else 'eps'
        plt.savefig(self.path + os.sep + 'delta_aicc_histograms_{}_{}.{}'.format(self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), ftype),
                    dpi=300, edges='tight')
        plt.close()

    def plot_1d_histograms(self, injection=True, weightskb=None, weightsgp=None, vector=False, cutoff=1e5, bins=100,
                           plot_style=None):
        ftype = 'png' if not vector else 'eps'
        key = 'injection' if injection else 'noninjection'

        if plot_style:
            plt.style.use(plot_style)
        fig, ax = plt.subplots()
        kbfrac_unc_flat = 1 / self.kbfrac_unc_rec[key].ravel()
        kbfrac_unc_flat = kbfrac_unc_flat[np.where(np.isfinite(kbfrac_unc_flat) & (kbfrac_unc_flat < cutoff))]
        ax.hist(kbfrac_unc_flat, bins=100, weights=weightskb, density=True)
        ax.set_yscale('log')
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('$K_{\\mathrm{recovered}}\\ /\\ \\sigma_{K}$')
        plt.savefig(self.path + os.sep + '1d_injection_histogram_kunc_{}_{}.{}'.format(
            self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), ftype),
                    edges='tight')
        plt.close()

        for field in self.gp_sorted[key].dtype.names:
            fig, ax = plt.subplots()
            gp_param = self.gp_sorted[key][field].ravel()
            gp_param = gp_param[np.where(np.isfinite(gp_param) & (np.abs(gp_param) < cutoff))]
            ax.hist(gp_param, bins=bins, weights=weightsgp, density=True)
            ax.set_xlabel(field)
            ax.set_yscale('log')
            ax.set_ylabel('{}weighted Probability Density'.format('un' if not weightsgp else ''))
            plt.savefig(self.path + os.sep + '1d_{}injection_histogram_{}_{}_{}.{}'.format('non' if not injection else '', field,
            self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), ftype),
                edges='tight')
            plt.close()

    @classmethod
    def from_pickle(cls, filepath, *args, **kwargs):
        """
        Create an InjectionRecovery object from an already-saved pickle file.  Must be of the correct format.
        Args:
            filepath: str, the path to the pickle file.
            *args: other arguments to give to the InjectionRecovery object.
            **kwargs: other keyword arguments to give to the InjectionRecovery object.

        Returns:
            InjectionRecovery object.
        """
        data = pickle.load(open(filepath, 'rb'))
        path = os.path.abspath(os.path.dirname(filepath))
        required_keys = ('kbfrac', 'kbfrac_unc', 'kb_rec', 'kbfrac_unc_rec', 'pars_sorted', 'gp_sorted', 'gp_unc_sorted',
                         'lnL', 'delta_lnL', 'aicc', 'delta_aicc', 'pars_like', 'periods', 'semiamps')
        for key in required_keys:
            assert key in data.keys(), "file missing key: {}".format(key)
        return cls(*args, output_path=path, **kwargs, **data)
