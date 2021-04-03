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

import pychell.orbits as pco


class InjectionRecovery:

    def __init__(self, data, p0, planets_dict, optimizer_type=pco.NelderMead, sampler_type=pco.AffInv, scorer_type=pco.RVPosterior,
                 likelihood_type=pco.RVLikelihood, kernel_type=pco.WhiteNoise, model_type=pco.RVModel,
                 output_path=None, star_name=None, k_range=(1, 100), p_range=(1.1, 100), k_resolution=20, p_resolution=30, p_shift=0.12345,
                 ecc_inj=0, w_inj=np.pi, tp_inj=None, scaling='log', slurm=False):
        """
        A class for running injection and recovery tests on a specific kernel.

        Args:
            data:
            p0:
            planets_dict:
            optimizer_type:
            sampler_type:
            scorer_type:
            likelihood_type:
            kernel_type:
            model_type:
            output_path:
            star_name:
            k_range:
            p_range:
            k_resolution:
            p_resolution:
            p_shift:
            ecc_inj:
            w_inj:
            tp_inj:
            scaling:
            slurm:
        """
        self.data = data
        self.p0 = p0
        self.optimizer_type = optimizer_type
        self.sampler_type = sampler_type
        self.scorer_type = scorer_type
        self.likelihood_type = likelihood_type
        self.kernel_type = kernel_type
        self.model_type = model_type
        self.planets_dict = planets_dict
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
        if not tp_inj:
            tp_inj = np.array([np.float(np.nanmedian(self.data.get_vec('t'))) * (np.random.rand() - 0.5) * p_inj for k_inj, p_inj in self.kp_array])
        assert (type(tp_inj) in (list, tuple, np.ndarray)) and (len(tp_inj) == len(self.kp_array)), "If specified, a unique Tp must be given for each combination of k and p"
        self.tp = tp_inj

        # Alias output path
        if output_path is None:
            output_path = os.path.abspath(os.path.dirname(__name__))
        self.path = output_path

        self.slurm = slurm
        # Define instance attributes to not be used until later
        self.priors = self.full_run_data = self.gp = self.gp_unc = self.maxlike_results_H = self.maxlike_results_L = \
            self.maxlike_priors_H = self.maxlike_priors_L = self.fruntype = self.gptype = self.gpunctype = \
            self.kbfrac = self.kbfrac_unc = self.kb_rec = self.kbfrac_unc_rec = self.pars_sorted = self.pars_likeH = \
            self.pars_likeL = self.lnLH = self.lnLL = self.delta_lnL = self.aiccH = self.aiccL = self.delta_aicc = \
            self.gp_sorted = self.gp_unc_sorted = self.periods = self.semiamps = None

    def inject_signal(self, k_inj, p_inj, tp_inj, folder_name):
        # New data
        data_mod = copy.deepcopy(self.data)

        # Inject signal into the data
        for _data in data_mod.values():
            _data.y += pco.pcrvmodels.planet_signal(_data.t, p_inj, tp_inj, self.ecc, self.w, k_inj)

        # Write injected RVs and diff RVs to radvel files
        data_mod.to_radvel_file(os.path.join(folder_name, '{}_{}_injected_{}d_{}mps.txt'.format(
            self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), p_inj, k_inj
        )))
        return data_mod

    def create_rvproblem(self, k_inj, p_inj, folder_name, data, pars, remove_injected_planet=False):
        planets_dict = copy.deepcopy(self.planets_dict)

        # Adjust model parameters to the injected period and semiamplitude
        injected_planet = list(planets_dict.keys())[-1]
        pars["per" + str(injected_planet)].value = p_inj
        if pars["k" + str(injected_planet)].vary == False:
            print("WARNING: Injected planet's K prior was set to not vary.  Resetting to allow variation.")
        pars["k" + str(injected_planet)] = pco.Parameter(value=k_inj, vary=True)
        positive_priors = [prior for prior in pars["k" + str(injected_planet)].priors if isinstance(prior, pco.Positive)]
        if not positive_priors:
            print("WARNING: Injected planet's K prior did not include any positive prior.  Adding a positive prior.")
            pars["k" + str(injected_planet)].add_prior(pco.Positive())
        pars["ecc" + str(injected_planet)].value = self.ecc
        pars["w" + str(injected_planet)].value = self.w

        if remove_injected_planet:
            for key, val in pars.items():
                if str(injected_planet) in key:
                    del pars[key]
            del planets_dict[injected_planet]

        # Create kernels, models, scorers, and optimizers
        par_names_gp = [pname for pname in pars.keys() if "gp" in pname]
        kernel = self.kernel_type(data=data, par_names=par_names_gp)
        model = self.model_type(planets_dict=planets_dict, data=data, p0=pars, kernel=kernel)
        scorer = self.scorer_type()
        scorer["rvs"] = self.likelihood_type(data=data, model=model)
        optimizer = self.optimizer_type(scorer=scorer)
        sampler = self.sampler_type(scorer=scorer, options=None)

        # Create the RVProblem
        rvprobi = pco.RVProblem(output_path=folder_name, star_name=self.star_name, data=data, p0=pars, optimizer=optimizer,
                                sampler=sampler, scorer=scorer)
        return rvprobi

    def injection_mcmc(self, k_inj, p_inj, tp_inj, folder_name=None, injection=True, *args, **kwargs):
        """
        Runs a single injection/recovery MCMC and outputs to a sub-folder in the output directory.

        Args:
            k_inj: Injected semiamplitude
            p_inj: Injected period
            tp_inj: Injected time of periastron
            folder_name: Name of folder to save the MCMC output to
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

        rvprobi = self.create_rvproblem(k_inj, p_inj, folder_name, data, pars)

        # Run the MCMC
        sampler_result = rvprobi.sample(save=False, *args, **kwargs)
        cornerplot = rvprobi.corner_plot(mcmc_result=sampler_result)

        # Save results to a pickle file for later use
        results_dict = {'sampler_result': sampler_result, 'priors': pars}
        with open(os.path.join(folder_name, '{}_{}injected_{}d_{}mps.pkl'.format(
            rvprobi.star_name.replace(' ', '_'), 'non' if not injection else '', p_inj, k_inj
        )), 'wb') as handle:
            pickle.dump(results_dict, handle)

        del sampler_result, cornerplot
        gc.collect()

        print("COMPLETED MCMC AT {}d, {}mps".format(p_inj, k_inj))
        return results_dict

    def injection_maxlikelihood(self, k_inj, p_inj, tp_inj, folder_names=None, *args, **kwargs):
        # Set up output folder
        if not folder_names:
            folder_names = []
            folder_names.append(os.path.join(self.path, '{}p_likelihood_run_{:.5f}d_{:.5f}mps'.format(len(self.planets_dict), p_inj, k_inj)))
        if not os.path.exists(folder_names[0]):
            os.makedirs(folder_names[0])

        # Run twice: once with the injected planet and once without
        data = self.inject_signal(k_inj, p_inj, tp_inj, folder_names[0])
        pars = copy.deepcopy(self.p0)
        rvprobi = self.create_rvproblem(k_inj, p_inj, folder_names[0], data, pars)
        opt_result = rvprobi.maxlikefit(save=False, *args, **kwargs)

        results_dict = {'opt_result': opt_result, 'priors': rvprobi.p0}
        with open(os.path.join(folder_names[0], '{}_{}p_likelihood_{}d_{}mps.pkl'.format(
            rvprobi.star_name.replace(' ', '_'), len(self.planets_dict), p_inj, k_inj
        )), 'wb') as handle:
            pickle.dump(results_dict, handle)

        if len(folder_names) == 1:
            folder_names.append(os.path.join(self.path, '{}p_likelihood_run_{:.5f}d_{:.5f}mps'.format(len(self.planets_dict) - 1, p_inj, k_inj)))
        if not os.path.exists(folder_names[1]):
            os.makedirs(folder_names[1])
        data.to_radvel_file(os.path.join(folder_names[1], '{}_{}_injected_{}d_{}mps.txt'.format(
            self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), p_inj, k_inj
        )))
        rvprobj = self.create_rvproblem(k_inj, p_inj, folder_names[1], data, pars, remove_injected_planet=True)
        opt_result2 = rvprobj.maxlikefit(*args, **kwargs)
        results_dict2 = {'opt_result': opt_result2, 'priors': rvprobj.p0}
        with open(os.path.join(folder_names[1], '{}_{}p_likelihood_{}d_{}mps.pkl'.format(
            rvprobj.star_name.replace(' ', '_'), len(self.planets_dict) - 1, p_inj, k_inj
        )), 'wb') as handle:
            pickle.dump(results_dict2, handle)

        print("COMPLETED MAXLIKEFITS AT {}d, {}mps".format(p_inj, k_inj))

    def __full_mcmc_run_parallel(self, njobs=-1, backend=None, injection=True, *args, **kwargs):
        """
        Runs all MCMCs for every combination of P and K.  WARNING: VERY COMPUTATIONALLY INTESNIVE AND TIME CONSUMING.
        Args:
            njobs: Number of MCMCs to run in parallel.
            backend: Parallel backend for joblib
            *args: MCMC function args.
            **kwargs: MCMC function kwargs.

        Returns:
            None
        """
        Parallel(n_jobs=njobs, backend=backend)(delayed(self.injection_mcmc)(ki, peri, tpi, injection=injection, *args, **kwargs) for (ki, peri), tpi in
                                               zip(self.kp_array, self.tp))
        print("ALL DONE WITH ALL {} MCMCs".format(len(self.kp_array)))

    def __full_mcmc_run_jobarray(self, injection=True, *args, **kwargs):
        id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        ki, peri = self.kp_array[id - 1:id][0]
        tpi = self.tp[id - 1]
        self.injection_mcmc(ki, peri, tpi, injection=injection, *args, **kwargs)
        print("ALL DONE WITH JOB ARRAY ID {}".format(id))

    def full_mcmc_run(self, *args, **kwargs):
        if not self.slurm:
            self.__full_mcmc_run_parallel(*args, **kwargs)
        else:
            self.__full_mcmc_run_jobarray(*args, **kwargs)

    def __full_maxlikefit_run_parallel(self, njobs=-1, backend=None, *args, **kwargs):
        """
        Runs all maxlikelihood fits for every combination of P and K.  WARNING: VERY COMPUTATIONALLY INTESNIVE AND TIME CONSUMING.
        Args:
            njobs: Number of maxlikelihood fits to run in parallel.
            backend: Parallel backend for joblib
            *args: MCMC function args.
            **kwargs: MCMC function kwargs.

        Returns:
            None
        """
        Parallel(n_jobs=njobs, backend=backend)(delayed(self.injection_maxlikelihood)(ki, peri, tpi, *args, **kwargs) for (ki, peri), tpi in
                                               zip(self.kp_array, self.tp))
        print("ALL DONE WITH ALL {} MAXLIKEFITS".format(len(self.kp_array)))

    def __full_maxlikefit_run_jobarray(self,  *args, **kwargs):
        id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        ki, peri = self.kp_array[id - 1:id][0]
        tpi = self.tp[id - 1]
        self.injection_maxlikelihood(ki, peri, tpi, *args, **kwargs)
        print("ALL DONE WITH JOB ARRAY ID {}".format(id))

    def full_maxlikefit_run(self, *args, **kwargs):
        if not self.slurm:
            self.__full_maxlikefit_run_parallel(*args, **kwargs)
        else:
            self.__full_maxlikefit_run_jobarray(*args, **kwargs)

    def gather_injection_data(self, injection=True):
        pickles = glob.glob(os.path.join(self.path, '**', '{}_{}injected_*d_*mps.pkl'.format(
            self.star_name.replace(' ', '_'), 'non' if not injection else '')), recursive=True)

        # Get just the first result
        result0 = self.load_data(pickles[0])
        priors0, frun_data0, frun_gp0, frun_gpunc0 = self.get_data(result0)

        # Define datatypes from the first result
        self.fruntype = [('pb_in', float), ('kb_in', float), ('pb_out', float), ('kb_out', float), ('pb_unc', object), ('kb_unc', object), ('lnL', float)]
        self.gptype = [(gp_key, float) for gp_key in frun_gp0.keys()]
        self.gpunctype = [(gp_key, object) for gp_key in frun_gpunc0.keys()]

        # Define structured arrays to store data
        self.priors = np.full(shape=(len(pickles),), dtype=object, fill_value=np.nan)
        self.full_run_data = np.full(shape=(len(pickles),), dtype=self.fruntype, fill_value=(np.nan,)*len(self.fruntype))
        self.gp = np.full(shape=(len(pickles),), dtype=self.gptype, fill_value=(np.nan,)*len(self.gptype))
        self.gp_unc = np.full(shape=(len(pickles),), dtype=self.gpunctype, fill_value=(np.nan,)*len(self.gpunctype))

        # Append the first results to the arrays
        self.priors[0] = priors0
        self.full_run_data[0] = frun_data0
        self.gp[0] = tuple(frun_gp0.values())
        self.gp_unc[0] = tuple(frun_gpunc0.values())

        # Iterate through the rest of the data and append accordingly
        for i, pkl in enumerate(pickles[1:len(pickles)]):
            result = self.load_data(pkl)
            self.priors[i+1], self.full_run_data[i+1], frun_gpi, frun_gpunci = self.get_data(result)
            self.gp[i+1] = tuple(frun_gpi.values())
            self.gp_unc[i+1] = tuple(frun_gpunci)

        return self.priors, self.full_run_data, self.gp, self.gp_unc

    def gather_likelihood_data(self):
        pickles_Xp = glob.glob(os.path.join(self.path, '**', '{}_{}p_likelihood_*d_*mps.pkl'.format(self.star_name.replace(' ', '_'),
                                                                                                    len(self.planets_dict))),
                               recursive=True)
        pickles_Yp = glob.glob(os.path.join(self.path, '**', '{}_{}p_likelihood_*d_*mps.pkl'.format(self.star_name.replace(' ', '_'),
                                                                                                    len(self.planets_dict) - 1)),
                               recursive=True)

        lnLtype = [('lnL', float), ('per', float), ('k', float)]
        self.maxlike_results_H = np.full(shape=(len(pickles_Xp),), dtype=lnLtype, fill_value=(np.nan,)*len(lnLtype))
        self.maxlike_results_L = np.full(shape=(len(pickles_Yp),), dtype=lnLtype, fill_value=(np.nan,)*len(lnLtype))
        self.maxlike_priors_H = np.full(shape=(len(pickles_Xp),), dtype=object, fill_value=np.nan)
        self.maxlike_priors_L = np.full(shape=(len(pickles_Yp),), dtype=object, fill_value=np.nan)

        for pickly in [pickles_Xp, pickles_Yp]:
            for i, pkl in enumerate(pickly):
                f = self.load_data(pkl)
                lnL = -f['opt_result']['fbest']
                per = f['priors']['per' + str(len(self.planets_dict))]
                k = f['priors']['k' + str(len(self.planets_dict))]
                if pickly == pickles_Xp:
                    self.maxlike_results_H[i] = tuple((lnL, per, k))
                    self.maxlike_priors_H[i] = f['priors']
                elif pickly == pickles_Yp:
                    self.maxlike_results_L[i] = tuple((lnL, per, k))
                    self.maxlike_priors_L[i] = f['priors']

        return self.maxlike_results_H, self.maxlike_results_L

    def organize_injection_data(self, save=True):
        self.periods = np.unique(self.full_run_data['pb_in'])
        self.semiamps = np.unique(self.full_run_data['kb_in'])

        # Alias
        pb = self.periods
        kbin = self.semiamps

        n = len(self.data.get_vec('y'))
        self.kbfrac = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.kbfrac_unc = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.kbfrac_unc_rec = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.kb_rec = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.lnLH = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.lnLL = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.delta_lnL = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.pars_sorted = np.full(shape=(len(kbin), len(pb)), dtype=object, fill_value=np.nan)
        self.pars_likeH = np.full(shape=(len(kbin), len(pb)), dtype=object, fill_value=np.nan)
        self.pars_likeL = np.full(shape=(len(kbin), len(pb)), dtype=object, fill_value=np.nan)
        self.aiccH = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.aiccL = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)
        self.delta_aicc = np.full(shape=(len(kbin), len(pb)), fill_value=np.nan)

        self.gp_sorted = np.full(shape=(len(kbin), len(pb)), dtype=self.gptype, fill_value=(np.nan,)*self.gptype)
        self.gp_unc_sorted = np.full(shape=(len(kbin), len(pb)), dtype=self.gpunctype, fill_value=(np.nan,)*self.gpunctype)

        for x in range(len(pb)):
            for y in range(len(kbin)):
                c = np.where(np.isclose(self.full_run_data['pb_in'], pb[x]) & np.isclose(self.full_run_data['kb_in'], kbin[y]))[0]
                eh = np.where(np.isclose(self.maxlike_results_H['per'], pb[x]) & np.isclose(self.maxlike_results_H['k'], kbin[y]))[0]
                el = np.where(np.isclose(self.maxlike_results_L['per'], pb[x]) & np.isclose(self.maxlike_results_L['k'], kbin[y]))[0]
                if c.size:
                    self.kbfrac[y, x] = float(self.full_run_data['kb_out'][c] / self.full_run_data['kb_in'][c])
                    self.kbfrac_unc[y, x] = float(np.mean(self.full_run_data['kb_unc'][c][0]) / self.full_run_data['kb_in'][c])
                    self.kbfrac_unc_rec[y, x] = float(np.mean(self.full_run_data['kb_unc'][c][0]) / self.full_run_data['kb_out'][c])
                    self.kb_rec[y, x] = float(self.full_run_data['kb_out'][c])
                    self.pars_sorted[y, x] = self.priors[c]
                    self.gp_sorted[y, x] = self.gp[c]
                    self.gp_unc_sorted[y, x] = self.gp_unc[c]
                if eh.size > 1:
                    eh = eh[0]
                    self.lnLH[y, x] = float(np.ma.masked_invalid(self.maxlike_results_H['lnL'][eh]))
                    self.pars_likeH[y, x] = self.maxlike_priors_H[eh]
                if el.size > 1:
                    el = el[0]
                    self.lnLL[y, x] = float(np.ma.masked_invalid(self.maxlike_results_L['lnL'][el]))
                    self.pars_likeL[y, x] = self.maxlike_priors_L[el]
                if eh.size and el.size:
                    self.delta_lnL[y, x] = self.lnLH[y, x] - self.lnLL[y, x]
                if c.size and eh.size and el.size:
                    kh = self.pars_likeH[y, x][0].num_varied()
                    kl = self.pars_likeL[y, x][0].num_varied()
                    self.aiccH[y, x] = self.get_aicc(kh, self.lnLH[y, x], n)
                    self.aiccL[y, x] = self.get_aicc(kl, self.lnLL[y, x], n)
                    self.delta_aicc[y, x] = self.aiccH[y, x] - self.aiccL[y, x]

        if save:
            savedict = {'kbfrac': self.kbfrac, 'kbfrac_unc': self.kbfrac_unc, 'kb_rec': self.kb_rec, 'kbfrac_unc_rec': self.kbfrac_unc_rec,
                        'lnLH': self.lnLH, 'lnLL': self.lnLL, 'delta_lnL': self.delta_lnL, 'pars_sorted': self.pars_sorted, 'pars_likeH': self.pars_likeH,
                        'pars_likeL': self.pars_likeL, 'aiccH': self.aiccH, 'aiccL': self.aiccL, 'delta_aicc': self.delta_aicc, 'gp_sorted': self.gp_sorted,
                        'gp_unc_sorted': self.gp_unc_sorted}
            serialized = json.dumps(savedict, indent=4, sort_keys=True)
            with open(self.path + os.sep + '{}_organized_data_{}.pkl'.format(self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb') as handle:
                pickle.dump(savedict, handle)
            with open(self.path + os.sep + '{}_organized_data_{}.json'.format(self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb') as handle:
                handle.write(serialized)

        return self.kbfrac, self.kbfrac_unc, self.kb_rec, self.kbfrac_unc_rec, self.gp_sorted, self.gp_unc_sorted, self.delta_lnL, self.delta_aicc

    @staticmethod
    def get_aicc(k, lnL, n):
        aic = -2.0 * lnL + 2.0 * k
        _aicc = aic
        denom = n - k - 1.0
        if denom > 0:
            _aicc += (2.0 * k * (k + 1.0)) / denom
        return _aicc

    @staticmethod
    def load_data(path):
        f = pickle.load(open(path, 'rb'))
        return f

    def get_data(self, result):
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

    def plot_injection_2D_hist(self, injection=True, vector=False):
        assert self.kbfrac is not None
        fig, (ax1, ax2) = plt.subplots(2)
        # norm = None if planets_model == 1 else colors.LogNorm()
        norm1 = colors.LogNorm()
        norm2 = colors.LogNorm()
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        cbar_ax = fig.add_axes([0.83, pos1.y0, 0.025, pos1.height])
        cbar2_ax = fig.add_axes([0.83, pos2.y0, 0.025, pos2.height])
        bounds = (1e-1, 20) if injection else (1e-3, 20)
        plot = ax1.pcolormesh(self.periods, self.semiamps, self.kbfrac, cmap='RdYlGn', vmin=bounds[0], vmax=bounds[1], norm=norm1,
                              shading='nearest')
        cb = fig.colorbar(plot, cax=cbar_ax, label='$K_{\\mathrm{recovered}}\\ /\\ K_{\\mathrm{injected}}$')
        bounds = (1 / 10, 1 / 1e-3) if injection else (1 / 50, 1 / 1e-1)
        plot_b = ax2.pcolormesh(self.periods, self.semiamps, 1/self.kbfrac_unc if injection else 1/self.kbfrac_unc_rec,
                                cmap='RdYlBu', vmin=bounds[0], vmax=bounds[1], norm=norm2, shading='nearest')
        istr = 'injected' if injection else 'recovered'
        cb2 = fig.colorbar(plot_b, cax=cbar2_ax, label='$K_{\\mathrm{%s}}\\ /\\ \\sigma_{K}$' % istr)
        ax2.set_xlabel('Injected Period [days]')
        fig.text(0.02, 0.5, 'Injected Semiamplitude [m s$^{-1}$]', va='center', rotation='vertical')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax1.minorticks_off()
        ax2.minorticks_off()
        step = 3
        ax1.set_xticks(self.periods[::2])
        ax1.set_yticks(self.semiamps[::step])
        ax2.set_xticks(self.periods[::2])
        ax2.set_yticks(self.semiamps[::step])
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
        ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
        cb2.ax.minorticks_off()
        fig.subplots_adjust(right=0.8, hspace=0.3)
        ftype = 'png' if not vector else 'eps'
        plt.savefig(self.path + os.sep + '{}injection_recovery_histograms_{}_{}.{}'.format('non' if not injection else '',
                                                                       self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                                                                      ftype),
                    bbox_extra_artist=(cb, cb2), dpi=300)
        plt.close()

    def plot_delta_aicc(self, vector=False):
        colors_below0 = plt.cm.PuOr(np.linspace(1, 0.5, 256))
        colors_above0 = plt.cm.RdYlGn(np.linspace(0.5, 1, 256))
        all_colors = np.vstack((colors_below0, colors_above0))
        colormap = colors.LinearSegmentedColormap.from_list('red-purple', all_colors)

        assert self.delta_aicc is not None
        fig, ax = plt.subplots()
        lognorm = colors.SymLogNorm(1, vmin=-10 ** 6, vmax=10 ** 6, base=10)
        plot = ax.pcolormesh(self.periods, self.semiamps, self.delta_aicc, cmap=colormap, norm=lognorm, rasterized=True, shading='nearest')
        ax.set_xlabel('Injected Period [days]')
        ax.set_ylabel('Injected Semiamplitude [m s$^{-1}$]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.minorticks_off()
        ax.set_xticks(self.periods[::2])
        ax.set_yticks(self.semiamps[::2])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        stat_str = 'delta_AICc'
        cbar = fig.colorbar(plot, ax=ax, label='$\\Delta$ AICc')
        cbar.set_ticks(
            [-10 ** 6, -10 ** 5, -10 ** 4, -10 ** 3, -10 ** 2, -10, -1, 0, 1, 10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5,
             10 ** 6])
        ftype = 'png' if not vector else 'eps'
        plt.savefig(self.path + os.sep + 'delta_aicc_histograms_{}_{}.{}'.format(self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), ftype),
                    dpi=300)
        plt.close()

    def plot_gp_histograms(self, weightskb=None, weightsgp=None, vector=False, cutoff=1e5, bins=100):
        ftype = 'png' if not vector else 'eps'

        fig, ax = plt.subplots()
        kbfrac_unc_flat = 1 / self.kbfrac_unc.ravel()
        kbfrac_unc_flat = kbfrac_unc_flat[np.where(np.isfinite(kbfrac_unc_flat) & (kbfrac_unc_flat < cutoff))]
        ax.hist(kbfrac_unc_flat, bins=100, weights=weightskb, density=True)
        ax.set_yscale('log')
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('$K_{\\mathrm{injected}}\\ /\\ \\sigma_{K}$')
        plt.savefig(self.path + os.sep + '1d_injection_histogram_kunc_{}_{}.{}'.format(
            self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), ftype),
                    edges='tight')
        plt.close()

        for field in self.gp_sorted.dtype.names:
            fig, ax = plt.subplots()
            gp_param = self.gp_sorted[field].ravel()
            gp_param = gp_param[np.where(np.isfinite(gp_param) & (np.abs(gp_param) < cutoff))]
            ax.hist(gp_param, bins=bins, weights=weightsgp, density=True)
            ax.set_xlabel('gp_' + field)
            ax.set_yscale('log')
            ax.set_ylabel('{}weighted Probability Density'.format('un' if not weightsgp else ''))
            plt.savefig(self.path + os.sep + '1d_injection_histogram_{}_{}_{}.{}'.format(field,
            self.star_name.replace(' ', '_'), datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), ftype),
                edges='tight')
            plt.close()
