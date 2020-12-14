import optimize.scores as optscore
import optimize.kernels as optnoisekernels
import optimize.optimizers as optimizers
import optimize.frameworks as optframeworks
import plotly.subplots
import gls
import plotly.graph_objects
from itertools import chain, combinations
import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
import copy
import pychell.orbits.rvmodels as pcrvmodels
plt.style.use("gadfly_stylesheet")

class ExoProblem(optframeworks.OptProblem):
    
    planet_par_base_names = pcrvmodels.RVModel.planet_par_base_names
    
    def __init__(self, *args, star_name=None, likes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.star_name = 'Star' if star_name is None else star_name
        self.scorer = likes
    
    def rv_phase_plot(self, planet_index, opt_result=None, show=True):
        """Creates a phased rv plot for a given planet with the model on top.

        Args:
            planet_index (int): The planet index/
            opt_result (dict, optional): The optimization or sampler result to use. Defaults to None, and uses the initial parameters.
            show (bool, optional): Whether to show or return the figure. Defaults to True.

        Returns:
            plt.figure: A matplotlib figure containing the plot if show is False, and None if True.
        """
        
        # Resolve which pars to use
        if opt_result is None:
            pars = self.p0
        else:
            pars = opt_result["pbest"]
            
        # Creat a figure
        plt.clf()
        fig = plt.figure(1)
        
        # Alias
        per = pars['per' + str(planet_index)].value
        tc = pars['tc' + str(planet_index)].value
        
        # Create the phased model, high res
        t_hr = np.linspace(0, per, num=500)
        phases_hr = self.get_phases(t_hr, per, tc)
        like0 = next(iter(self.scorer.values()))
        planet_model_phased = like0.model.build_planet(pars, t_hr, planet_index)
        
        # Loop over likes
        for like in self.scorer.values():
            
            # Create a data rv vector where everything is removed except this planet
            if like.model.has_gp:
                residuals = like.residuals_after_kernel(pars)
            else:
                residuals = like.residuals_before_kernel(pars)
            
            # Compute the data with only the planet
            mod_data = like.model.data_only_planet(pars, planet_index)
            
            # Compute error bars
            errors = like.model.kernel.compute_data_errors(pars)
            
            # Loop over instruments and plot each
            for data in like.data.values():
                _errors = errors[like.model.data_inds[data.label]]
                _data = mod_data[data.label]
                #_data = residuals[like.model.data_inds[data.label]] + like.model.build_planet(pars, data.t, planet_index)
                phases_data = self.get_phases(data.t, per, tc)
                plt.errorbar(phases_data, _data, yerr=_errors, marker='o', lw=0, elinewidth=1, alpha=0.8, label=data.label)

        # Plot the model on top
        ss = np.argsort(phases_hr)
        plt.plot(phases_hr[ss], planet_model_phased[ss], c='black')
        
        # Axis labels
        plt.xlabel('Phase', fontsize=12, fontweight='bold')
        plt.ylabel('RV $\mathrm{ms}^{-1}$', fontsize=12, fontweight='bold')
        plt.title(self.star_name + ' ' + like0.model.planets_dict[planet_index], fontweight='bold', fontsize=14)
        
        # Legend
        plt.legend(loc='upper right')
        
        # Show or return
        if show:
            plt.show()
        else:
            return fig

    def rv_plot(self, opt_result, n_rows=1, n_model_pts=5000, time_offset=2450000, show=True, backend='plotly'):
        """Creates an rv plot for the full dataset and rv model.

        Args:
            opt_result (dict, optional): The optimization or sampler result to use. Defaults to None, and uses the initial parameters.
            show (bool, optional): Whether to show or return the figure. Defaults to True.
            n_rows (int, optional): The number of rows to split the plot into. Defaults to 1.

        Returns:
            plt.figure: A matplotlib figure containing the plot if show is False, and None if True.
        """
        
        # Resolve which pars to use
        if opt_result is None:
            pars = self.p0
        else:
            pars = opt_result["pbest"]
            
        # Create a figure
        if backend == 'pyplot':
            fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True)
        else:
            fig = plotly.subplots.make_subplots(rows=2, cols=1)
        
        # Create the full planet model, high res.
        # Use a different time grid for the gp since det(K)~ O(n^3)
        t_data_all = self.data.get_vec('t')
        t_start, t_end = np.nanmin(t_data_all), np.nanmax(t_data_all)
        t_hr = np.linspace(t_start, t_end, num=n_model_pts)
        like0 = next(iter(self.scorer.values()))
        model_arr_hr = like0.model._builder(pars, t_hr)
        
        # Loop over likes and:
        # 1. Create high res GP
        # 2. Plot high res GP and data
        for like in self.scorer.values():
            
            # High res grid for this gp, smartly sampled
            t_hr_gp = np.array([], dtype=float)
            gpmus = []
            
            # Loop over all indices for this like
            for i in range(like.data_t.size):
                t_hr_gp = np.concatenate((t_hr_gp, np.linspace(like.data_t[i] - 3, like.data_t[i] + 3, num=20)))
            t_hr_gp = np.sort(t_hr_gp)
        
            # Create the high res GP and plot as well, separately
            errors = like.model.kernel.compute_data_errors(pars)
            if like.model.has_gp:
                residuals = like.residuals_before_kernel(pars)
                gpmu, gpstddev = like.model.kernel.realize(pars, xpred=t_hr_gp, residuals=residuals, return_unc=True)

            # Further loop over actual instruments and plot each
            for data in like.data.values():
                data_arr_offset = data.rv - pars['gamma_' + data.label].value
                _errors = errors[like.model.data_inds[data.label]]
                if backend == 'pyplot':
                    axarr[0].errorbar(data.t - time_offset, data_arr_offset, yerr=_errors, marker='o', lw=0, elinewidth=1, alpha=0.8, label=data.label)
                else:
                    _yerr = dict(type='constant', array=_errors)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=data_arr_offset, name=data.label, error_y=_yerr, mode='markers'), row=1, col=1)

            # Plot the model
            if backend == 'pyplot':
                axarr[0].plot(t_hr - time_offset, model_arr_hr, c='black', lw=1)
                axarr[0].plot(t_hr_gp - time_offset, gpmu, c='red', lw=0.8)
            else:
                fig.add_trace(plotly.graph_objects.Scatter(x=t_hr - time_offset, y=model_arr_hr, line=dict(color='black', width=2)), row=1, col=1)
                if like.model.has_gp:
                    fig.add_trace(plotly.graph_objects.Scatter(x=t_hr_gp - time_offset, y=gpmu, line=dict(width=1)), row=1, col=1)
        
            # Now plot residuals
            if like.model.has_gp:
                residuals = like.residuals_after_kernel(pars)
            else:
                residuals = like.residuals_before_kernel(pars)
            
            # Loop over actual instruments and plot each again
            for data in like.data.values():
                _errors = errors[like.model.data_inds[data.label]]
                _residuals = residuals[like.model.data_inds[data.label]]
                if backend == 'pyplot':
                    axarr[1].errorbar(data.t - time_offset, _residuals, yerr=_errors, marker='o', lw=0, elinewidth=1, alpha=0.8, label=data.label)
                else:
                    _yerr = dict(type='constant', array=_errors)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_residuals, error_y=_yerr, mode='markers'), row=2, col=1)
                
        # Add a legend
        if backend == 'pyplot':
            axarr[0].legend(loc='upper right')
            
        # Plot the residuals
        # First plot a zero line
        if backend == 'pyplot':
            axarr[1].axhline(0, ls=':', alpha=0.5)
        else:
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=0, line=dict(color='Black'), xref='paper', yref='paper')
        
        # Labels
        if backend == 'pyplot':
            axarr[1].set_xlabel('BJD - ' + str(time_offset), fontsize=12, fontweight='bold')
            axarr[0].set_ylabel('RVs [m/s]', fontsize=12, fontweight='bold')
            axarr[1].set_ylabel('Residual RVs [m/s]', fontsize=8, fontweight='bold')
            axarr[0].set_title(self.star_name + ' RVs', fontweight='bold')
        else:
            fig.update_xaxes(title_text='BJD - ' + str(time_offset), row=2, col=1)
            fig.update_yaxes(title_text='RVs [m/s]', row=1, col=1)
            fig.update_yaxes(title_text='Residual RVs [m/s]', row=2, col=1)
            fig.update_yaxes(title_text=self.star_name + ' RVs', row=1, col=1)
        
        # Show or return
        if backend == 'pyplot':
            if show:
                plt.show()
            else:
                return fig
        else:
            return fig
        
    def rv_period_search(self, pars=None, pmin=None, pmax=None, apply_gp=True, remove_planets=None):
        
        # Resolve parameters
        if pars is None:
            pars = self.p0
            
        # Resolve which planets to remove by default
        if remove_planets is None:
            remove_planets = {}
        
        # Resolve period min and period max
        if pmin is None:
            pmin = 1.1
        if pmax is None:
            times = self.data.get_vec('x')
            pmax = np.max(times) - np.min(times)
        if pmax <= pmin:
            raise ValueError("Pmin is less than Pmax")
            
        # Arrays for periodogram
        data_times = np.array([], dtype=float)
        data_rvs = np.array([], dtype=float)
        data_errors = np.array([], dtype=float)
        
        # Loop over likes and add to above arrays
        for like in self.scorer.values():
            
            # Add times and errors
            data_times = np.concatenate((data_times, like.data.get_vec('x')))
            data_errors = np.concatenate((data_errors, like.model.kernel.compute_data_errors(pars)))
            
            # Remove GP from data first
            if like.model.has_gp and apply_gp:
                
                # Copy original parameters, optimizer, and planets dict
                p0cp = copy.deepcopy(self.p0)
                p0_mod = copy.deepcopy(self.p0)
                planets_dict_cp = copy.deepcopy(like.model.planets_dict)
                optimizercp = copy.deepcopy(self.optimizer)
                
                # Loop over planets and remove from parameters
                for planet in planets_dict_cp:
                    if planet in remove_planets:
                        self.remove_planet_pars(p0_mod, planet)
                    
                # Set the new parameters without planets
                self.set_pars(p0_mod)
                self.optimizer = optimizers.NelderMead(scorer=self.optimizer.scorer)
                
                # Set the planets dict to be what's left (difference of two dicts)
                like.model.planets_dict = dict_diff(planets_dict_cp, remove_planets)
                
                # Optimize without some planets
                opt_result = self.optimize()
                
                # Alias pbest
                pbest = opt_result['pbest']
                
                # Add to data vec
                data_rvs = np.concatenate((data_rvs, like.residuals_after_kernel(pbest)))
                
                # Set pars, planets dict, and optimizer back to original
                self.set_pars(p0cp)
                like.model.planets_dict = planets_dict_cp
                self.optimizer = optimizercp
                
            else:
                
                # Copy original parameters, optimizer, and planets dict
                p0cp = copy.deepcopy(self.p0)
                p0_mod = copy.deepcopy(self.p0)
                planets_dict_cp = copy.deepcopy(like.model.planets_dict)
                optimizercp = copy.deepcopy(self.optimizer)
                
                # Loop over planets and remove from parameters
                for planet in planets_dict_cp:
                    if planet in remove_planets:
                        self.remove_planet_pars(p0_mod, planet)
                    
                # Set the new parameters without planets
                self.set_pars(p0_mod)
                self.optimizer = optimizers.NelderMead(scorer=self.optimizer.scorer)
                
                # Set the planets dict to be what's left (difference of two dicts)
                like.model.planets_dict = dict_diff(planets_dict_cp, remove_planets)
                
                # Optimize without some planets
                opt_result = self.optimize()
                
                # Alias pbest
                pbest = opt_result['pbest']
                
                # Add to data vec
                data_rvs = np.concatenate((data_rvs, like.residuals_after_kernel(pbest)))
                
                # Set pars, planets dict, and optimizer back to original
                self.set_pars(p0cp)
                like.model.planets_dict = planets_dict_cp
                self.optimizer = optimizercp
        
        # Call GLS
        pgram = gls.Gls((data_times, data_rvs, data_errors), Pbeg=pmin, Pend=pmax)
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        fig.add_trace(plotly.graph_objects.Scatter(x=1 / pgram.f, y=pgram.power, line=dict(color='black', width=1)), row=1, col=1)
        fig.update_xaxes(title_text='Period [days]', row=1, col=1)
        fig.update_yaxes(title_text='Power', row=1, col=1)
        return fig
        

    @staticmethod
    def get_phases(t, per, tc):
        """Given input times, a period, and time of conjunction, returns the phase [0, 1] at each time t.

        Args:
            t (np.ndarray): The times.
            per (float): The period of the planet
            tc (float): The time of conjunction (time of transit).

        Returns:
            np.ndarray: The phases between 0 and 1
        """
        alpha = tc - per / 2
        phases = (t - alpha) % per
        phases /= np.nanmax(phases)
        return phases
    
    @staticmethod
    def generate_all_planet_dicts(planets_dict):
        pset = powerset(planets_dict.items())
        planet_dicts = []
        for item in pset:
            pdict = {}
            for subitem in item:
                pdict[subitem[0]] = subitem[1]
            planet_dicts.append(pdict)
        return planet_dicts
    
    def model_comparison(self, do_planets=True, do_gp=False):
        
        model_comp_results: Union[None, list] = None
        
        # Go through every permutation of planet dicts
        if do_planets:
            
            model_comp_results = []
            
            like0 = next(iter(self.scorer.values()))
            
            # Get all combos
            planet_dicts = self.generate_all_planet_dicts(like0.model.planets_dict)
            
            # Loop over combos
            for i, planets_dict in enumerate(planet_dicts):
                
                # Copy self
                _optprob = copy.deepcopy(self)
                
                # Alias pars
                p0 = _optprob.p0
                
                # Remove all other planets except this combo.
                for like in _optprob.scorer.values():
                    like.model.planets_dict = planets_dict
                for planet in like0.model.planets_dict:
                    if planet not in planets_dict:
                        _optprob.remove_planet_pars(p0, planet)
                
                # Pars
                _optprob.set_pars(p0)
                
                # Reset the optimizer
                _optprob.optimizer = optimizers.NelderMead(scorer=_optprob.optimizer.scorer)

                # Run the max like
                opt_result = _optprob.optimize()
                pbest = opt_result['pbest']
                lnL = -1 * opt_result['fbest']
                
                # Run the BIC
                bic = _optprob.optimizer.scorer.compute_bic(pbest)
                
                # Run the AICc
                aicc = _optprob.optimizer.scorer.compute_aicc(pbest)
                
                # Store
                model_comp_results.append({'planets_dict': planets_dict, 'lnL': lnL, 'bic': bic, 'aicc': aicc, 'pbest': pbest})
                
                del _optprob
                
            aicc_vals = np.array([model_comp_result['aicc'] for model_comp_result in model_comp_results], dtype=float)
            ss = np.argsort(aicc_vals)
            model_comp_results = [model_comp_results[ss[i]] for i in range(len(ss))]
            aicc_diffs = np.abs(aicc_vals[ss] - np.nanmin(aicc_vals)) # NOTE: 
            for i, model_comp_result in enumerate(model_comp_results):
                model_comp_result['delta_aicc'] = aicc_diffs[i]
            
        return model_comp_results
    
    def remove_planet_pars(self, pars, planet_index):
        """Removes all parameters corresponding to planets for the given index.

        Args:
            pars (Parameters): The parameters to remove planets from.
            planet_index (int): The planet index.
        """
        remove = []
        for pname1 in self.planet_par_base_names:
            for pname2 in pars:
                if pname2 == pname1 + str(planet_index):
                    remove.append(pname2)

        for pname in remove:
            del pars[pname]
            
    def corner_plot(self, *args, opt_result=None, **kwargs):
        return self.sampler.corner_plot(*args, sampler_result=opt_result, **kwargs)
            
            
            
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
            

def dict_diff(d1, d2):
    out = {}
    common_keys = set(d1) - set(d2)
    for key in common_keys:
        if key in d1:
            out[key] = d1[key]
    return out