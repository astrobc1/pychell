import optimize.scores as optscore
import optimize.kernels as optnoisekernels
import optimize.optimizers as optimizers
import optimize.knowledge as optknow
import pylatex
import corner
import scipy.constants
import optimize.frameworks as optframeworks
import plotly.subplots
import pychell.orbits.gls as gls
import itertools
from sklearn.cluster import DBSCAN
import tqdm
import plotly.graph_objects
import pickle
from itertools import chain, combinations
from joblib import Parallel, delayed
import numpy as np
from numba import jit, njit
import pylatex.utils
import matplotlib.pyplot as plt
import copy
import abc
import pychell.orbits.rvmodels as pcrvmodels
import optimize.kernels as optkernels
import pychell.orbits.rvkernels as pcrvkernels
import pychell.utils as pcutils
PLOTLY_COLORS = pcutils.PLOTLY_COLORS
import os
import pychell
import pychell.utils as pcutils
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

class ExoProblem(optframeworks.OptProblem):

    def __init__(self, output_path=None, data=None, p0=None, optimizer=None, sampler=None, likes=None, star_name=None, mstar=None, mstar_unc=None):
        super().__init__(p0=p0, data=data, optimizer=optimizer, sampler=sampler, scorer=likes)
        self.star_name = 'Star' if star_name is None else star_name
        self.output_path = output_path
        self.mstar = mstar
        self.mstar_unc = mstar_unc
        gen_latex_labels(self.p0, self.planets_dict)
        
    def generate_report(self, maxlike_result=None, model_comp_result=None, mcmc_result=None, n_model_pts=5000, time_offset=2450000, kernel_sampling=100):
        
        # Which result to use for plots
        if maxlike_result is not None:
            opt_result = maxlike_result
        elif mcmc_result is not None:
            opt_result = mcmc_result
        else:
            raise ValueError("Must provide either maxlike_result or mcmc_result")
        
        
        doc.preamble.append(pylatex.Command('title', self.star_name + ' Radial Velocities'))
        doc.append(pylatex.NoEscape(r'\maketitle'))
        
        # Model Comparison section
        if model_comp_result is not None:
            with doc.create(pylatex.LongTabu("X[r] X[r] X[r] X[r] X[r] X[r] X[r] X[r]")) as mctable:
                header_row = ["Model", "N_{free}", "N_{data}", "\\lnL", "BIC", "AICc", "\\Delta AICC"]
                mctable.add_row(header_row, mapper=[pylatex.utils.bold])
                mctable.add_hline()
                mctable.end_table_header()
                row = ["PA", "9", "$100", "%10", "$1000", "Test"]
                for mc_result in model_comp_result:
                    _model = ""
                    if len(mc_result["planets_dict"]) > 0:
                        for planet_dict in mc_result["planets_dict"].values():
                            _model += planets_dict["label"] + ", "
                        _model = _model[0:-2]
                    else:
                        _model = "-"
                    n_pars_vary = str(mc_result["pbest"].num_varied())
                    n_data = str(len(self.data.get_vec('t')))
                    lnL = str(round(mc_result["lnL"], 2))
                    bic = str(mc_result["bic"])
                    aicc = str(mc_result["aicc"])
                    delta_aicc = str(mc_result["delta_aicc"])
                    row = [_model, n_pars_vary, n_data, lnL, bic, aicc, delta_aicc]
                    mctable.add_row(row)
            
        # Parameters
        if mcmc_result is not None:
            with doc.create(pylatex.Subsection("MCMC Results")):
                doc.append(str(mcmc_result["pbest"]))
        else:
            with doc.create(pylatex.Subsection("Max Like Results")):
                doc.append(str(maxlike_result["pbest"]))
                
        # Full rv plot
        full_rv_plot = self.rv_plot(opt_result, n_model_pts=n_model_pts, time_offset=time_offset, kernel_sampling=kernel_sampling)
        fname = self.output_path + self.star_name.replace(' ', '_') + '_rvs_full_' + pcutils.gendatestr(time=True) + '.png'
        full_rv_plot.write_image(fname)
        with doc.create(pylatex.Subsection("Full RV Plot")):
            with doc.create(pylatex.Figure(position='h!')) as _full_rv_plot:
                _full_rv_plot.add_image(fname, width='120px')
                _full_rv_plot.add_caption('Best-fit model for ' + self.star_name + ' from the maximum likelihood model. Error bars are computed by adding in quadrature the provided error bars with a per-instrument Gaussian jitter-term. The per-isntrument offsets and agnostic linear and quadratic trends are removed from the data.')
        
        # Phased planet plots
        for planet_index in self.planets_dict:
            phased_rv_plot = self.rv_phase_plot(planet_index, opt_result=opt_result)
            label = self.planets_dict[planet_index]["label"]
            fname = self.output_path + self.star_name.replace(' ', '_') + label + '_rvs_phased_' + pcutils.gendatestr(time=True) + '.png'
            phased_rv_plot.write_image(fname)
            with doc.create(pylatex.Subsection("Planets")):
                with doc.create(pylatex.Figure(position='h!')) as _phased_rv_plot:
                    _phased_rv_plot.add_image(fname, width='120px')
                    _phased_rv_plot.add_caption('Best-fit Keplerian model for ' + self.star_name + label + ' from the maximum likelihood model. Error bars are computed by adding in quadrature the provided error bars with a per-instrument Gaussian jitter-term. The appropriate modified model has been subtracted off.')
                    
        # Corner plot
        if mcmc_result is not None:
            corner_plot = self.corner_plot(mcmc_result)
            fname = self.output_path + self.star_name.replace(' ', '_') + '_corner_' + pcutils.gendatestr(time=True) + '.png'
            corner_plot.write_image(fname)
            with doc.create(pylatex.Subsection("Corner plot")):
                with doc.create(pylatex.Figure(position='h!')) as _corner_plot:
                    _corner_plot.add_image(fname, width='120px')
                    _corner_plot.add_caption('The posterior distributions for ' + self.star_name + '. Truth values (blue lines) correspond to the values in table 2.')

        # Save doc
        fname = self.output_path + self.star_name.replace(' ', '_') + "_report_" + pcutils.gendatestr(time=True)
        doc.generate_pdf(fname, clean_tex=False)
        
        # Retusn doc
        return doc

    def rv_phase_plot(self, planet_index, opt_result=None, plot_width=1000, plot_height=600):
        """Creates a phased rv plot for a given planet with the model on top.

        Args:
            planet_index (int): The planet index
            opt_result (dict, optional): The optimization or sampler result to use. Defaults to None, and uses the initial parameters.

        Returns:
            plotly.figure: A plotly figure containing the plot.
        """
        
        # Resolve which pars to use
        if opt_result is None:
            pars = self.p0
        else:
            pars = opt_result["pbest"]
            
        # Creat a figure
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        
        # Convert parameters to standard basis and compute tc for consistent plotting
        per, tp, ecc, w, k = self.planets_dict[planet_index]["basis"].to_standard(pars)
        tc = pcrvmodels.tp_to_tc(tp, per, ecc, w)
        
        # Create the phased model, high res
        t_hr_one_period = np.linspace(tc, tc + per, num=500)
        phases_hr_one_period = get_phases(t_hr_one_period, per, tc)
        like0 = next(iter(self.scorer.values()))
        planet_model_phased = like0.model.build_planet(pars, t_hr_one_period, planet_index)
        
        # Loop over likes
        color_index = 0
        for like in self.scorer.values():
            
            # Create a data rv vector where everything is removed except this planet
            if like.model.has_gp:
                residuals = like.residuals_after_kernel(pars)
            else:
                residuals = like.residuals_before_kernel(pars)
            
            # Compute error bars
            errors = like.model.kernel.compute_data_errors(pars)
            
            # Loop over instruments and plot each
            for data in like.data.values():
                _errors = errors[like.model.data_inds[data.label]]
                _data = residuals[like.model.data_inds[data.label]] + like.model.build_planet(pars, data.t, planet_index)
                phases_data = get_phases(data.t, per, tc)
                _yerr = dict(array=_errors)
                fig.add_trace(plotly.graph_objects.Scatter(x=phases_data, y=_data, name="<b>" + data.label + "</b>", error_y=_yerr, mode='markers', marker=dict(color=PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)])))
                color_index += 1

        # Plot the model on top
        ss = np.argsort(phases_hr_one_period)
        fig.add_trace(plotly.graph_objects.Scatter(x=phases_hr_one_period[ss], y=planet_model_phased[ss], line=dict(color='black', width=2), name="<b>Keplerian Model</b>"))
        
        # Labels
        fig.update_xaxes(title_text='<b>Phase</b>')
        fig.update_yaxes(title_text='<b>RVs [m/s]</b>')
        fig.update_yaxes(title_text='<b>Residual RVs [m/s]</b>')
        fig.update_layout(title='<b>' + self.star_name + ' ' + like0.model.planets_dict[planet_index]["label"] + '<br>' + 'P = ' + str(per) + ', e = ' + str(ecc) + '</b>')
        fig.update_layout(template="ggplot2")
        fig.update_layout(font=dict(size=16), margin=dict(l=80, r=80, t=100, b=80))
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_layout(width=plot_width, height=plot_height, margin=dict(l=0, r=0, b=0, t=0, pad=0))
        fig.write_html(self.output_path + self.star_name.replace(' ', '_') + self.planets_dict[planet_index]["label"] + '_rvs_phased_' + pcutils.gendatestr(time=True) + '.html')
        
        # Return fig
        return fig

    def rv_plot(self, opt_result, n_model_pts=5000, time_offset=2450000, kernel_sampling=100, plot_width=1200, plot_height=800):
        """Creates an rv plot for the full dataset and rv model.

        Args:
            opt_result (dict, optional): The optimization or sampler result to use. Defaults to None, and uses the initial parameters.
            show (bool, optional): Whether to show or return the figure. Defaults to True.
            n_rows (int, optional): The number of rows to split the plot into. Defaults to 1.

        Returns:
            plotly.figure: A Plotly figure.
        """
        
        # Resolve which pars to use
        if opt_result is None:
            pars = self.p0
        else:
            pars = opt_result["pbest"]
            
        # Create a figure
        fig = plotly.subplots.make_subplots(rows=2, cols=1)
        
        # Create the full planet model, high res.
        # Use a different time grid for the gp since det(K)~ O(n^3)
        t_data_all = self.data.get_vec('t')
        t_start, t_end = np.nanmin(t_data_all), np.nanmax(t_data_all)
        dt = t_end - t_start
        t_hr = np.linspace(t_start - dt / 100, t_end + dt / 100, num=n_model_pts)
        
        # Create hr planet model
        like0 = next(iter(self.scorer.values()))
        model_arr_hr = like0.model._builder(pars, t_hr)

        # Plot the planet model
        fig.add_trace(plotly.graph_objects.Scatter(x=t_hr - time_offset, y=model_arr_hr, line=dict(color='black', width=2), name="<b>Keplerian Model</b>"), row=1, col=1)
        
        # Loop over likes and:
        # 1. Create high res GP
        # 2. Plot high res GP and data
        color_index = 0
        for like in self.scorer.values():
            
            # Data errors
            errors = like.model.kernel.compute_data_errors(pars)
            
            # RV Color
            if isinstance(like.model.kernel, pcrvkernels.RVColor):
                
                # Generate the residuals
                residuals_with_noise = like.residuals_before_kernel(pars)
                residuals_no_noise = like.residuals_after_kernel(pars)
                
                s = pars["gp_per"].value
                
                # Plot a GP for each instrument
                for wavelength in like.model.kernel.unique_wavelengths:
                    
                    print("Plotting Wavelength = " + str(wavelength) + " nm")
                    
                    # Smartly sample gp
                    inds = like.model.kernel.get_wave_inds(wavelength)
                    t_hr_gp = np.array([], dtype=float)
                    gpmu_hr = np.array([], dtype=float)
                    gpstddev_hr = np.array([], dtype=float)
                    tvec = like.data.get_vec('t')
                    for i in range(tvec.size):
                        _t_hr_gp = np.linspace(tvec[i] - s, tvec[i] + s, num=kernel_sampling)
                        _gpmu, _gpstddev = like.model.kernel.realize(pars, xpred=_t_hr_gp, residuals=residuals_with_noise, return_unc=True, wavelength=wavelength)
                        t_hr_gp = np.concatenate((t_hr_gp, _t_hr_gp))
                        gpmu_hr = np.concatenate((gpmu_hr, _gpmu))
                        gpstddev_hr = np.concatenate((gpstddev_hr, _gpstddev))

                    ss = np.argsort(t_hr_gp)
                    t_hr_gp = t_hr_gp[ss]
                    gpmu_hr = gpmu_hr[ss]
                    gpstddev_hr = gpstddev_hr[ss]
                    
                    # Plot the GP
                    tt = t_hr_gp - time_offset
                    gp_lower, gp_upper = gpmu_hr - gpstddev_hr, gpmu_hr + gpstddev_hr
                    instnames = like.model.kernel.get_instnames_for_wave(wavelength)
                    label = '<b>GP ' + '&#x3BB;' + ' = ' + str(wavelength) + ' nm ' + '['
                    for instname in instnames:
                        label += instname + ', '
                    label = label[0:-2]
                    label += ']</b>'
                    fig.add_trace(plotly.graph_objects.Scatter(x=tt, y=gpmu_hr, line=dict(width=1.2, color=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.8)), name=label, showlegend=False), row=1, col=1)
                    fig.add_trace(plotly.graph_objects.Scatter(x=np.concatenate([tt, tt[::-1]]),
                                             y=np.concatenate([gp_upper, gp_lower[::-1]]),
                                             fill='toself',
                                             line=dict(color=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.8), width=1),
                                             fillcolor=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.6),
                                             name=label))
                    color_index += 1
                    
                # Plot the data on top of the GPs, and the residuals
                data_arr = np.copy(like.data_rv)
                data_arr = like.model.apply_offsets(data_arr, pars)
                for data in like.data.values():
                    
                    # Data errors for this instrument
                    _errors = errors[like.model.data_inds[data.label]]
                    _yerr = dict(array=_errors)
                    
                    # Data on top of the GP, only offset by gammas
                    _data = data_arr[like.model.data_inds[data.label]]
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_data, name="<b>" + data.label + "</b>", error_y=_yerr, mode='markers', marker=dict(color=PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)])), row=1, col=1)
                    
                    # Residuals for this instrument after the noise kernel has been removed
                    _residuals = residuals_no_noise[like.model.data_inds[data.label]]
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_residuals, name="<b>" + data.label + "</b>", error_y=_yerr, mode='markers', marker=dict(color=PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)]), showlegend=False), row=2, col=1)
                    
                    # Increase color index
                    color_index += 1
                    
            # Standard GP
            elif isinstance(like.model.kernel, optkernels.GaussianProcess):
                
                # Make hr array for GP
                if 'gp_per' in pars:
                    s = pars['gp_per'].value
                elif 'gp_decay' in pars:
                    s = pars['gp_decay'].value / 10
                else:
                    s = 10
                    
                t_hr_gp = np.array([], dtype=float)
                gpmu_hr = np.array([], dtype=float)
                gpstddev_hr = np.array([], dtype=float)
                residuals_with_noise = like.residuals_before_kernel(pars)
                for i in range(like.data_t.size):
                    _t_hr_gp = np.linspace(like.data_t[i] - s, like.data_t[i] + s, num=kernel_sampling)
                    _gpmu, _gpstddev = like.model.kernel.realize(pars, xpred=_t_hr_gp, residuals=residuals_with_noise, return_unc=True)
                    t_hr_gp = np.concatenate((t_hr_gp, _t_hr_gp))
                    gpmu_hr = np.concatenate((gpmu_hr, _gpmu))
                    gpstddev_hr = np.concatenate((gpstddev_hr, _gpstddev))
                
                ss = np.argsort(t_hr_gp)
                t_hr_gp = t_hr_gp[ss]
                gpmu_hr = gpmu_hr[ss]
                gpstddev_hr = gpstddev_hr[ss]
                
                # Plot the GP
                tt = t_hr_gp - time_offset
                gp_lower, gp_upper = gpmu_hr - gpstddev_hr, gpmu_hr + gpstddev_hr
                label = '<b>GP ' + like.label.replace('_', ' ') +  '</b>'
                fig.add_trace(plotly.graph_objects.Scatter(x=tt, y=gpmu_hr, line=dict(width=0.8, color=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.8)), name=label, showlegend=False), row=1, col=1)
                fig.add_trace(plotly.graph_objects.Scatter(x=np.concatenate([tt, tt[::-1]]),
                                            y=np.concatenate([gp_upper, gp_lower[::-1]]),
                                            fill='toself',
                                            line=dict(color='rgba(255,255,255,0)'),
                                            fillcolor=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.4),
                                            name=label))
            
                # Generate the residuals without noise
                residuals_no_noise = like.residuals_after_kernel(pars)
                
                # For each instrument, plot
                for data in like.data.values():
                    data_arr_offset = like.model.apply_offsets(data.rv, pars, instname=data.label)
                    _errors = errors[like.model.data_inds[data.label]]
                    _residuals = residuals_no_noise[like.model.data_inds[data.label]]
                    _yerr = dict(array=_errors)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=data_arr_offset, name=data.label, error_y=_yerr, mode='markers', marker=dict(color=PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)])), row=1, col=1)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_residuals, error_y=_yerr, mode='markers', marker=dict(color=PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)]), showlegend=False), row=2, col=1)
                    color_index += 1
                
            # White noise
            else:
                residuals_no_noise = like.residuals_after_kernel(pars)
                
                # For each instrument, plot
                for data in like.data.values():
                    data_arr_offset = data.rv - pars['gamma_' + data.label].value
                    _errors = errors[like.model.data_inds[data.label]]
                    _residuals = residuals_no_noise[like.model.data_inds[data.label]]
                    _yerr = dict(array=_errors)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=data_arr_offset, name=data.label, error_y=_yerr, mode='markers', marker=dict(color=PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)])), row=1, col=1)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_residuals, error_y=_yerr, mode='markers', marker=dict(color=PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)]), showlegend=False), row=2, col=1)
                    color_index += 1


        # Labels
        fig.update_xaxes(title_text='<b>BJD - ' + str(time_offset) + '</b>', row=2, col=1)
        fig.update_yaxes(title_text='<b>RVs [m/s]</b>', row=1, col=1)
        fig.update_yaxes(title_text='<b>Residual RVs [m/s]</b>', row=2, col=1)
        fig.update_yaxes(title_text='<b>' + self.star_name + ' RVs</b>', row=1, col=1)
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")

        # Limits
        fig.update_xaxes(range=[t_start - dt / 10 - time_offset, t_end + dt / 10 - time_offset], row=1, col=1)
        fig.update_xaxes(range=[t_start - dt / 10 - time_offset, t_end + dt / 10 - time_offset], row=2, col=1)
        
        # Appearance
        fig.update_layout(template="ggplot2")
        fig.update_layout(font=dict(size=16), margin=dict(l=80, r=80, t=100, b=80))
        fig.update_layout(width=plot_width, height=plot_height, margin=dict(l=0, r=0, b=0, t=0, pad=0))
        fig.write_html(self.output_path + self.star_name.replace(' ', '_') + '_rvs_full_' + pcutils.gendatestr(time=True) + '.html')
        
        # Return the figure for streamlit
        return fig
        
    def gls_periodogram(self, pmin=None, pmax=None, apply_gp=True, remove_planets=None):
        
        if remove_planets is None:
            remove_planets = []
        
        # Resolve period min and period max
        if pmin is None:
            pmin = 1.1
        if pmax is None:
            times = self.data.get_vec('x')
            pmax = np.max(times) - np.min(times)
        if pmax <= pmin:
            raise ValueError("Pmin is less than Pmax")
            
        # Get the data times. RVs and errors are created in each case below.
        data_times = self.data.get_vec('t')
        data_rvs = np.zeros_like(data_times)
        data_errors = np.zeros_like(data_times)
        
        # Cases
        if apply_gp and len(remove_planets) > 0:
            
            # Create new pars and planets dict, keep copies of old
            p0cp = copy.deepcopy(self.p0)
            p0mod = copy.deepcopy(self.p0)
            planets_dict_cp = copy.deepcopy(self.scorer.like0.model.planets_dict)
            planets_dict_mod = copy.deepcopy(self.scorer.like0.model.planets_dict)
            
            # If the planet is not in remove_planets, we want to disable it from the initial fitting so it stays in the data
            for planet_index in remove_planets:
                if planet_index not in remove_planets:
                    self.disable_planet_pars(p0mod, planets_dict_cp, planet_index)
                    del planets_dict_cp[planet_index]
                    
            # Set the planets dict
            for like in self.scorer.values():
                like.model.planets_dict = planets_dict_mod
                
            # Set the modified parameters
            self.set_pars(p0mod)
            
            # Perform max like fit
            opt_result = self.optimize()
            pbest = opt_result["pbest"]
                
            # Construct the GP for each like and remove from the data
            for like in self.scorer.values():
                errors = like.model.kernel.compute_data_errors(pbest)
                residuals_no_noise = like.residuals_after_kernel(pbest)
                for data in like.data.values():
                    inds = self.data.get_inds(data.label)
                    data_rvs[inds] = residuals_no_noise[like.model.data_inds[data.label]]
                    data_errors[inds] = errors[like.model.data_inds[data.label]]
                    
            # Reset parameters dict
            for like in self.scorer.values():
                like.model.planets_dict = planets_dict_cp
                
            # Reset parameters
            self.set_pars(p0cp)
        
        elif apply_gp and len(remove_planets) == 0:
            
            # Create new pars and planets dict, keep copies of old
            p0cp = copy.deepcopy(self.p0)
            p0mod = copy.deepcopy(self.p0)
            planets_dict_cp = copy.deepcopy(self.scorer.like0.model.planets_dict)
            planets_dict_mod = {}
            
            # Disable all planet parameters
            for planet_index in planets_dict_cp:
                self.disable_planet_pars(p0mod, planets_dict_cp, planet_index)
                    
            # Set the planets dict
            for like in self.scorer.values():
                like.model.planets_dict = planets_dict_mod
                
            # Set the modified parameters
            self.set_pars(p0mod)

            # Perform max like fit
            opt_result = self.optimize()
            pbest = opt_result["pbest"]
                
            # Construct the GP for each like and remove from the data
            for like in self.scorer.values():
                errors = like.model.kernel.compute_data_errors(pbest)
                residuals = like.residuals_before_kernel(pbest)
                gp_mean = like.model.kernel.realize(pbest, residuals, return_unc=False)
                data_arr = np.copy(like.data_rv)
                data_arr = like.model.apply_offsets(data_arr, pbest)
                for data in like.data.values():
                    inds = self.data.get_inds(data.label)
                    data_rvs[inds] = data_arr[like.model.data_inds[data.label]] - gp_mean[like.model.data_inds[data.label]]
                    data_errors[inds] = errors[like.model.data_inds[data.label]]
            
            # Reset parameters and planets_dict
            self.set_pars(p0cp)
            
            # Reset planets dict
            for like in self.scorer.values():
                like.model.planets_dict = planets_dict_cp
                
        elif not apply_gp and len(remove_planets) > 0:
            
            # Create new pars and planets dict, keep copies of old
            p0cp = copy.deepcopy(self.p0)
            p0mod = copy.deepcopy(self.p0)
            planets_dict_cp = copy.deepcopy(self.scorer.like0.model.planets_dict)
            planets_dict_mod = copy.deepcopy(self.scorer.like0.model.planets_dict)
            
            # If the planet is not in remove_planets, we want to disable it from the initial fitting so it stays in the data
            for planet_index in remove_planets:
                if planet_index not in remove_planets:
                    self.disable_planet_pars(p0mod, planets_dict_cp, planet_index)
                    del planets_dict_cp[planet_index]
                    
            # Set the planets dict
            for like in self.scorer.values():
                like.model.planets_dict = planets_dict_mod
                
            # Set the modified parameters
            self.set_pars(p0mod)
            
            # Perform max like fit
            opt_result = self.optimize()
            pbest = opt_result["pbest"]
                
            for like in self.scorer.values():
                errors = like.model.kernel.compute_data_errors(pbest)
                data_arr = np.copy(like.data_rv)
                data_arr = like.model.apply_offsets(data_arr, pbest)
                for data in like.data.values():
                    inds = self.data.get_inds(data.label)
                    data_rvs[inds] = data_arr[like.model.data_inds[data.label]]
                    data_errors[inds] = errors[like.model.data_inds[data.label]]
                    
            # Reset parameters dict
            for like in self.scorer.values():
                like.model.planets_dict = planets_dict_cp
            
            # Construct the best fit planets and remove from the data
            for planet_index in planets_dict_mod:
                data_rvs -= self.scorer.like0.model.build_planet(pbest, data_times, planet_index)
            
            # Reset parameters
            self.set_pars(p0cp)
            
        else:
            
            for like in self.scorer.values():
                errors = like.model.kernel.compute_data_errors(self.p0)
                data_arr = np.copy(like.data_rv)
                data_arr = like.model.apply_offsets(data_arr, self.p0)
                for data in like.data.values():
                    inds = self.data.get_inds(data.label)
                    data_rvs[inds] = data_arr[like.model.data_inds[data.label]]
                    data_errors[inds] = errors[like.model.data_inds[data.label]]

        
        # Call GLS
        pgram = gls.Gls((data_times, data_rvs, data_errors), Pbeg=pmin, Pend=pmax)
        
        return pgram

    def rv_period_search(self, pars=None, pmin=None, pmax=None, n_periods=None, n_cores=1, planet_index=None):
        
        # Resolve parameters
        if pars is None:
            pars = self.p0
            
        if planet_index is None:
            planet_index = 1
        
        # Resolve period min and period max
        if pmin is None:
            pmin = 1.1
        if pmax is None:
            times = self.data.get_vec('x')
            pmax = np.max(times) - np.min(times)
        if pmax <= pmin:
            raise ValueError("Pmin is less than Pmax")
        if n_periods is None:
            n_periods = 500
        
        # Periods array
        periods = np.geomspace(pmin, pmax, num=n_periods)
        args_pass = []
        
        # Construct all args
        for i in range(n_periods):
            args_pass.append((self, periods[i], planet_index))
        
        # Run in parallel
        persearch_results = Parallel(n_jobs=n_cores, verbose=0, batch_size=2)(delayed(self._rv_period_search_wrapper)(*args_pass[i]) for i in tqdm.tqdm(range(n_periods)))
        
        # Save
        with open(self.output_path + self.star_name.replace(' ', '_') + '_persearch_results_' + pcutils.gendatestr(time=True) + '.pkl', 'wb') as f:
            pickle.dump({"periods": periods, "persearch_results": persearch_results}, f)
            
        return periods, persearch_results
    
    def sample(self, *args, **kwargs):
        sampler_result = super().sample(*args, **kwargs)
        return sampler_result
    
    def corner_plot(self, sampler_result):
        """Constructs a corner plot.

        Args:
            sampler_result (dict): The sampler result

        Returns:
            fig: A matplotlib figure.
        """
        plt.clf()
        pbest_vary_dict = sampler_result["pmed"].unpack(vary_only=True)
        truths = pbest_vary_dict["value"]
        labels = [par.latex_str for par in sampler_result["pbest"].values() if par.vary]
        corner_plot = corner.corner(sampler_result["chains"], labels=labels, truths=truths, show_titles=True)
        corner_plot.savefig(self.output_path + self.star_name.replace(' ', '_') + '_corner_' + pcutils.gendatestr(time=True) + '.png')
        return corner_plot
    
    def full_rvcolor(self, pars=None, sep=0.3, time_offset=2450000, plot_width=1000, plot_height=500):
        """Computes the RV Color for all possible wavelengths.

        Args:
            pars (Parameters, optional): The parameters to use. Defaults to self.p0.
            sep (float, optional): The max separation to allow for. Defaults to 0.3.
            time_offset (int, optional): The time offset. Defaults to 2450000.
            plot_width (int, optional): The plot width in pixels. Defaults to 1000.
            plot_height (int, optional): The plot height in pixels. Defaults to 500.
        """
        
        if pars is None:
            pars = self.p0
            
        wave_vec = self.data.get_wave_vec()
        wave_pairs = self.generate_all_wave_pairs(wave_vec)
        for wave_pair in wave_pairs:
            rvcolor_result = self.compute_rvcolor(pars, wave_pair[0], wave_pair[1], sep=sep)
            self.plot_rvcolor(wave_pair[0], wave_pair[1], rvcolor_result, time_offset=time_offset, plot_width=plot_width, plot_height=plot_height)
    
    def plot_rvcolor(self, wave1, wave2, rvcolor_result, time_offset=2450000, plot_width=1000, plot_height=500):
        
        # Figure
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        
        # Offset coarse and dense time arrays
        t = rvcolor_result["jds_avg"] - time_offset
        t_hr = rvcolor_result["t_gp_hr"] - time_offset
        
        # Plot data color
        color_map = {}
        label_map = {}
        color_index = 0
        for i in range(len(t)):
            
            # The yerr
            yerr = dict(array=np.array(rvcolor_result["unccolor_data"][i]))
            
            # Check if this combo is in color_map
            show_legend = False
            if rvcolor_result["instnames"][i] not in color_map:
                instname1, instname2 = rvcolor_result["instnames"][i]
                color_map[rvcolor_result["instnames"][i]] = pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index], a=0.8)
                label_map[rvcolor_result["instnames"][i]] = "<b>Data Color, " + instname1 + " - " + instname2 + "</b>"
                show_legend = True
                color_index += 1
            fig.add_trace(plotly.graph_objects.Scatter(x=np.array(t[i]), y=np.array(rvcolor_result["rvcolor_data"][i]), name=label_map[rvcolor_result["instnames"][i]], error_y=yerr, mode='markers', marker=dict(color=color_map[rvcolor_result["instnames"][i]]), showlegend=show_legend))
        
        # Plot the difference of the GPs
        yerr = dict(array=rvcolor_result["gpstddev_color_hr"])
        fig.add_trace(plotly.graph_objects.Scatter(x=t_hr, y=rvcolor_result["gpmean_color_hr"], line=dict(color=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.8), width=1.2), showlegend=False))
        gp_lower = rvcolor_result["gpmean_color_hr"] - rvcolor_result["gpstddev_color_hr"]
        gp_upper = rvcolor_result["gpmean_color_hr"] + rvcolor_result["gpstddev_color_hr"]
        fig.add_trace(plotly.graph_objects.Scatter(x=np.concatenate([t_hr, t_hr[::-1]]),
                                    y=np.concatenate([gp_upper, gp_lower[::-1]]),
                                    fill='toself',
                                    line=dict(color=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.4)),
                                    fillcolor=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.4),
                                    name="<b>GP Color</b>"))
        
        # Labels
        title = "<b>GP RV Color (&#x3BB; [" + str(550) + "] - &#x3BB; [" + str(2350) + "])</b>"
        fig.update_layout(title_text=title, margin=dict(l=80, r=80, t=10, b=80))
        fig.update_xaxes(title_text='<b>BJD - ' + str(time_offset) + '</b>')
        fig.update_yaxes(title_text='<b>RVs [m/s]</b>')
        fig.update_yaxes(title_text='<b>RV Color [m/s]</b>')
        fig.update_layout(title='<b>' + self.star_name + ' Chromaticity')
        fig.update_layout(template="ggplot2")
        fig.update_layout(font=dict(size=16))
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_layout(width=plot_width, height=plot_height, margin=dict(l=0, r=0, b=0, t=0, pad=0))
        fig.write_html(self.output_path + self.star_name.replace(' ', '_') + '_rvcolor_' + pcutils.gendatestr(time=True) + '.html')
        
        return fig

    def plot_rvcolor_one_to_one(self, pars=None, plot_width=1000, plot_height=500, sep=0.3):
        
        # Pars
        if pars is None:
            pars = self.p0
            
        # Figure
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
            
        # Compute all possible wave pairs
        wave_vec = self.data.get_wave_vec()
        wave_pairs = self.generate_all_wave_pairs(wave_vec)
        
        # Add a 1-1 line
        x_line = np.arange(-150, 150, 1)
        fig.add_trace(plotly.graph_objects.Scatter(x=x_line, y=x_line, line=dict(color='black', dash='dash'), showlegend=False))
        
        # Loop over wave pairs and
        # 1. Compute data rv color
        # 2. Compute GP color
        color_index = 0
        for wave_pair in wave_pairs:
            
            # Compute RV Color for this pair
            rvcolor_result = self.compute_rvcolor(pars, wave_pair[0], wave_pair[1], sep=sep)
        
            # Loop over points for this pair and plot.
            color_map = {}
            label_map = {}
            t = rvcolor_result["jds_avg"]
            x = rvcolor_result["rvcolor_data"]
            x_unc = rvcolor_result["unccolor_data"]
            y = rvcolor_result["gp_color_data"]
            y_unc = rvcolor_result["gp_color_unc_data"]
            for i in range(len(t)):
            
                # The yerr for the data and GP
                _x_unc = dict(array=np.array(x_unc[i]))
                _y_unc = dict(array=np.array(y_unc[i]))
            
                # Check if this combo is in color_map
                show_legend = False
                if rvcolor_result["instnames"][i] not in color_map:
                    instname1, instname2 = rvcolor_result["instnames"][i]
                    color_map[rvcolor_result["instnames"][i]] = pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index], a=0.8)
                    label_map[rvcolor_result["instnames"][i]] = "<b>RV Color, " + instname1 + " - " + instname2 + "</b>"
                    show_legend = True
                    color_index += 1
                fig.add_trace(plotly.graph_objects.Scatter(x=np.array(x[i]), y=np.array(y[i]), name=label_map[rvcolor_result["instnames"][i]], error_x=_x_unc, error_y=_y_unc, mode='markers', marker=dict(color=color_map[rvcolor_result["instnames"][i]]), showlegend=show_legend))
                
        # Labels
        title = "<b>RV Color (&#x3BB; [" + str(550) + "] - &#x3BB; [" + str(2350) + "])</b>"
        fig.update_layout(title_text=title, margin=dict(l=80, r=80, t=10, b=80))
        fig.update_xaxes(title_text='<b>Data Color [m/s]</b>')
        fig.update_yaxes(title_text='<b>GP Color [m/s]</b>')
        fig.update_layout(title='<b>' + self.star_name + ' Chromaticity')
        fig.update_layout(template="ggplot2")
        fig.update_layout(font=dict(size=16))
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_layout(width=plot_width, height=plot_height, margin=dict(l=0, r=0, b=0, t=0, pad=0))
        fig.write_html(self.output_path + self.star_name.replace(' ', '_') + '_rvcolor_1_to_1_' + pcutils.gendatestr(time=True) + '.html')
        
        return fig

    def compute_rvcolor(self, pars, wave1, wave2, sep=0.3):
        """Computes the "RV-color" for wavelengths 1 and 2.

        Args:
            wave1 (float): The first wavelength.
            wave2 (float): The second wavelength.
            sep (float, optional): The max separation allowed to consider for observations in days. Defaults to 0.3.

        Returns:
            dict: A dictionary with the following keys:
                jds_avg: The mean jds for the data.
                rvcolor_data: The data RVs corresponding to this jd, RV1 - RV2.
                unccolor_data: The corresponding data uncertainties, computed by adding in quadrature.
                gp_color_data: The coarsely sampled GP-color.
                gp_color_unc_data: The corresponding unc for the coarsely sampled GP-color, GP1 - GP2.
                t_gp_hr: The densely sampled times for the GP.
                gpmean1_hr: The densely sampled GP1 mean.
                gpstddev1_hr: The densely sampled GP1 stddev.
                gpmean1_hr: The densely sampled GP2 mean.
                gpstddev2_hr: The densely sampled GP2 stddev.
                gpmean_color_hr: The densely sampled GP-color, GP1 - GP2.
                gpstddev_color_hr: The corresponding densely sampled GP-color unc, computed by adding in quadrature.
                instnames: A list containing 2-tuples of the instruments of each night, [, ... (instname1, isntname2), ...].
        """
        
        # Parameters
        if pars is None:
            pars = self.p0
        
        # Filter data to only consider each wavelength
        wave_vec = self.scorer.like0.model.kernel.make_wave_vec()
        times_vec = self.data.get_vec('t')
        rv_vec = self.data.get_vec('rv')
        rv_vec = self.like0.model.apply_offsets(rv_vec, pars)
        unc_vec = self.like0.model.kernel.compute_data_errors(pars)
        tel_vec = self.data.make_tel_vec()
        inds1 = np.where((wave_vec == wave1) | (wave_vec == wave2))[0]
        times_vec, rv_vec, unc_vec, wave_vec, tel_vec = times_vec[inds1], rv_vec[inds1], unc_vec[inds1], wave_vec[inds1], tel_vec[inds1]
        n_inst = len(np.unique(tel_vec))
        
        # If only 1 instrument, nothing to compute
        if n_inst < 2:
            return None

        # Loop over RVs and look for near-simultaneous RV color observations.
        prev_i = 0
        jds_avg = []
        rvcolor_data = []
        unccolor_data = []
        instnames = []
        n_data = len(times_vec)
        for i in range(n_data - 1):
            
            # If dt > sep, we have moved to the next night.
            # But first ook at all RVs from this night
            if times_vec[i+1] - times_vec[i] > sep:
                
                # The number of RVs on this night for these two wavelengths.
                n_obs_night = i - prev_i + 1
                
                # If only 1 observation for this night, skipi.
                if n_obs_night < 2:
                    prev_i = i + 1
                    continue
                
                # The indices for this night, relative to the filtered arrays 
                inds = np.arange(prev_i, i + 1).astype(int)
                
                # The number of unique wavelengths on this night. Will be either 1 or 2.
                n_waves = len(np.unique(wave_vec[inds]))
                
                # If not exactly 2, skip.
                if n_waves != 2:
                    prev_i = i + 1
                    continue
                
                # Determine which index is which, ensure they are different.
                ind1 = np.where(wave_vec[inds] == wave1)[0][0]
                ind2 = np.where(wave_vec[inds] == wave2)[0][0]
                assert ind1 != ind2
                
                # Compute color from these two observations.
                jds_avg.append(np.mean(times_vec[inds]))
                rvcolor_data.append(rv_vec[inds][ind1] - rv_vec[inds][ind2])
                unccolor_data.append(np.sqrt(unc_vec[inds][ind1]**2 + unc_vec[inds][ind2]**2))
                instnames.append((tel_vec[inds][ind1], tel_vec[inds][ind2]))
                
                # Move on.
                prev_i = i + 1
        
        # Check last night.
        n_obs_night = n_data - prev_i
        inds = np.arange(prev_i, n_data).astype(int)
        if n_obs_night == 2 and len(np.unique(wave_vec[inds])) == 2:
            ind1 = np.where(wave_vec[inds] == wave1)[0][0]
            ind2 = np.where(wave_vec[inds] == wave2)[0][0]
            assert ind1 != ind2
            jds_avg.append(np.mean(times_vec[inds]))
            rvcolor_data.append(rv_vec[inds][ind1] - rv_vec[inds][ind2])
            unccolor_data.append(np.sqrt(unc_vec[inds][ind1]**2 + unc_vec[inds][ind2]**2))
            instnames.append((tel_vec[inds][ind1], tel_vec[inds][ind2]))
         
        # Convert to numpy arrays
        breakpoint()
        jds_avg = np.array(jds_avg)
        rvcolor_data = np.array(rvcolor_data)
        unccolor_data = np.array(unccolor_data)
         
        # Now for the GP color.
        # Residuals with noise
        residuals_with_noise = self.like0.residuals_before_kernel(pars)
        
        # Compute the coarsely sampled GP for each wavelength.
        gp_mean1_data, gp_stddev1_data = self.like0.model.kernel.realize(pars, residuals_with_noise, xpred=jds_avg, return_unc=True, wavelength=wave1)
        gp_mean2_data, gp_stddev2_data = self.like0.model.kernel.realize(pars, residuals_with_noise, xpred=jds_avg, return_unc=True, wavelength=wave2)
        
        # Compute the coarsely sampled GP-color and unc
        gp_color_data = gp_mean1_data - gp_mean2_data
        gp_color_unc_data = np.sqrt(gp_stddev1_data**2 + gp_stddev2_data**2)
        
        # Compute the densely sampled GP-color
        t_gp_hr, gpmean1_hr, gpstddev1_hr = self.gp_smart_sample(pars, self.like0, s=pars["gp_per"].value, t=jds_avg, residuals=residuals_with_noise, kernel_sampling=100, return_unc=True, wavelength=wave1)
        _, gpmean2_hr, gpstddev2_hr = self.gp_smart_sample(pars, self.like0, s=pars["gp_per"].value, t=jds_avg, residuals=residuals_with_noise, kernel_sampling=100, return_unc=True, wavelength=wave2)
        gpmean_color_hr = gpmean1_hr - gpmean2_hr
        gpstddev_color_hr = np.sqrt(gpstddev1_hr**2 + gpstddev2_hr**2)
                
        # Return a mega dictionary
        out = dict(jds_avg=jds_avg, rvcolor_data=rvcolor_data, unccolor_data=unccolor_data,
                   gp_color_data=gp_color_data, gp_color_unc_data=gp_color_unc_data,
                   t_gp_hr=t_gp_hr, gpmean1_hr=gpmean1_hr, gpstddev1_hr=gpstddev1_hr,
                   gpmean2_hr=gpmean2_hr, gpstddev2_hr=gpstddev2_hr,
                   gpmean_color_hr=gpmean_color_hr, gpstddev_color_hr=gpstddev_color_hr,
                   instnames=instnames)
        return out
    
    def model_comparison(self):
        """Runs a model comparison for all combinations of planets.

        Returns:
            list: Each entry is a dict containing the model comp results for each case, and is sorted according to the small sample AIC.
        """
        
        # Go through every permutation of planet dicts
        if do_planets:
            
            # Store results in a list
            model_comp_results = []
            
            # Alias like0
            like0 = self.scorer.like0
            
            # Original planets dict
            planets_dict_cp = copy.deepcopy(like0.model.planets_dict)
            
            # Get all combos
            planet_dicts = self.generate_all_planet_dicts(like0.model.planets_dict)
            
            # Loop over combos
            for i, planets_dict in enumerate(planet_dicts):
                
                # Copy self
                _optprob = copy.deepcopy(self)
                
                # Alias pars
                p0 = _optprob.p0
                
                # Remove all other planets except this combo.
                for planet_index in planets_dict_cp:
                    if planet_index not in planets_dict:
                        self.disable_planet_pars(p0, planets_dict_cp, planet_index)
                
                # Set planets dict for each model
                for like in _optprob.scorer.values():
                    like.model.planets_dict = planets_dict
                
                # Pars
                _optprob.set_pars(p0)

                # Run the max like
                opt_result = _optprob.optimize()
                
                # Alias best fit params
                pbest = opt_result['pbest']
                
                # Recompute the max like to NOT include any priors to keep things consistent.
                lnL = self.scorer.compute_logL(pbest, apply_priors=False)
                
                # Run the BIC
                bic = _optprob.optimizer.scorer.compute_bic(pbest, apply_priors=False)
                
                # Run the AICc
                aicc = _optprob.optimizer.scorer.compute_aicc(pbest, apply_priors=False)
                
                # Store
                model_comp_results.append({'planets_dict': planets_dict, 'lnL': lnL, 'bic': bic, 'aicc': aicc, 'pbest': pbest})
                
                del _optprob
                
            aicc_vals = np.array([mcr['aicc'] for mcr in model_comp_results], dtype=float)
            ss = np.argsort(aicc_vals)
            model_comp_results = [model_comp_results[ss[i]] for i in range(len(ss))]
            aicc_diffs = np.abs(aicc_vals[ss] - np.nanmin(aicc_vals))
            for i, mcr in enumerate(model_comp_results):
                mcr['delta_aicc'] = aicc_diffs[i]
        
        # Save
        with open(self.output_path + self.star_name.replace(' ', '_') + '_modelcomp_' + pcutils.gendatestr(time=True) + '.pkl', 'wb') as f:
            pickle.dump(model_comp_results, f)
            
        return model_comp_results
    
    def compute_semimajor_axis(self, sampler_result, mstar=None, mstar_unc=None):
        """Computes the semi-major axis of each planet.

        Args:
            sampler_result (dict): The returned value from calling sample.
            mstar (float): The mass of the star in solar units.
            mstar (list): The uncertainty of the mass of the star in solar units, lower, upper.

        Returns:
            (dict): The semi-major axis, lower, and upper uncertainty of each planet in a dictionary.
        """
        if mstar is None:
            mstar = self.mstar
        if mstar_unc is None:
            mstar_unc = self.mstar_unc
        aplanets = {} # In AU
        G = scipy.constants.G
        for planet_index in self.planets_dict:
            perdist = []
            tpdist = []
            eccdist = []
            wdist = []
            kdist = []
            adist = []
            pars = copy.deepcopy(sampler_result["pmed"])
            for i in range(sampler_result["n_steps"]):
                for pname in self.planets_dict[planet_index]["basis"].pnames:
                    if pars[pname].vary:
                        ii = pars.index_from_par(pname, rel_vary=True)
                        pars[pname].value = sampler_result["chains"][i, ii]
                per, tp, ecc, w, k = self.planets_dict[planet_index]["basis"].to_standard(pars)
                perdist.append(per)
                tpdist.append(tp)
                eccdist.append(ecc)
                wdist.append(w)
                kdist.append(k)
                a = (G / (4 * np.pi**2))**(1 / 3) * (mstar * 1.988435E30)**(1 / 3) * (per * 86400)**(2 / 3) / 1.496E11 # in AU
                adist.append(a)
            val, unc_low, unc_high = self.sampler.chain_uncertainty(adist)
            if self.mstar_unc is not None:
                da_dMstar = (G / (4 * np.pi**2))**(1 / 3) * (mstar * 1.988435E30)**(-2 / 3) / 3 * (per * 86400)**(2 / 3)
                unc_low = np.sqrt(unc_low**2 + da_dMstar**2 * mstar_unc[0]**2)
                unc_high = np.sqrt(unc_high**2 + da_dMstar**2 * mstar_unc[1]**2)
                aplanets[planet_index] = (val, unc_low, unc_high)
            else:
                aplanets[planet_index] = (val, unc_low, unc_high)
        return aplanets
    
    def compute_planet_masses(self, sampler_result, mstar=None, mstar_unc=None):
        """Computes the value of msini and uncertainty for each planet in units of Earth Masses.

        Args:
            sampler_result (dict): The returned value from calling sample.
            mstar (float): The mass of the star in solar units.
            mstar (list): The uncertainty of the mass of the star in solar units, lower, upper.

        Returns:
            (dict): The mass, lower, and upper uncertainty of each planet in a dictionary.
        """
        if mstar is None:
            mstar = self.mstar
        if mstar_unc is None:
            mstar_unc = self.mstar_unc
        msiniplanets = {} # In jupiter masses
        for planet_index in self.planets_dict:
            perdist = []
            tpdist = []
            eccdist = []
            wdist = []
            kdist = []
            mdist = []
            pars = copy.deepcopy(sampler_result["pmed"])
            for i in range(sampler_result["n_steps"]):
                for pname in self.planets_dict[planet_index]["basis"].pnames:
                    if pars[pname].vary:
                        ii = pars.index_from_par(pname, rel_vary=True)
                        pars[pname].value = sampler_result["chains"][i, ii]
                per, tp, ecc, w, k = self.planets_dict[planet_index]["basis"].to_standard(pars)
                perdist.append(per)
                tpdist.append(tp)
                eccdist.append(ecc)
                wdist.append(w)
                kdist.append(k)
                mdist.append(compute_planet_mass(per, ecc, k, mstar))
            val, unc_low, unc_high = self.sampler.chain_uncertainty(mdist)
            if self.mstar_unc is not None:
                unc_low = np.sqrt(unc_low**2 + compute_planet_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[0]**2)
                unc_high = np.sqrt(unc_high**2 + compute_planet_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[1]**2)
                msiniplanets[planet_index] = (val, unc_low, unc_high)
            else:
                msiniplanets[planet_index] = (val, unc_low, unc_high)
        return msiniplanets
    
    def gp_smart_sample(self, pars, like, s, t, residuals, kernel_sampling=100, return_unc=True, wavelength=None):
        """Smartly samples the GP. Could be smarter.

        Args:
            like (RVLikelihood or RVChromaticLikelihood)
            s ([type]): The window around each point in t to sample the GP in days.
            t (np.ndarray): The times to consider.
            residuals (np.ndarray): The residuals to use to realize the GP.
            kernel_sampling (int): The number of points in each window.
            return_unc (bool, optional): Whether or not to return the gpstddev. Defaults to True.
            wavelength (float, optional): The wavelength to realize if relevant. Defaults to None.

        Returns:
            np.ndarray: The densley sampled times.
            np.ndarray: The densley sampled GP.
            np.ndarray: The densley sampled GP stddev if return_unc is True.
        """
        t_hr_gp = np.array([], dtype=float)
        gpmu_hr = np.array([], dtype=float)
        gpstddev_hr = np.array([], dtype=float)
        for i in range(t.size):
            _t_hr_gp = np.linspace(t[i] - s, t[i] + s, num=kernel_sampling)
            if return_unc:
                _gpmu, _gpstddev = like.model.kernel.realize(pars, xpred=_t_hr_gp, residuals=residuals, return_unc=return_unc, wavelength=wavelength)
            else:
                _gpmu = like.model.kernel.realize(pars, xpred=_t_hr_gp, residuals=residuals, return_unc=return_unc, wavelength=wavelength)
            t_hr_gp = np.concatenate((t_hr_gp, _t_hr_gp))
            gpmu_hr = np.concatenate((gpmu_hr, _gpmu))
            if return_unc:
                gpstddev_hr = np.concatenate((gpstddev_hr, _gpstddev))

        ss = np.argsort(t_hr_gp)
        t_hr_gp = t_hr_gp[ss]
        gpmu_hr = gpmu_hr[ss]
        if return_unc:
            gpstddev_hr = gpstddev_hr[ss]
            
        if return_unc:
            return t_hr_gp, gpmu_hr, gpstddev_hr
        else:
            return t_hr_gp, gpmu_hr
    
    @property
    def likes(self):
        return self.scorer
    
    @property
    def like0(self):
        return self.scorer.like0
    
    @property
    def planets_dict(self):
        return self.like0.model.planets_dict
    
    @staticmethod
    def _rv_period_search_wrapper(optprob, period, planet_index):
        """Internal function called by the brute force period search.

        Args:
            optprob (ExoProb): The optimize problem.
            period (float): The period to test.
            planet_index (int): The planet index.

        Returns:
            dict: The opt_result corresponding to this period.
        """
        p0 = optprob.p0
        p0['per' + str(planet_index)].value = period
        optprob.set_pars(p0)
        opt_result = optprob.optimize()
        return opt_result

    @staticmethod
    def generate_all_wave_pairs(wave_vec):
        wave_vec_unq = np.sort(np.unique(wave_vec))
        return list(itertools.combinations(wave_vec_unq, 2))

    @staticmethod
    def generate_all_planet_dicts(planets_dict):
        """Generates all possible planet dictionaries through a powerset.

        Args:
            planets_dict (dict): The planets dict.

        Returns:
            list: A list of all possible planet dicts.
        """
        pset = pcutils.powerset(planets_dict.items())
        planet_dicts = []
        for item in pset:
            pdict = {}
            for subitem in item:
                pdict[subitem[0]] = subitem[1]
            planet_dicts.append(pdict)
        return planet_dicts

    @staticmethod
    def disable_planet_pars(pars, planets_dict, planet_index):
        """Disables (sets vary=False) in-place for the planet parameters corresponding to planet_index.

        Args:
            pars (Parameters): The parameters.
            planets_dict (dict): The planets dict.
            planet_index (int): The index to disable.
        """
        for par in pars.values():
            for planet_par_name in planets_dict[planet_index]["basis"].names:
                if par.name == planet_par_name + str(planet_index):
                    pars[par.name].vary = False

def gen_latex_labels(pars, planets_dict):
    """Default Settings for latex labels for orbit fitting. Any GP parameters must be set manually via parameter.latex_str = "$latexname$"

    Args:
        pars (Parameters): The parameters to generate labels for
    
    Returns:
        dict: Keys are parameter names, values are latex labels.
    """
    
    for pname in pars:
        
        # Planets (per, tc, k, ecc, w, sqecosw, sqesinw, other bases added later if necessary)
        if pname.startswith('per') and pname[3:].isdigit():
            pars[pname].latex_str = "$P_{" + planets_dict[int(pname[-1])]["label"] + "}$"
        elif pname.startswith('tc') and pname[2:].isdigit():
            pars[pname].latex_str = "$Tc_{" + planets_dict[int(pname[-1])]["label"] + "}$"
        elif pname.startswith('ecc') and pname[3:].isdigit():
            pars[pname].latex_str = "$e_{" + planets_dict[int(pname[-1])]["label"] + "}$"
        elif pname.startswith('w') and pname[1:].isdigit():
            pars[pname].latex_str = "$\omega_{" + planets_dict[int(pname[-1])]["label"] + "}$"
        elif pname.startswith('k') and pname[1:].isdigit():
            pars[pname].latex_str = "$K_{" + planets_dict[int(pname[-1])]["label"] + "}$"
        elif pname.startswith('sqecosw') and pname[7:].isdigit():
            pars[pname].latex_str = "$\sqrt{e} \cos{\omega}_{" + planets_dict[int(pname[-1])]["label"] + "}$"
        elif pname.startswith('sqesinw') and pname[7:].isdigit():
            pars[pname].latex_str = "$\sqrt{e} \sin{\omega}_{" + planets_dict[int(pname[-1])]["label"] + "}$"
        elif pname.startswith('cosw') and pname[7:].isdigit():
            pars[pname].latex_str = "$\cos{\omega}_{" + planets_dict[int(pname[-1])]["label"] + "}$"
        elif pname.startswith('sinw') and pname[7:].isdigit():
            pars[pname].latex_str = "$\sin{\omega}_{" + planets_dict[int(pname[-1])]["label"] + "}$"
            
        # Gammas
        elif pname.startswith('gamma') and not pname.endswith('dot'):
            pars[pname].latex_str = "$\gamma_{" + pname.split('_')[-1] + "}$"
        elif pname.startswith('gamma') and pname.endswith('_dot'):
            pars[pname].latex_str = "$\dot{\gamma}$"
        elif pname.startswith('gamma') and pname.endswith('_ddot'):
            pars[pname].latex_str = "$\ddot{\gamma}$"
        
        # Jitter
        elif pname.startswith('jitter'):
            pars[pname].latex_str = "$\sigma_{" + pname.split('_')[-1] + "}$"
            
def compute_planet_mass(per, ecc, k, mstar):
    """Computes the planet mass from the semi-amplitude equation.

    Args:
        per (float): The period of the orbit in days.
        ecc (float): The eccentricity.
        k (float): The RV semi-amplitude in m/s.
        mstar (float): The mass of the star in solar units.

    Returns:
        float: The planet mass in units of Earth masses.
    """
    MJ = 317.82838 # mass of jupiter in earth masses
    mass = k * np.sqrt(1 - ecc**2) / 28.4329 * (per / 365.25)**(1 / 3) * mstar**(2 / 3) * MJ
    return mass

def get_phases(t, per, tc):
    """Given input times, a period, and time of conjunction, returns the phase [0, 1] at each time t.
    Args:
        t (np.ndarray): The times.
        per (float): The period of the planet
        tc (float): The time of conjunction (time of transit).
    Returns:
        np.ndarray: The phases between 0 and 1
    """
    phases = (t - tc - per / 2) % per
    phases /= per
    return phases

def compute_planet_mass_deriv_mstar(per, ecc, k, mstar):
    """Computes the derivative of the semi-amplitude equation inverted for mass, d(M_planet) / d(M_Star)

    Args:
        per (float): The period of the orbit in days.
        ecc (float): The eccentricity.
        k (float): The RV semi-amplitude in m/s.
        mstar (float): The mass of the star in solar units.

    Returns:
        float: The derivative (unitless).
    """
    MJ = 317.82838 # mass of jupiter in earth masses
    alpha = k * np.sqrt(1 - ecc**2) / 28.4329 * (per / 365.25)**(1 / 3)
    dMp_dMstar = (2 / 3) * alpha * mstar**(-1 / 3)
    return dMp_dMstar