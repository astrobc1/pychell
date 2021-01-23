import optimize.scores as optscore
import optimize.kernels as optnoisekernels
import optimize.optimizers as optimizers
import optimize.knowledge as optknow
import pylatex
import corner
import optimize.frameworks as optframeworks
import plotly.subplots
import pychell.orbits.gls as gls
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

    def __init__(self, output_path, *args, star_name=None, likes=None, mstar=None, mstar_unc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.star_name = 'Star' if star_name is None else star_name
        self.scorer = likes
        self.output_path = output_path
        self.mstar = mstar
        self.mstar_unc = mstar_unc
        gen_latex_labels(self.p0, self.planets_dict)
        
    def generate_report(self, maxlike_result=None, model_comp_result=None, mcmc_result=None, n_model_pts=5000, time_offset=2450000, kernel_sampling=5):
        
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
        phases_hr_one_period = self.get_phases(t_hr_one_period, per, tc)
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
                phases_data = self.get_phases(data.t, per, tc)
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
        fig.update_layout(font=dict(size=16))
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
        
        # Add a zero line for the residuals
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=0, line=dict(color='Black'), xref='paper', yref='paper', row=2, col=1)
        
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
                    fig.add_trace(plotly.graph_objects.Scatter(x=tt, y=gpmu_hr, line=dict(width=0.8, color='black'), name=label, showlegend=False), row=1, col=1)
                    fig.add_trace(plotly.graph_objects.Scatter(x=np.concatenate([tt, tt[::-1]]),
                                             y=np.concatenate([gp_upper, gp_lower[::-1]]),
                                             fill='toself',
                                             fillcolor=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.2),
                                             name=label))
                    
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
                fig.add_trace(plotly.graph_objects.Scatter(x=tt, y=gpmu_hr, line=dict(width=0.8, color='black'), name=label, showlegend=False), row=1, col=1)
                fig.add_trace(plotly.graph_objects.Scatter(x=np.concatenate([tt, tt[::-1]]),
                                            y=np.concatenate([gp_upper, gp_lower[::-1]]),
                                            fill='toself',
                                            fillcolor=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], a=0.2),
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
                    
        # Plot the light curve as well
        #fig.add_trace(plotly.graph_objects.Scatter(x=t1 - time_offset, y=y1, line=dict(color='red'), showlegend=False), row=1, col=1)
        
                
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
        fig.update_layout(font=dict(size=16))
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
    
    @staticmethod
    def disable_planet_pars(pars, planets_dict, planet_index):
        for par in pars.values():
            for planet_par_name in planets_dict[planet_index]["basis"].names:
                if par.name == planet_par_name + str(planet_index):
                    pars[par.name].vary = False
        return pars

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
        
    def compute_rvcolor(self, pars, wave1=550, wave2=2350):
        
        # Compute difference between data < 0.3 days apart
        data_times = self.data.get_vec('t')
        data_rvs = self.data.get_vec('rv')
        data_rvs = self.scorer.like0.model.apply_offsets(data_rvs, pars)
        data_errors = self.scorer.like0.model.kernel.compute_data_errors(pars)
        rvcolor_data_t = []
        rvcolor_data_rv = []
        rvcolor_data_unc = []
        inds = self.get_rvcolor_nights(data_times, wave1, wave2)
        for i in range(len(inds)):
            rvcolor_data_t.append(np.mean(data_times[inds[i]]))
            rvcolor_data_rv.append(data_rvs[inds[i][0]] - data_rvs[inds[i][1]])
            rvcolor_data_unc.append(np.sqrt(data_errors[inds[i][0]]**2 + data_errors[inds[i][1]]**2))
            
        rvcolor_data_t = np.array(rvcolor_data_t)
        rvcolor_data_rv = np.array(rvcolor_data_rv)
        rvcolor_data_unc = np.array(rvcolor_data_unc)
        # Compute GP diffs
        residuals = self.scorer.like0.residuals_before_kernel(pars)
        rvcolor_gp, rvcolor_gp_unc = self.scorer.like0.model.kernel.compute_rv_color(pars=pars, residuals=residuals, xpred=rvcolor_data_t, wave1=wave1, wave2=wave2)
        rvcolor_result = dict(t=rvcolor_data_t, rvcolor_data_rv=rvcolor_data_rv, rvcolor_data_unc=rvcolor_data_unc, rvcolor_gp=rvcolor_gp, rvcolor_gp_unc=rvcolor_gp_unc)
        
        return rvcolor_result
    
    def plot_rvcolor(self, rvcolor_result, time_offset=2450000, plot_width=1000, plot_height=500):
        
        # Figure
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        
        # Time array
        t = rvcolor_result["t"] - time_offset
        
        # Plot data
        color_index = 0
        _yerr = dict(array=rvcolor_result["rvcolor_data_unc"])
        fig.add_trace(plotly.graph_objects.Scatter(x=t, y=rvcolor_result["rvcolor_data_rv"], name="<b>Data Color</b>", error_y=_yerr, mode='markers', marker=dict(color=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index], a=0.8))))
        color_index += 1
        
        # Plot the difference of the GPs
        _yerr = dict(array=rvcolor_result["rvcolor_gp_unc"])
        fig.add_trace(plotly.graph_objects.Scatter(x=t, y=rvcolor_result["rvcolor_gp"], line=dict(color='black', width=2), name='<b>GP Color</b>', error_y=_yerr, mode='markers', marker=dict(color=pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index], a=0.5))))
        
        # Labels
        title = "<b>GP RV Color (&#x3BB; [" + str(550) + "] - &#x3BB; [" + str(2350) + "])"
        fig.update_layout(title_text=title)
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

    def get_rvcolor_nights(self, jds, wave1, wave2, sep=0.3):
    
        # Number of spectra
        n = len(jds)
        
        wave_vec = self.scorer.like0.model.kernel.make_wave_vec()

        prev_i = 0
        # Calculate mean JD date and number of observations per night for nightly
        # coadded RV points; assume that observations on separate nights are
        # separated by at least 0.3 days.
        jds_nightly = []
        n_obs_nights = []
        inds = []
        waves = []
        instnamepairs = []
        times_vec = self.data.get_vec('t')
        for i in range(n-1):
            if jds[i+1] - jds[i] > sep:
                n_obs_night = i - prev_i + 1
                if n_obs_night != 2:
                    prev_i = i + 1
                    continue
                _inds = np.arange(prev_i, i+1).astype(int)
                if wave_vec[_inds[0]] == wave_vec[_inds[1]]:
                    prev_i = i + 1
                    continue
                if wave_vec[_inds[0]] == wave1:
                    inds.append(_inds)
                else:
                    inds.append(_inds[::-1])
                prev_i = i + 1
            
        if n - prev_i == 2 and wave_vec[-1] != wave_vec[-2]:
            _inds = inds.append(np.arange(prev_i, n).astype(int))
            if wave_vec[_inds[0]] == wave1:
                inds.append(_inds)
            else:
                inds.append(_inds[::-1])
                
        return inds

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
        phases = (t - tc - per / 2) % per
        phases /= per
        return phases
    
    @staticmethod
    def generate_all_planet_dicts(planets_dict):
        pset = pcutils.powerset(planets_dict.items())
        planet_dicts = []
        for item in pset:
            pdict = {}
            for subitem in item:
                pdict[subitem[0]] = subitem[1]
            planet_dicts.append(pdict)
        return planet_dicts
    
    @property
    def planets_dict(self):
        return self.scorer.like0.model.planets_dict
    
    def model_comparison(self, do_planets=True, do_gp=False):
        
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
                        p0 = self.disable_planet_pars(p0, planets_dict_cp, planet_index)
                
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
        aplanets = {} # In jupiter masses
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
                G = scipy.constants.gravitational_constant
                #a = G * 
                adist.append(a)
            val, unc_low, unc_high = self.sampler.chain_uncertainty(adist)
            if self.mstar_unc is not None:
                unc_low = np.sqrt(unc_low**2 + self.compute_planet_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[0]**2)
                unc_high = np.sqrt(unc_high**2 + self.compute_planet_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[1]**2)
                msiniplanets[planet_index] = (val, unc_low, unc_high)
            else:
                msiniplanets[planet_index] = (val, unc_low, unc_high)
        return msiniplanets
        
    
    def compute_planet_masses(self, sampler_result, mstar=None, mstar_unc=None):
        """Computes the value of msini for each planet.

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
                mdist.append(self.compute_planet_mass(per, ecc, k, mstar))
            val, unc_low, unc_high = self.sampler.chain_uncertainty(mdist)
            if self.mstar_unc is not None:
                unc_low = np.sqrt(unc_low**2 + self.compute_planet_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[0]**2)
                unc_high = np.sqrt(unc_high**2 + self.compute_planet_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[1]**2)
                msiniplanets[planet_index] = (val, unc_low, unc_high)
            else:
                msiniplanets[planet_index] = (val, unc_low, unc_high)
        return msiniplanets
    
    @staticmethod
    def compute_planet_mass(per, ecc, k, mstar):
        MJ = 317.82838 # mass of jupiter in earth masses
        mass = k * np.sqrt(1 - ecc**2) / 28.4329 * (per / 365.25)**(1 / 3) * mstar**(2 / 3) * MJ
        return mass
    
    @staticmethod
    def compute_planet_mass_deriv_mstar(per, ecc, k, mstar):
        MJ = 317.82838 # mass of jupiter in earth masses
        alpha = k * np.sqrt(1 - ecc**2) / 28.4329 * (per / 365.25)**(1 / 3)
        dMp_dMstar = (2 / 3) * alpha * mstar**(-1 / 3)
        return dMp_dMstar
        
    @staticmethod
    def _rv_period_search_wrapper(optprob, period, planet_index):
        p0 = optprob.p0
        p0['per' + str(planet_index)].value = period
        optprob.set_pars(p0)
        opt_result = optprob.optimize()
        return opt_result

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