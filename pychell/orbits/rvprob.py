# Base Python
import copy
import itertools
import os
import pickle

# Parallel
from joblib import Parallel, delayed
import tqdm

# Maths
import numpy as np
import scipy.constants
from PyAstronomy.pyTiming import pyPeriod

# Pychell deps
import pychell
import pychell.utils as pcutils
import pychell.orbits.planetmaths as planetmath

# Optimize
from optimize.frameworks import BayesianProblem
from optimize.noise import CorrelatedNoiseProcess, GaussianProcess

# Plots
import corner
import matplotlib.pyplot as plt
import plotly.subplots
import plotly.graph_objects
PLOTLY_COLORS = pcutils.PLOTLY_COLORS
COLORS_HEX_GADFLY = pcutils.COLORS_HEX_GADFLY
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

class RVProblem(BayesianProblem):
    """The primary, top-level container for Exoplanet optimization problems. As of now, this only deals with RV data. Photometric modeling will be included in future updates, but will leverage existing libraries (Batman, etc.).
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, output_path=None, p0=None, post=None, optimizer=None, sampler=None, star_name=None, tag=None):
        """Constructs the primary exoplanet problem object.

        Args:
            output_path (The output path for plots and pickled objects, optional): []. Defaults to this current working direcrory.
            p0 (Parameters, optional): The initial parameters. Defaults to None.
            optimizer (Optimizer, optional): The max like optimizer. Defaults to None.
            sampler (Sampler, optional): The MCMC sampler object. Defaults to None.
            post (RVPosterior, optional): The composite likelihood object. Defaults to None.
            star_name (str, optional): The name of the star, may contain spaces. Defaults to None.
        """
        
        # Pass relevant items to base class constructor
        super().__init__(p0=p0, optimizer=optimizer, sampler=sampler, post=post)
        
        # The output path
        self.output_path = output_path
        
        # The tag of this run for filenames
        self.tag = "" if tag is None else tag
        
        # Star name
        self.star_name = 'Star' if star_name is None else star_name
        
        # Full tag for filenames
        self.full_tag = f"{self.star_name}_{self.tag}"
        
        # Generate latex labels for the parameters.
        self._gen_latex_labels()
        
        # Make a color map for plotting.
        self._make_color_map()
        
    def _make_color_map(self):
        self.color_map = {}
        color_index = 0
        for like in self.post.values():
            for instname in like.model.data:
                if instname not in self.color_map:
                    self.color_map[instname] = COLORS_HEX_GADFLY[color_index % len(COLORS_HEX_GADFLY)]
                    color_index += 1
        
        
    def _gen_latex_labels(self):
        planets_dict = self.planets_dict
    
        for par in self.p0.values():
        
            pname = par.name
        
            # Planets (per, tc, k, ecc, w, sqecosw, sqesinw, other bases added later if necessary)
            if pname.startswith('per') and pname[3:].isdigit():
                ii = int(pname[-1])
                par.latex_str = "$P_{" + planets_dict[ii]["label"] + "}$"
            elif pname.startswith('tc') and pname[2:].isdigit():
                ii = int(pname[-1])
                par.latex_str = "$Tc_{" + planets_dict[ii]["label"] + "}$"
            elif pname.startswith('ecc') and pname[3:].isdigit():
                ii = int(pname[-1])
                par.latex_str = "$e_{" + planets_dict[ii]["label"] + "}$"
            elif pname.startswith('w') and pname[1:].isdigit():
                ii = int(pname[-1])
                par.latex_str = "$\omega_{" + planets_dict[ii]["label"] + "}$"
            elif pname.startswith('k') and pname[1:].isdigit():
                ii = int(pname[-1])
                par.latex_str = "$K_{" + planets_dict[ii]["label"] + "}$"
            elif pname.startswith('sqecosw') and pname[7:].isdigit():
                ii = int(pname[-1])
                par.latex_str = "$\sqrt{e} \cos{\omega}_{" + planets_dict[ii]["label"] + "}$"
            elif pname.startswith('sqesinw') and pname[7:].isdigit():
                ii = int(pname[-1])
                par.latex_str = "$\sqrt{e} \sin{\omega}_{" + planets_dict[ii]["label"] + "}$"
            elif pname.startswith('cosw') and pname[7:].isdigit():
                ii = int(pname[-1])
                par.latex_str = "$\cos{\omega}_{" + planets_dict[ii]["label"] + "}$"
            elif pname.startswith('sinw') and pname[7:].isdigit():
                ii = int(pname[-1])
                par.latex_str = "$\sin{\omega}_{" + planets_dict[ii]["label"] + "}$"
                
            # Gammas
            elif pname.startswith('gamma') and not pname.endswith('dot'):
                par.latex_str = "$\gamma_{" + pname.split('_')[-1] + "}$"
            elif pname.startswith('gamma') and pname.endswith('_dot'):
                par.latex_str = "$\dot{\gamma}$"
            elif pname.startswith('gamma') and pname.endswith('_ddot'):
                par.latex_str = "$\ddot{\gamma}$"
        
            # Jitter
            elif pname.startswith('jitter'):
                par.latex_str = "$\sigma_{" + pname.split('_')[-1] + "}$"

    
    #######################################
    #### STANDARD OPTIMIZATION METHODS ####
    #######################################
    
    def run_mcmc(self, *args, save=True, **kwargs):
        """Runs the mcmc.

        Returns:
            *args: Any args.
            **kwargs: Any keyword args.
        
        Returns:
            dict: A dictionary with the mcmc results.
        """
        mcmc_result = super().run_mcmc(*args, **kwargs)
        if save:
            fname = self.output_path + self.star_name.replace(' ', '_') + '_mcmc_results_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(mcmc_result, f)
        return mcmc_result
    
    def run_mapfit(self, save=True):
        """Runs the optimizer.
            
        Returns:
            dict: A dictionary with the optimize results.
        """
        self.initialize()
        map_result = self.optimizer.optimize()
        if save:
            fname = f"{self.output_path}{self.star_name.replace(' ', '_')}_map_results_{pcutils.gendatestr(time=True)}_{self.tag}.pkl"
            with open(fname, 'wb') as f:
                pickle.dump(map_result, f)
        return map_result
    
    ###################################
    #### Standard Plotting Methods ####
    ###################################
    
    def plot_phased_rvs(self, planet_index, pars=None, plot_width=1000, plot_height=600, save=True):
        """Creates a phased rv plot for a given planet with the model on top. An html figure is saved with a unique filename.

        Args:
            planet_index (int): The planet index.
            pars (Parameters, optional): The parameters to use. Defaults to self.p0.
            plot_width (int, optional): The plot width, in pixels.
            plot_height (int, optional): The plot height, in pixels.

        Returns:
            plotly.figure: A plotly figure containing the plot. The figure is also saved.
        """
        
        # Resolve which pars to use
        if pars is None:
            pars = self.p0
            
        # Creat a plotly figure
        fig = plotly.subplots.make_subplots(rows=1, cols=1)
        
        # Convert parameters to standard basis and compute tc for consistent plotting
        per, tp, ecc, w, k = self.planets_dict[planet_index]["basis"].to_standard(pars)
        tc = planetmath.tp_to_tc(tp, per, ecc, w)
        
        # A high res time grid with an arbitrary starting point [BJD]
        t_hr_one_period = np.linspace(tc, tc + per, num=500)
        
        # Convert grid to phases [0, 1]
        phases_hr_one_period = planetmath.get_phases(t_hr_one_period, per, tc)
        
        # Build high res model for this planet
        planet_model_phased = self.like0.model.build_planet(pars, t_hr_one_period, planet_index)
        
        # Sort the phased model
        ss = np.argsort(phases_hr_one_period)
        phases_hr_one_period = phases_hr_one_period[ss]
        planet_model_phased = planet_model_phased[ss]
        
        # Store the data in order to bin the phased RVs.
        phases_data_all = np.array([], dtype=float)
        rvs_data_all = np.array([], dtype=float)
        unc_data_all = np.array([], dtype=float)
        
        # Loop over likes
        for like in self.post.values():
            
            # Compute the final residuals
            residuals = like.model.compute_residuals(pars)
            
            # Compute the noise model
            if isinstance(like.model.noise_process, CorrelatedNoiseProcess):
                errors = like.model.compute_data_errors(pars, include_corr_error=True)
            else:
                errors = like.model.compute_data_errors(pars)
            
            # Loop over instruments and plot each
            for data in like.model.data.values():
                errors_arr = errors[like.model.data.indices[data.label]]
                data_arr = residuals[like.model.data.indices[data.label]] + like.model.build_planet(pars, data.t, planet_index)
                phases_data = planetmath.get_phases(data.t, per, tc)
                fig.add_trace(plotly.graph_objects.Scatter(x=phases_data, y=data_arr,
                                                           error_y=dict(array=errors_arr),
                                                           name=f"<b>{data.label}</b>",
                                                           mode='markers',
                                                           marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.8))))
                
                # Store for binning at the end
                phases_data_all = np.concatenate((phases_data_all, phases_data))
                rvs_data_all = np.concatenate((rvs_data_all, data_arr))
                unc_data_all = np.concatenate((unc_data_all, errors_arr))

        # Plot the model on top
        fig.add_trace(plotly.graph_objects.Scatter(x=phases_hr_one_period, y=planet_model_phased,
                                                   line=dict(color='black', width=2),
                                                   name="<b>Keplerian Model</b>"))
        
        # Plot the the binned data.
        ss = np.argsort(phases_data_all)
        phases_data_all, rvs_data_all, unc_data_all = phases_data_all[ss], rvs_data_all[ss], unc_data_all[ss]
        phases_binned, rvs_binned, unc_binned = planetmath.bin_phased_rvs(phases_data_all, rvs_data_all, unc_data_all, nbins=10)
        fig.add_trace(plotly.graph_objects.Scatter(x=phases_binned,
                                                   y=rvs_binned,
                                                   error_y=dict(array=unc_binned),
                                                   mode='markers', marker=dict(color='Maroon', size=12, line=dict(width=2, color='DarkSlateGrey')),
                                                   showlegend=False))
        
        # Labels
        fig.update_xaxes(title_text='<b>Phase</b>')
        fig.update_yaxes(title_text='<b>RV [m/s]</b>')
        fig.update_layout(title=f"<b>{self.star_name} {self.like0.model.planets_dict[planet_index]['label']} <br> P = {round(per, 6)}, K = {round(k, 2)} e = {round(ecc, 3)}</b>")
        fig.update_layout(template="plotly_white")
        fig.update_layout(font=dict(size=16))
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_layout(width=plot_width, height=plot_height)
        if save:
            fig.write_html(f"{self.output_path}{self.star_name.replace(' ', '_')}{self.planets_dict[planet_index]['label']}_rvs_phased_ {pcutils.gendatestr(time=True)}_{self.tag}.html")
        
        # Return fig
        return fig
    
    def plot_phased_rvs_all(self, pars=None, plot_width=1000, plot_height=600, save=True):
        """Wrapper to plot the phased RV model for all planets.

        Args:
            pars (Parameters, optional): The parameters to use. Defaults to self.p0.
        
        Returns:
            list: A list of Plotly figures. The figures are also saved.
        """
        
        # Default parameters
        if pars is None:
            pars = self.p0
        
        plots = []
        for planet_index in self.planets_dict:
            plot = self.plot_phased_rvs(planet_index, pars=pars, plot_width=plot_width, plot_height=plot_height, save=save)
            plots.append(plot)
        return plots
        
    def plot_full_rvs(self, pars=None, ffp=None, n_model_pts=5000, time_offset=2450000, kernel_sampling=500, kernel_window=10, plot_width=1800, plot_height=1200, save=True):
        """Creates an rv plot for the full dataset and rv model.

        Args:
            pars (Parameters, optional): The parameters to use. Defaults to self.p0.
            n_model_pts (int, optional): The number of points for the densly sampled Keplerian model.
            time_offset (float, optional): The time to subtract from the times.
            kernel_sampling (int, optional): The number of points per period to sample for the correlated noise kernel. If there is no noise kernel, this argument is irrelevant.
            plot_width (int, optional): The plot width in pixels. Defaults to 1800.
            plot_height (int, optional): The plot width in pixels. Defaults to 1200.

        Returns:
            plotly.figure: A Plotly figure. The figure is also saved.
        """
        
        # Resolve which pars to use
        if pars is None:
            pars = self.p0
            
        # Create a figure
        fig = plotly.subplots.make_subplots(rows=2, cols=1)
        
        # Get the high resolution Keplerian model + trend
        if self.like0.model.n_planets > 0 or self.like0.model.trend_model.poly_order > 0:
            
            # Generate a high resolution data grid.
            t_data_all = self.data_t
            t_start, t_end = np.nanmin(t_data_all), np.nanmax(t_data_all)
            dt = t_end - t_start
            t_hr = np.linspace(t_start - dt / 100, t_end + dt / 100, num=n_model_pts)
        
            # Generate the high res Keplerian + Trend model.
            model_arr_hr = self.like0.model.builder(pars, t_hr)
            
            # Label
            if self.like0.model.n_planets > 0 and self.like0.model.trend_model.poly_order > 0:
                name = "<b>Keplerian + Trend Model</b>"
            elif self.like0.model.n_planets > 0 and self.like0.model.trend_model.poly_order == 0:
                name = "<b>Keplerian Model</b>"
            else:
                name = "<b>Trend Model</b>"

            # Plot the planet model
            fig.add_trace(plotly.graph_objects.Scatter(x=t_hr - time_offset, y=model_arr_hr,
                                                       name=name,
                                                       line=dict(color='black', width=2)),
                          row=1, col=1)
        
        # Loop over likes and:
        # 1. Plot high res GP if present
        # 2. Plot data
        color_index = 0
        for like in self.post.values():

            # Correlated Noise
            if isinstance(like.model.noise_process, CorrelatedNoiseProcess):
                
                # Make hr arrays for GP
                time_gp_hr = np.array([], dtype=float)
                gps_hr = {}
                residuals_raw = like.model.compute_raw_residuals(pars)
                for i in range(like.model.data.t.size):
                    
                    # Create a time array centered on this point
                    t_hr_window = np.linspace(like.model.data.t[i] - kernel_window,
                                           like.model.data.t[i] + kernel_window,
                                           num=kernel_sampling)
                    
                    # Sample GP for this window
                    _gps_hr = like.model.noise_process.compute_noise_components(pars=pars, linpred=residuals_raw, xpred=t_hr_window)
                    for noise_label in _gps_hr:
                        if i == 0:
                            gps_hr[noise_label] = np.array([], dtype=float), np.array([], dtype=float)
                        gps_hr[noise_label] = np.concatenate((gps_hr[noise_label][0], _gps_hr[noise_label][0])), np.concatenate((gps_hr[noise_label][1], _gps_hr[noise_label][1]))
                                              
                    # Store
                    time_gp_hr = np.concatenate((time_gp_hr, t_hr_window))
                    
                # Sort each
                ss = np.argsort(time_gp_hr)
                time_gp_hr = time_gp_hr[ss]
                for noise_label in gps_hr:
                    gps_hr[noise_label] = gps_hr[noise_label][0][ss], gps_hr[noise_label][1][ss]
                    
                
                # Compute data errors
                errors = like.model.noise_process.compute_data_errors(pars, include_corr_error=False, linpred=residuals_raw)
                
                # Plot each GP
                for noise_label in gps_hr:
                    
                    # Get the relevant variables
                    tt = time_gp_hr - time_offset
                    gp, gp_unc = gps_hr[noise_label][0], gps_hr[noise_label][1]
                    gp_lower, gp_upper = gp - gp_unc, gp + gp_unc
                    
                    # For legend
                    label = f"<b>{noise_label.replace('_', ' ')}</b>"
                        
                    # Plot the actual GP
                    for instname in like.model.data:
                        if instname in noise_label:
                            _instname = instname
                            break
                    fig.add_trace(plotly.graph_objects.Scatter(x=tt, y=gp,
                                                                line=dict(width=0.8, color=pcutils.hex_to_rgba(self.color_map[_instname], a=0.6)),
                                                                name=label, showlegend=False),
                                    row=1, col=1)
                    
                    # Plot the gp unc
                    fig.add_trace(plotly.graph_objects.Scatter(x=np.concatenate([tt, tt[::-1]]),
                                                                y=np.concatenate([gp_upper, gp_lower[::-1]]),
                                                                fill='toself',
                                                                line=dict(color=pcutils.hex_to_rgba(self.color_map[_instname], a=0.6), width=1),
                                                                fillcolor=pcutils.hex_to_rgba(self.color_map[_instname], a=0.5),
                                                                name=label, showlegend=True),
                                    row=1, col=1)
                    color_index += 1
                
                # Generate the residuals without noise
                residuals = like.model.compute_residuals(pars)
                
                # Plot each instrument
                for data in like.model.data.values():
                    
                    # Raw data - zero point
                    data_arr = data.rv - pars[f"gamma_{data.label}"].value
                    
                    # Errors
                    errors_arr = errors[like.model.data.indices[data.label]]
                    
                    # Final residuals
                    residuals_arr = residuals[like.model.data.indices[data.label]]
                    
                    # Plot rvs
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=data_arr,
                                                               error_y=dict(array=errors_arr),
                                                               name=data.label,
                                                               mode='markers',
                                                               marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14)),
                                  row=1, col=1)
                    
                    # Plot residuals
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=residuals_arr,
                                                               error_y=dict(array=errors_arr),
                                                               mode='markers',
                                                               marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14), showlegend=False),
                                  row=2, col=1)
            

            else:
                
                # Generate the residuals
                residuals = like.model.compute_residuals(pars)
                
                # Compute data errors
                errors = like.model.noise_process.compute_data_errors(pars)
                
                for data in like.model.data.values():
                    
                    # Raw data - zero point
                    data_arr = data.rv - pars[f"gamma_{data.label}"].value
                    
                    # Errors
                    errors_arr = errors[like.model.data.indices[data.label]]
                    
                    # Final residuals
                    residuals_arr = residuals[like.model.data.indices[data.label]]
                    
                    # Plot rvs
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=data_arr,
                                                               error_y=dict(array=errors_arr),
                                                               name=data.label,
                                                               mode='markers',
                                                               marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14)),
                                  row=1, col=1)
                    
                    # Plot residuals
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=residuals_arr,
                                                               error_y=dict(array=errors_arr),
                                                               mode='markers',
                                                               marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14), showlegend=False),
                                  row=2, col=1)

        # Labels
        fig.update_xaxes(title_text=f"<b>BJD - {time_offset}</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Residual RV [m/s]</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>RV [m/s]</b>", row=1, col=1)
        fig.update_xaxes(tickprefix="<b>", ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>", ticksuffix ="</b><br>")
        
        # Appearance
        fig.update_layout(template="plotly_white")
        fig.update_layout(font=dict(size=20))
        fig.update_layout(width=plot_width, height=plot_height)
        
        # Save
        if save:
            fig.write_html(f"{self.output_path}{self.star_name.replace(' ', '_')}_rvs_full_{pcutils.gendatestr(time=True)}_{self.tag}.html")
        
        # Return the figure
        return fig
        
    def corner_plot(self, mcmc_result, save=True):
        """Constructs a corner plot.

        Args:
            mcmc_result (dict): The mcmc result

        Returns:
            fig: A matplotlib figure.
        """
        plt.clf()
        pbest_vary_dict = mcmc_result["pmed"].unpack(vary_only=True)
        truths = pbest_vary_dict["value"]
        labels = [par.latex_str for par in mcmc_result["pbest"].values() if par.vary]
        corner_plot = corner.corner(mcmc_result["chains"], labels=labels, truths=truths, show_titles=True)
        if save:
            corner_plot.savefig(self.output_path + self.star_name.replace(' ', '_') + '_corner_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.png')
        return corner_plot
        
        
    ######################
    #### COMPONENTS ######
    ######################
    
    def compute_components(self, pars):
        
        comps = {}
        for like in self.post.values():
            
            # Time
            comps[f"{like.label}_data_t"] = like.model.data.t
            
            # Errors
            comps[f"{like.label}_data_rverr"] = like.model.data.rverr
            
            # Zero point corrected data
            comps[f"{like.label}_data_rv"] = like.model.data.rv - like.model.trend_model.build_trend_zero(pars, t=like.model.data.t)
            
            # Global trend
            if like.model.trend_model.poly_order > 0:
                comps[f"{like.label}_global_trend"] = like.model.trend_model.build_global_trend(pars, t=like.model.data.t)
            
            # GPs
            residuals_raw = like.model.compute_raw_residuals(pars)
            comps[f"{like.label}_gps"] = like.model.noise_process.compute_noise_components(pars=pars, linpred=residuals_raw)
            
            # Planets
            for planet_index in like.model.planets_dict:
                comps[f"{like.label}_planet_{planet_index}"] = like.model.build_planet(pars, like.model.data.t, planet_index)
                
        return comps

    #################################
    #### BRUTE FORCE PERIODOGRAM ####
    #################################
            
    def brute_force_periodogram(self, periods, planet_index=1, n_cores=1):
        # Run in parallel
        results = Parallel(n_jobs=n_cores, verbose=0, batch_size=1)(delayed(self._brute_force_wrapper)(self, periods[i], planet_index) for i in tqdm.tqdm(range(len(periods))))
        return results

            
    ##################
    #### RV Color ####
    ##################
    
    def full_rvcolor(self, pars=None, sep=0.3, time_offset=2450000, plot_width=1000, plot_height=600):
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
            
        wave_vec = self.like0.data.gen_wave_vec()
        wave_pairs = self._generate_all_wave_pairs(wave_vec)
        rvcolor_results = []
        plots_time = []
        for wave_pair in wave_pairs:
            rvcolor_result = self.compute_rvcolor(pars, wave_pair[0], wave_pair[1], sep=sep)
            p = self.plot_rvcolor(wave_pair[0], wave_pair[1], rvcolor_result, time_offset=time_offset, plot_width=plot_width, plot_height=plot_height)
            rvcolor_results.append(rvcolor_result)
            plots_time.append(p)
            
        plot_11 = self.plot_rvcolor_one_to_one(pars=pars, plot_width=plot_width, plot_height=plot_height, sep=sep)

        return rvcolor_results, plots_time, plot_11
    
    def plot_rvcolor(self, wave1, wave2, rvcolor_result, time_offset=2450000, plot_width=1000, plot_height=600):
        
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
            fig.add_trace(plotly.graph_objects.Scatter(x=np.array(t[i]), y=np.array(rvcolor_result["rvcolor_data"][i]), name=label_map[rvcolor_result["instnames"][i]], error_y=yerr, mode='markers', marker=dict(color=color_map[rvcolor_result["instnames"][i]], size=12), showlegend=show_legend))
        
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
        title = "<b>RV Color (&#x3BB; [" + str(int(wave1)) + " nm] - &#x3BB; [" + str(int(wave2)) + " nm])</b>"
        fig.update_layout(title_text=title)
        fig.update_xaxes(title_text='<b>BJD - ' + str(time_offset) + '</b>')
        fig.update_yaxes(title_text='<b>RVs [m/s]</b>')
        fig.update_yaxes(title_text='<b>RV Color [m/s]</b>')
        fig.update_layout(template="plotly_white")
        fig.update_layout(font=dict(size=16))
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_layout(width=plot_width, height=plot_height)
        fig.write_html(self.output_path + self.star_name.replace(' ', '_') + '_rvcolor_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.html')
        
        return fig

    def plot_rvcolor_one_to_one(self, pars=None, plot_width=1000, plot_height=600, sep=0.3):
        """Plots the data RV color vs. the GP RV color.

        Args:
            pars (Parameters, optional): The parameters to use in order to remove the polynomial trend. Defaults to self.p0.
            plot_width (int, optional): The plot width in pixels. Defaults to 1000.
            plot_height (int, optional): The plot height in pixels. Defaults to 500.
            sep (float, optional): The max separation in time to be considered simultaneous (in days). Defaults to 0.3.

        Returns:
            plotly.figure: The plotly figure.
        """
        
        # Pars
        if pars is None:
            pars = self.p0
            
        # Figure
        fig = plotly.subplots.make_subplots(rows=1, cols=1)

        # Compute all possible wave pairs
        wave_vec = self.like0.data.gen_wave_vec()
        wave_pairs = self._generate_all_wave_pairs(wave_vec)
        
        # Add a 1-1 line
        x_line = np.arange(-150, 150, 1)
        fig.add_trace(plotly.graph_objects.Scatter(x=x_line, y=x_line, line=dict(color='black', dash='dash'), showlegend=False))
        
        # Loop over wave pairs and
        # 1. Compute data rv color
        # 2. Compute GP color
        color_index = 0
        tt = np.array([], dtype=float)
        yy1 = np.array([], dtype=float)
        yy2 = np.array([], dtype=float)
        yyunc1 = np.array([], dtype=float)
        yyunc2 = np.array([], dtype=float)
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
            tt = np.concatenate((tt, t))
            yy1 = np.concatenate((yy1, x))
            yy2 = np.concatenate((yy2, y))
            yyunc1 = np.concatenate((yyunc1, x_unc))
            yyunc2 = np.concatenate((yyunc2, y_unc))
            for i in range(len(t)):
            
                # The yerr for the data and GP
                _x_unc = dict(array=np.array(x_unc[i]))
                _y_unc = dict(array=np.array(y_unc[i]))
            
                # Check if this combo is in color_map
                show_legend = False
                alpha = 0.8
                if rvcolor_result["instnames"][i] not in color_map:
                    instname1, instname2 = rvcolor_result["instnames"][i]
                    if instname1 in ["CARM-Vis", "CARM-NIR"] and instname2 in ["CARM-Vis", "CARM-NIR"]:
                        alpha = 0.2
                    if instname1 in ["iSHELL", "SPIRou"] and instname2 in ["iSHELL", "SPIRou"]:
                        continue
                    color_map[rvcolor_result["instnames"][i]] = pcutils.csscolor_to_rgba(PLOTLY_COLORS[color_index], a=alpha)
                    label_map[rvcolor_result["instnames"][i]] = "<b>RV Color, " + instname1 + " - " + instname2 + "</b>"
                    show_legend = True
                    color_index += 1
                fig.add_trace(plotly.graph_objects.Scatter(x=np.array(x[i]), y=np.array(y[i]), name=label_map[rvcolor_result["instnames"][i]], error_x=_x_unc, error_y=_y_unc, mode='markers', marker=dict(color=color_map[rvcolor_result["instnames"][i]], size=12), showlegend=show_legend))
                
        # Labels
        title = "<b>RV Color (&#x3BB; [" + str(550) + "] - &#x3BB; [" + str(2350) + "])</b>"
        fig.update_layout(title_text=title)
        fig.update_xaxes(title_text='<b>Data Color [m/s]</b>')
        fig.update_yaxes(title_text='<b>GP Color [m/s]</b>')
        fig.update_layout(title='<b>' + self.star_name + ' Chromaticity')
        fig.update_layout(template="plotly_white")
        fig.update_layout(font=dict(size=16))
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_layout(width=plot_width, height=plot_height)
        fig.write_html(self.output_path + self.star_name.replace(' ', '_') + '_rvcolor_1_to_1_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.html')
        
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
        wave_vec = self.like0.data.gen_wave_vec()
        times_vec = self.like0.data.get_vec('t')
        rv_vec = self.like0.data.get_vec('rv')
        rv_vec = self.like0.model.apply_offsets(rv_vec, pars)
        unc_vec = self.like0.noise.compute_data_errors(pars, include_gp_error=False, data_with_noise=None, gp_error=None)
        tel_vec = self.like0.data.gen_instname_vec()
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
        rv_data1 = []
        rv_data2 = []
        rv_unc_data1 = []
        rv_unc_data2 = []
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
                rv_data1.append(rv_vec[inds][ind1])
                rv_data2.append(rv_vec[inds][ind2])
                rv_unc_data1.append(unc_vec[inds][ind1])
                rv_unc_data2.append(unc_vec[inds][ind2])
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
            rv_data1.append(rv_vec[inds][ind1])
            rv_data2.append(rv_vec[inds][ind2])
            rv_unc_data1.append(unc_vec[inds][ind1])
            rv_unc_data2.append(unc_vec[inds][ind2])
            jds_avg.append(np.mean(times_vec[inds]))
            rvcolor_data.append(rv_vec[inds][ind1] - rv_vec[inds][ind2])
            unccolor_data.append(np.sqrt(unc_vec[inds][ind1]**2 + unc_vec[inds][ind2]**2))
            instnames.append((tel_vec[inds][ind1], tel_vec[inds][ind2]))
         
        # Convert to numpy arrays
        jds_avg = np.array(jds_avg)
        rv_data1 = np.array(rv_data1)
        rv_data2 = np.array(rv_data2)
        rv_unc_data1 = np.array(rv_unc_data1)
        rv_unc_data2 = np.array(rv_unc_data2)
        rvcolor_data = np.array(rvcolor_data)
        unccolor_data = np.array(unccolor_data)
         
        # Now for the GP color.
        # Residuals with noise
        residuals_with_noise = self.like0.compute_data_pre_noise_process(pars)
        
        # Compute the coarsely sampled GP for each wavelength.
        gp_mean1_data, gp_stddev1_data = self.like0.noise.realize(pars, residuals_with_noise, xpred=jds_avg, return_gp_error=True, wavelength=wave1)
        gp_mean2_data, gp_stddev2_data = self.like0.noise.realize(pars, residuals_with_noise, xpred=jds_avg, return_gp_error=True, wavelength=wave2)
        
        # Compute the coarsely sampled GP-color and unc
        gp_color_data = gp_mean1_data - gp_mean2_data
        gp_color_unc_data = np.sqrt(gp_stddev1_data**2 + gp_stddev2_data**2)
        
        # Compute the densely sampled GP-color
        t_gp_hr, gpmean1_hr, gpstddev1_hr = self.gp_smart_sample(pars, self.like0, s=pars["gp_per"].value*2, t=jds_avg, data_with_noise=residuals_with_noise, sampling=200, return_gp_error=True, wavelength=wave1)
        _, gpmean2_hr, gpstddev2_hr = self.gp_smart_sample(pars, self.like0, s=pars["gp_per"].value*2, t=jds_avg, data_with_noise=residuals_with_noise, sampling=200, return_gp_error=True, wavelength=wave2)
        
        gpmean_color_hr = gpmean1_hr - gpmean2_hr
        gpstddev_color_hr = np.sqrt(gpstddev1_hr**2 + gpstddev2_hr**2)
                
        # Return a mega dictionary
        out = dict(jds_avg=jds_avg, rv_data1=rv_data1, rv_data2=rv_data2, rv_unc_data1=rv_unc_data1, rv_unc_data2=rv_unc_data2, rvcolor_data=rvcolor_data, unccolor_data=unccolor_data, gp_color_data=gp_color_data, gp_color_unc_data=gp_color_unc_data, t_gp_hr=t_gp_hr, gpmean1_hr=gpmean1_hr, gpstddev1_hr=gpstddev1_hr, gpmean2_hr=gpmean2_hr, gpstddev2_hr=gpstddev2_hr, gpmean_color_hr=gpmean_color_hr, gpstddev_color_hr=gpstddev_color_hr, instnames=instnames, wave1=wave1, wave2=wave2)
        
        return out
    
    
    def get_simult_obs(self, pars=None, sep=0.3):
        
        # Parameters
        if pars is None:
            pars = self.p0
            
        # Filter data to only consider each wavelength
        wave_vec = self.obj.like0.noise.make_wave_vec()
        tel_vec = self.data.gen_instname_vec()
        data_t = self.data.get_vec('t')
        data_rvs = self.data.get_vec('rv') # Offset rvs
        data_rvs = self.like0.model.apply_offsets(data_rvs, pars)
        data_rvs_unc = self.like0.kernel.compute_data_errors(pars, include_white_error=True, include_kernel_error=False)
        
        # Loop over RVs and look for near-simultaneous RV color observations.
        prev_i = 0
        waves_nights = []
        data_t_nights = []
        data_rv_nights = []
        data_rv_unc_nights = []
        instnames_nights = []
        for i in range(len(data_t) - 1):
            
            # If dt > sep, we have moved to the next night.
            # But first ook at all RVs from this night
            if data_t[i+1] - data_t[i] > sep:
                
                # The number of RVs on this night for these two wavelengths.
                n_obs_night = i - prev_i + 1
                
                # If only 1 observation for this night, skipi.
                if n_obs_night < 2:
                    prev_i = i + 1
                    continue
                
                # The indices for this night, relative to the filtered arrays 
                inds = np.arange(prev_i, i + 1).astype(int)
                
                # Nightly info
                waves_nights.append(wave_vec[inds])
                instnames_nights.append(tel_vec[inds])
                data_t_nights.append(data_t[inds])
                data_rv_nights.append(data_rvs[inds])
                data_rv_unc_nights.append(data_rvs_unc[inds])
                
                # Move on.
                prev_i = i + 1
                
        # Last night.
        inds = np.arange(prev_i, len(data_t)).astype(int)
        waves_nights.append(wave_vec[inds])
        instnames_nights.append(tel_vec[inds])
        data_t_nights.append(data_t[inds])
        data_rv_nights.append(data_rvs[inds])
        data_rv_unc_nights.append(data_rvs_unc[inds])
        
        return {"waves_nights": waves_nights, "data_t_nights": data_t_nights, "data_rv_nights": data_rv_nights, "data_rv_unc_nights": data_rv_unc_nights, "data_rv_unc_nights": data_rv_unc_nights, "instnames_nights": instnames_nights}
    
    
    ##########################
    #### MODEL COMPARISON ####
    ##########################
    
    def model_comparison(self, save=True):
        """Runs a model comparison for all combinations of planets.

        Returns:
            list: Each entry is a dict containing the model comp results for each case, and is sorted according to the small sample AIC.
        """
            
        # Store results in a list
        mc_results = []
        
        # Alias like0
        like0 = self.post.like0
        
        # Original planets dict
        planets_dict_cp = copy.deepcopy(like0.model.planets_dict)
        
        # Get all planet combos through a powerset
        planet_dicts = self._generate_all_planet_dicts(like0.model.planets_dict)
        
        # Loop over combos
        for i, planets_dict in enumerate(planet_dicts):
            
            # Copy self
            _optprob = copy.deepcopy(self)
            
            # Alias pars
            p0 = _optprob.p0
            
            # Remove all other planets except this combo.
            for planet_index in planets_dict_cp:
                if planet_index not in planets_dict:
                    self.like0.model.kep_model._disable_planet_pars(p0, planets_dict_cp, planet_index)
            
            # Set planets dict for each model
            for like in _optprob.post.values():
                like.model.kep_model.planets_dict = planets_dict
            
            # Pars
            _optprob.set_pars(p0)

            # Run the max like
            opt_result = _optprob.run_mapfit(save=False)
            
            # Alias best fit params
            pbest = opt_result['pbest']
            
            # Recompute the max like to NOT include any priors to keep things consistent.
            lnL = _optprob.post.compute_logL(pbest)
            
            # Run the BIC
            bic = _optprob.optimizer.obj.compute_bic(pbest)
            
            # Run the AICc
            aicc = _optprob.optimizer.obj.compute_aicc(pbest)
            
            # Red chi 2
            try:
                redchi2 = _optprob.optimizer.obj.compute_redchi2(pbest, include_uncorr_error=True)
            except:
                redchi2 = _optprob.optimizer.obj.compute_redchi2(pbest)
            
            # Store
            mc_results.append({'planets_dict': planets_dict, 'lnL': lnL, 'bic': bic, 'aicc': aicc, 'pbest': pbest, 'redchi2': redchi2})
            
            del _optprob
            
        # Get the aicc and bic vals for each model
        aicc_vals = np.array([mcr['aicc'] for mcr in mc_results], dtype=float)
        
        # Sort according to aicc val (smaller is "better")
        ss = np.argsort(aicc_vals)
        mc_results = [mc_results[ss[i]] for i in range(len(ss))]
        
        # Grab the aicc and bic vals again
        aicc_vals = np.array([mcr['aicc'] for mcr in mc_results], dtype=float)
        bic_vals = np.array([mcr['bic'] for mcr in mc_results], dtype=float)
        
        # Compute aicc and bic vals
        aicc_diffs = np.abs(aicc_vals - np.nanmin(aicc_vals))
        bic_diffs = np.abs(bic_vals - np.nanmin(bic_vals))
        
        # Store diffs
        for i, mcr in enumerate(mc_results):
            mcr['delta_aicc'] = aicc_diffs[i]
            mcr['delta_bic'] = bic_diffs[i]
    
        # Save
        if save:
            fname = f"{self.output_path}{self.star_name.replace(' ', '_')}_modelcomp_{pcutils.gendatestr(time=True)}_{self.tag}.pkl"
            with open(fname, 'wb') as f:
                pickle.dump(mc_results, f)
        
        return mc_results

    def compute_masses(self, mcmc_result, mstar, mstar_unc=None):
        """Computes the planet masses and uncertainties in Earth Masses.

        Args:
            mcmc_result (dict): The MCMC result.
            mstar (float): The mass of the star in solar units.
            mstar_unc (list of floats): The lower and upper uncertainty in mstar (both positive).

        Returns:
            dict: A dictionary with keys identical to the planets dictionary and values containing a tuple with the mass, lower uncertainty, and upper uncertainty in Earth units.
        """
        
        if mstar_unc is None:
            mstar_unc = (0, 0)
        
        msiniplanets = {} # In earth masses
        for planet_index in self.planets_dict:
            perdist = []
            tpdist = []
            eccdist = []
            wdist = []
            kdist = []
            mdist = []
            pars = copy.deepcopy(mcmc_result["pmed"])
            for i in range(mcmc_result["n_steps"]):
                for pname in self.planets_dict[planet_index]["basis"].pnames:
                    if pars[pname].vary:
                        ii = pars.index_from_par(pname, rel_vary=True)
                        pars[pname].value = mcmc_result["chains"][i, ii]
                per, tp, ecc, w, k = self.planets_dict[planet_index]["basis"].to_standard(pars)
                perdist.append(per)
                tpdist.append(tp)
                eccdist.append(ecc)
                wdist.append(w)
                kdist.append(k)
                mdist.append(planetmath.compute_mass(per, ecc, k, mstar))
            val, unc_low, unc_high = self.sampler.chain_uncertainty(mdist)
            if mstar_unc is not None:
                unc_low = np.sqrt(unc_low**2 + planetmath.compute_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[0]**2)
                unc_high = np.sqrt(unc_high**2 + planetmath.compute_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[1]**2)
                msiniplanets[planet_index] = (val, unc_low, unc_high)
        return msiniplanets
           
    def compute_semimajor_axes(self, mcmc_result, mstar, mstar_unc):
        """Computes the semi-major axis of each planet and uncertainty.

        Args:
            mcmc_result (dict): The returned value from calling sample.
            mstar (float): The mass of the star in solar units.
            mstar (list): The uncertainty of the mass of the star in solar units, lower, upper.

        Returns:
            (dict): The semi-major axis, lower, and upper uncertainty of each planet in a dictionary.
        """
        
        if mstar_unc is None:
            mstar_unc = (0, 0)
        
        sa_dict = {} # In AU
        
        for planet_index in self.planets_dict:
            perdist = []
            tpdist = []
            eccdist = []
            wdist = []
            kdist = []
            adist = []
            pars = copy.deepcopy(mcmc_result["pmed"])
            for i in range(mcmc_result["n_steps"]):
                for pname in self.planets_dict[planet_index]["basis"].pnames:
                    if pars[pname].vary:
                        ii = pars.index_from_par(pname, rel_vary=True)
                        pars[pname].value = mcmc_result["chains"][i, ii]
                per, tp, ecc, w, k = self.planets_dict[planet_index]["basis"].to_standard(pars)
                perdist.append(per)
                tpdist.append(tp)
                eccdist.append(ecc)
                wdist.append(w)
                kdist.append(k)
                a = planetmath.compute_sa(per, mstar)
                adist.append(a)
            val, unc_low, unc_high = self.sampler.chain_uncertainty(adist)
            da_dMstar = planetmath.compute_sa_deriv_mstar(per, mstar) # in AU / M_SUN
            unc_low = np.sqrt(unc_low**2 + da_dMstar**2 * mstar_unc[0]**2)
            unc_high = np.sqrt(unc_high**2 + da_dMstar**2 * mstar_unc[1]**2)
            sa_dict[planet_index] = (val, unc_low, unc_high)
            
        return sa_dict
            
    def compute_densities(self, mcmc_result, mstar, mstar_unc=None, rplanets_dict=None):
        """Computes the value of msini and uncertainty for each planet in units of Earth Masses.

        Args:
            mcmc_result (dict): The returned value from calling sample.
        Returns:
            (dict): The density, lower, and upper uncertainty of each planet in a dictionary, in units of grams/cm^3.
        """
        if mstar_unc is None:
            mstar_unc = (0, 0)
        if rplanets_dict is None:
            rplanets_dict = {}
            
        mplanets = self.compute_masses(mcmc_result, mstar, mstar_unc)
        rho_dict = {} # In jupiter masses
        for planet_index in self.planets_dict:
            mass = mplanets[planet_index][0] * planetmath.MASS_EARTH_GRAMS
            mass_unc_low = mplanets[planet_index][1] * planetmath.MASS_EARTH_GRAMS
            mass_unc_high = mplanets[planet_index][2] * planetmath.MASS_EARTH_GRAMS
            radius = rplanets_dict[planet_index][0] * planetmath.RADIUS_EARTH_CM
            radius_unc_low = rplanets_dict[planet_index][1] * planetmath.RADIUS_EARTH_CM
            radius_unc_high = rplanets_dict[planet_index][2] * planetmath.RADIUS_EARTH_CM
            rho = 3 * mass / (4 * np.pi * radius**3)
            rho_unc_low = np.sqrt((3 / (4 * np.pi * radius**3))**2 * mass_unc_low**2 + (9 * mass / (4 * np.pi * radius**4))**2 * radius_unc_high**2)
            rho_unc_high = np.sqrt((3 / (4 * np.pi * radius**3))**2 * mass_unc_high**2 + (9 * mass / (4 * np.pi * radius**4))**2 * radius_unc_low**2)
            rho_dict[planet_index] = (rho, rho_unc_low, rho_unc_high)
        return rho_dict
            
    
    ###############
    #### MISC. ####
    ###############
    
    @staticmethod
    def _brute_force_wrapper(rvprob, per, planet_index):
        rvprob.p0[f"per{planet_index}"].value = per
        opt_result = rvprob.run_mapfit(save=False)
        return opt_result

    @staticmethod
    def _generate_all_planet_dicts(planets_dict):
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

    @property
    def instnames(self):
        return list(self.data.keys())
    
    @property
    def planets_dict(self):
        return self.like0.model.planets_dict
    
    @property
    def data_t(self):
        t = np.array([], dtype=float)
        for like in self.post.values():
            for data in self.post.values():
                t = np.concatenate((t, like.model.data.t))
        t = np.sort(t)
        return t