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
import pychell.orbits.maths as planetmath

# Optimize
from optimize.problems import OptProblem
import optimize.samplers
from optimize.noise import CorrelatedNoiseProcess, GaussianProcess

# Plots
import corner
import matplotlib.pyplot as plt
import plotly.subplots
import plotly.graph_objects
PLOTLY_COLORS = pcutils.PLOTLY_COLORS
COLORS_HEX_GADFLY = pcutils.COLORS_HEX_GADFLY
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

class RVProblem(OptProblem):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, p0=None, post=None, star_name=None, tag=None, output_path=None):
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
        super().__init__(name=f"RV Modeling for {star_name}", p0=p0, obj=post)
        
        # The output path
        self.output_path = output_path
        
        # The tag of this run for filenames
        self.tag = "" if tag is None else tag
        
        # Star name
        self.star_name = 'Star' if star_name is None else star_name
        
        # Full tag for filenames
        self.full_tag = f"{self.star_name}_{self.tag}"
        
        # Generate latex labels for the parameters.
        self.gen_latex_labels()
        
            
    def gen_latex_labels(self):
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
    
    def run_mapfit(self, optimizer, save=True):
        """Runs the optimizer.
            
        Returns:
            dict: A dictionary with the optimize results.
        """
        map_result = optimizer.optimize(p0=self.p0, obj=self.post)
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
        planet_model_phased = self.post.like0.model.build_planet(pars, t_hr_one_period, planet_index)
        
        # Sort the phased model
        ss = np.argsort(phases_hr_one_period)
        phases_hr_one_period = phases_hr_one_period[ss]
        planet_model_phased = planet_model_phased[ss]
        
        # Store the data in order to bin the phased RVs.
        phases_data_all = np.array([], dtype=float)
        rvs_data_all = np.array([], dtype=float)
        unc_data_all = np.array([], dtype=float)
        
        # Loop over likes
        for like in self.post.likes.values():
            
            # Compute the final residuals
            residuals = like.compute_residuals(pars)
            
            # Compute the noise model
            if isinstance(like.noise_process, CorrelatedNoiseProcess):
                errors = like.compute_data_errors(pars)
                noise_components = like.compute_noise_components(pars, like.datax)
                for comp in noise_components:
                    residuals[noise_components[comp][2]] -= noise_components[comp][0][noise_components[comp][2]]
            else:
                errors = like.compute_data_errors(pars)
            
            # Loop over instruments and plot each
            for data in like.model.data.values():
                errors_arr = errors[like.model.data.indices[data.instname]]
                data_arr = residuals[like.model.data.indices[data.instname]] + like.model.build_planet(pars, data.t, planet_index)
                phases_data = planetmath.get_phases(data.t, per, tc)
                fig.add_trace(plotly.graph_objects.Scatter(x=phases_data, y=data_arr,
                                                           error_y=dict(array=errors_arr),
                                                           name=f"<b>{data.instname}</b>",
                                                           mode='markers',
                                                           marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.instname], a=0.8))))
                
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
        fig.update_layout(title=f"<b>{self.star_name} {self.post.like0.model.planets_dict[planet_index]['label']} <br> P = {round(per, 6)}, K = {round(k, 2)} e = {round(ecc, 3)}</b>")
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
        if self.post.like0.model.n_planets > 0 or self.post.like0.model.trend_poly_order > 0:
            
            # Generate a high resolution data grid.
            t_data_all = np.array([], dtype=float)
            for like in self.post.likes.values():
                t_data_all = np.concatenate((t_data_all, like.model.data.t))
            t_start, t_end = np.nanmin(t_data_all), np.nanmax(t_data_all)
            dt = t_end - t_start
            t_hr = np.linspace(t_start - dt / 100, t_end + dt / 100, num=n_model_pts)
        
            # Generate the high res Keplerian + Trend model.
            model_arr_hr = self.post.like0.model.build(pars, t_hr)
            
            # Label
            if self.post.like0.model.n_planets > 0 and self.post.like0.model.trend_poly_order > 0:
                name = "<b>Keplerian + Trend Model</b>"
            elif self.post.like0.model.n_planets > 0 and self.post.like0.model.trend_poly_order == 0:
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
        for like in self.post.likes.values():

            # Correlated Noise
            if isinstance(like.noise_process, CorrelatedNoiseProcess):
                
                # Make hr arrays for high res GP
                noise_components_temp = like.compute_noise_components(pars, like.model.data.t[0:2])
                noise_labels = list(noise_components_temp.keys())
                t_gp_hr = np.array([], dtype=float)
                gps_hr = {label : np.array([], dtype=float) for label in noise_labels}
                gps_error_hr = {label : np.array([], dtype=float) for label in noise_labels}
                residuals = like.compute_residuals(pars)
                for i in range(like.model.data.t.size):
                    
                    # Create a time array centered on this point
                    t_hr_window = np.linspace(like.model.data.t[i] - kernel_window,
                                              like.model.data.t[i] + kernel_window,
                                              num=kernel_sampling)
                    t_gp_hr = np.concatenate((t_gp_hr, t_hr_window))
                    
                    # Sample GP for this window
                    noise_components = like.compute_noise_components(pars, t_hr_window)
                    for comp in noise_components:
                        gps_hr[comp] = np.concatenate((gps_hr[comp], noise_components[comp][0]))
                        gps_error_hr[comp] = np.concatenate((gps_error_hr[comp], noise_components[comp][1]))
                    
                # Sort each
                ss = np.argsort(t_gp_hr)
                t_gp_hr = t_gp_hr[ss]
                for label in noise_labels:
                    gps_hr[label] = gps_hr[label][ss]
                    gps_error_hr[label] = gps_error_hr[label][ss]

                # Data errors
                data_errors = like.compute_data_errors(pars)
                    
                # Plot the GPs
                for label in noise_labels:

                    # Colors
                    color_line = pcutils.hex_to_rgba(self.color_map[label], a=0.6)
                    color_fill = pcutils.hex_to_rgba(self.color_map[label], a=0.3)

                    fig.add_trace(plotly.graph_objects.Scatter(x=t_gp_hr - time_offset, y=gps_hr[label],
                                                              line=dict(width=0.8, color=color_line),
                                                              name=f"<b>{label}</b>", showlegend=False),
                                  row=1, col=1)
                
                    # Plot the gp unc
                    gp_hr_lower, gps_hr_upper = gps_hr[label] - gps_error_hr[label], gps_hr[label] + gps_error_hr[label]
                    fig.add_trace(plotly.graph_objects.Scatter(x=np.concatenate([t_gp_hr, t_gp_hr[::-1]]) - time_offset,
                                                               y=np.concatenate([gps_hr_upper, gp_hr_lower[::-1]]),
                                                               fill='toself',
                                                               line=dict(width=1, color=color_line),
                                                               fillcolor=color_fill,
                                                               name=f"<b>{label}</b>", showlegend=True),
                                  row=1, col=1)
                
                noise_components = like.compute_noise_components(pars, like.datax)
                for comp in noise_components:
                    residuals[noise_components[comp][2]] -= noise_components[comp][0][noise_components[comp][2]]

                # Plot each instrument
                for data in like.model.data.values():
                    
                    # Raw data - zero point
                    data_arr = data.rv - pars[f"gamma_{data.instname}"].value
                    
                    # Errors
                    errors_arr = data_errors[like.model.data.indices[data.instname]]
                    
                    # Final residuals
                    residuals_arr = residuals[like.model.data.indices[data.instname]]
                    
                    # Plot rvs
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=data_arr,
                                                               error_y=dict(array=errors_arr),
                                                               name=data.instname,
                                                               mode='markers',
                                                               marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.instname], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14)),
                                  row=1, col=1)
                    
                    # Plot residuals
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=residuals_arr,
                                                               error_y=dict(array=errors_arr),
                                                               mode='markers',
                                                               marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.instname], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14), showlegend=False),
                                  row=2, col=1)
            

            else:
                
                # Generate the residuals
                residuals = like.compute_residuals(pars)
                
                # Compute data errors
                errors = like.compute_data_errors(pars)
                
                for data in like.model.data.values():
                    
                    # Raw data - zero point
                    data_arr = data.rv - pars[f"gamma_{data.instname}"].value
                    
                    # Errors
                    errors_arr = errors[like.model.data.indices[data.instname]]
                    
                    # Final residuals
                    residuals_arr = residuals[like.model.data.indices[data.instname]]
                    
                    # Plot rvs
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=data_arr,
                                                               error_y=dict(array=errors_arr),
                                                               name=data.instname,
                                                               mode='markers',
                                                               marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.instname], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14)),
                                  row=1, col=1)
                    
                    # Plot residuals
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=residuals_arr,
                                                               error_y=dict(array=errors_arr),
                                                               mode='markers',
                                                               marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.instname], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14), showlegend=False),
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
        


    #################################
    #### BRUTE FORCE PERIODOGRAM ####
    #################################
            
    def brute_force_periodogram(self, optimizer, periods, planet_index=1, n_cores=1):
        # Run in parallel
        results = Parallel(n_jobs=n_cores, verbose=0, batch_size=1)(delayed(self._brute_force_wrapper)(self, optimizer, periods[i], planet_index) for i in tqdm.tqdm(range(len(periods))))
        return results
    
    ##########################
    #### MODEL COMPARISON ####
    ##########################
    
    def model_comparison(self, optimizer, save=True):
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
                    _optprob.post.like0.model.disable_planet_pars(p0, planets_dict_cp, planet_index)
            
            # Set planets dict for each model
            for like in _optprob.post.likes.values():
                like.model.planets_dict = planets_dict

            # Run the max like
            _optprob.p0 = p0
            opt_result = _optprob.run_mapfit(optimizer, save=False)
            
            # Alias best fit params
            pbest = opt_result['pbest']
            
            # Recompute the max like to NOT include any priors to keep things consistent.
            lnL = _optprob.post.compute_logL(pbest)
            
            # Run the BIC
            bic = _optprob.post.compute_bic(pbest)
            
            # Run the AICc
            aicc = _optprob.post.compute_aicc(pbest)
            
            # Red chi 2
            redchi2 = _optprob.post.compute_redchi2(pbest)
            
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
            val, unc_low, unc_high = optimize.samplers.emceeLikeSampler.chain_uncertainty(mdist)
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
            val, unc_low, unc_high = optimize.samplers.emceeLikeSampler.chain_uncertainty(adist)
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
    def _brute_force_wrapper(rvprob, optimizer, per, planet_index):
        rvprob.p0[f"per{planet_index}"].value = per
        opt_result = rvprob.run_mapfit(optimizer, save=False)
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
    def post(self):
        return self.obj

    @property
    def instnames(self):
        return list(self.data.keys())
    
    @property
    def planets_dict(self):
        return self.post.like0.model.planets_dict
    