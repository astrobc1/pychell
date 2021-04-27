
# Standard library 
import copy
import multiprocessing
import itertools
import os
import pickle
N_CPUS = multiprocessing.cpu_count()

# Utils
from numba import jit, njit
from joblib import Parallel, delayed
import tqdm

# Arrays, linear algebra, and science
import numpy as np
import scipy.constants
from PyAstronomy.pyTiming import pyPeriod

# Pychell deps
import pychell
import pychell.maths as pcmath
import pychell.orbits.rvkernels as pcrvkernels
import pychell.orbits.rvobjectives as pcrvobj
import pychell.utils as pcutils
PLOTLY_COLORS = pcutils.PLOTLY_COLORS
COLORS_HEX_GADFLY = pcutils.COLORS_HEX_GADFLY
import pychell.orbits.rvmodels as pcrvmodels
import pychell.orbits.planetmath as planetmath

# Optimize
import optimize.frameworks as optframeworks
import optimize.kernels as optkernels

# Plots
import corner
import matplotlib.pyplot as plt
import plotly.subplots
import plotly.graph_objects
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

class RVProblem(optframeworks.OptProblem):
    """The primary, top-level container for Exoplanet optimization problems. As of now, this only deals with RV data. Photometric modeling will be included in future updates, but will leverage existing libraries (Batman, etc.).
    """
    
    #############################################
    #### Constructor Methods (incl. helpers) ####
    #############################################
    
    def __init__(self, output_path=None, data=None, p0=None, optimizer=None, sampler=None, obj=None, star_name=None, mstar=None, mstar_unc=None, rplanets=None, tag=None):
        """Constructs the primary exoplanet problem object.

        Args:
            output_path (The output path for plots and pickled objects, optional): []. Defaults to this current working direcrory.
            data (CompositeRVData, optional): The composite data object. Defaults to None.
            p0 (Parameters, optional): The initial parameters. Defaults to None.
            optimizer (Optimizer, optional): The max like optimizer. Defaults to None.
            sampler (Sampler, optional): The MCMC sampler object. Defaults to None.
            obj (CompositeRVLikelihood, optional): The composite likelihood object. Defaults to None.
            star_name (str, optional): The name of the star, may contain spaces. Defaults to None.
            mstar (float, optional): The mass of the star in solar units. Defaults to None.
            mstar_unc (list, optional): The uncertainty in mstar, same units. Defaults to None.
            rplanets (dict, optional): The radius of the planet in Earth units. Defaults to None. Format is {planet_index: (val, lower, upper)}
        """
        
        # Pass relevant items to base class constructor
        super().__init__(p0=p0, optimizer=optimizer, sampler=sampler, obj=obj, output_path=output_path)
        
        # The tag of this run for filenames
        self.tag = "" if tag is None else tag
        
        # Star name
        self.star_name = 'Star' if star_name is None else star_name
        
        # Full tag for filenames
        self.full_tag = f"{self.star_name}_{self.tag}"
        
        # Mass of star for deriving mass uncertainties.
        self.mstar = mstar
        self.mstar_unc = mstar_unc
        
        # Radius of planets for deriving densities.
        self.rplanets = rplanets
        
        # Generate latex labels for the parameters.
        self._gen_latex_labels()
        
        # Make a color map for plotting.
        self._make_color_map()
        
        # Ensure parameters are forwarded to relevant components.
        self.set_pars()
        
    def _make_color_map(self):
        self.color_map = {}
        color_index = 0
        for like in self.likes.values():
            for data in like.data.values():
                self.color_map[data.label] = COLORS_HEX_GADFLY[color_index % len(COLORS_HEX_GADFLY)]
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
    #### Standard Optimization Methods ####
    #######################################
    
    def sample(self, *args, save=True, **kwargs):
        """Runs the mcmc.

        Returns:
            *args: Any args.
            **kwargs: Any keyword args.
        
        Returns:
            dict: A dictionary with the mcmc results.
        """
        mcmc_result = super().sample(*args, **kwargs)
        if save:
            fname = self.output_path + self.star_name.replace(' ', '_') + '_mcmc_results_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(mcmc_result, f)
        return mcmc_result
    
    def optimize(self, *args, save=True, **kwargs):
        """Runs the optimizer.

        Args:
            *args: Any args.
            **kwargs: Any keyword args.
            
        Returns:
            dict: A dictionary with the optimize results.
        """
        map_result = super().optimize(*args, **kwargs)
        if save:
            fname = self.output_path + self.star_name.replace(' ', '_') + '_map_results_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(map_result, f)
        return map_result
    
    def mapfit(self, *args, **kwargs):
        """Alias for optimize.

        Args:
            *args: Any args.
            **kwargs: Any keyword args.
            
        Returns:
            dict: A dictionary with the optimize results.
        """
        return self.optimize(*args, **kwargs)
        
    def mcmc(self, *args, **kwargs):
        """Alias for sample.

        Args:
            *args: Any args.
            **kwargs: Any keyword args.
            
        Returns:
            dict: A dictionary with the mcmc results.
        """
        return self.sample(*args, **kwargs)
        
    ###################################
    #### Standard Plotting Methods ####
    ###################################
    
    def plot_phased_rvs(self, planet_index, pars=None, plot_width=1000, plot_height=600):
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
            
        # Creat a figure
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
        color_index = 0
        for like in self.obj.values():
            
            # Create a data rv vector where everything is removed except this planet
            residuals_with_noise = like.residuals_with_noise(pars)
            residuals_no_noise = like.residuals_no_noise(pars)
            
            # Compute error bars
            errors = like.kernel.compute_data_errors(pars, include_white_error=True, include_kernel_error=True, residuals_with_noise=residuals_with_noise, kernel_error=None)
            
            # Loop over instruments and plot each
            for data in like.data.values():
                _errors = errors[like.model.data_inds[data.label]]
                _data = residuals_no_noise[like.model.data_inds[data.label]] + like.model.build_planet(pars, data.t, planet_index)
                phases_data = planetmath.get_phases(data.t, per, tc)
                _yerr = dict(array=_errors)
                fig.add_trace(plotly.graph_objects.Scatter(x=phases_data, y=_data, name="<b>" + data.label + "</b>", error_y=_yerr, mode='markers', marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.8))))
                color_index += 1
                
                # Store for binning
                phases_data_all = np.concatenate((phases_data_all, phases_data))
                rvs_data_all = np.concatenate((rvs_data_all, _data))
                unc_data_all = np.concatenate((unc_data_all, _errors))

        # Plot the model on top
        fig.add_trace(plotly.graph_objects.Scatter(x=phases_hr_one_period, y=planet_model_phased, line=dict(color='black', width=2), name="<b>Keplerian Model</b>"))
        
        # Lastly, generate and plot the binned data.
        ss = np.argsort(phases_data_all)
        phases_data_all, rvs_data_all, unc_data_all = phases_data_all[ss], rvs_data_all[ss], unc_data_all[ss]
        phases_binned, rvs_binned, unc_binned = planetmath.bin_phased_rvs(phases_data_all, rvs_data_all, unc_data_all, window=0.1)
        fig.add_trace(plotly.graph_objects.Scatter(x=phases_binned, y=rvs_binned, error_y=dict(array=unc_binned), mode='markers', marker=dict(color='Maroon', size=12, line=dict(width=2, color='DarkSlateGrey')), showlegend=False))
        
        # Labels
        fig.update_xaxes(title_text='<b>Phase</b>')
        fig.update_yaxes(title_text='<b>RV [m/s]</b>')
        fig.update_layout(title='<b>' + self.star_name + ' ' + self.like0.model.planets_dict[planet_index]["label"] + '<br>' + 'P = ' + str(round(per, 6)) + ', e = ' + str(round(ecc, 5)) + '</b>')
        fig.update_layout(template="plotly_white")
        fig.update_layout(font=dict(size=16))
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_layout(width=plot_width, height=plot_height)
        fig.write_html(self.output_path + self.star_name.replace(' ', '_') + self.planets_dict[planet_index]["label"] + '_rvs_phased_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.html')
        
        # Return fig
        return fig
    
    def plot_phased_rvs_all(self, pars=None, plot_width=1000, plot_height=600):
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
            plot = self.plot_phased_rvs(planet_index, pars=pars, plot_width=plot_width, plot_height=plot_height)
            plots.append(plot)
        return plots
        
    def plot_full_rvs(self, pars=None, ffp=None, n_model_pts=5000, time_offset=2450000, kernel_sampling=200, plot_width=1800, plot_height=1200):
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
        
        # Create the full planet model, high res.
        t_data_all = self.data_t
        t_start, t_end = np.nanmin(t_data_all), np.nanmax(t_data_all)
        dt = t_end - t_start
        t_hr = np.linspace(t_start - dt / 100, t_end + dt / 100, num=n_model_pts)
        
        # Create hr planet model
        model_arr_hr = self.like0.model._builder(pars, t_hr)

        # Plot the planet model
        fig.add_trace(plotly.graph_objects.Scatter(x=t_hr - time_offset, y=model_arr_hr, line=dict(color='black', width=2), name="<b>Keplerian Model</b>"), row=1, col=1)
        
        # ffp pred
        if ffp is not None:
            fig.add_trace(plotly.graph_objects.Scatter(x=ffp[:, 0] - time_offset, y=200 * ffp[:, 1] / np.std(ffp[:, 1]), line=dict(color='red', width=2, dash='dot'), name="<b>FF' Spot Prediction</b>"), row=1, col=1)
        
        # Loop over likes and:
        # 1. Create high res GP
        # 2. Plot high res GP and data
        color_index = 0
        for like in self.obj.values():
            
            # Data errors
            residuals_with_noise = like.residuals_with_noise(pars)
            errors = like.kernel.compute_data_errors(pars, include_white_error=True, include_kernel_error=True, residuals_with_noise=residuals_with_noise)
            
            # RV Color 1
            if type(like) is pcrvobj.RVChromaticLikelihood:
                
                # Generate the residuals
                residuals_with_noise = like.residuals_with_noise(pars)
                residuals_no_noise = like.residuals_no_noise(pars)
                
                s = 4 * pars["gp_per"].value
                
                # Plot a GP for each instrument
                for wavelength in like.kernel.unique_wavelengths:
                    
                    print("Plotting Wavelength = " + str(int(wavelength)) + " nm")
                    
                    # Smartly sample gp
                    inds = like.kernel.get_wave_inds(wavelength)
                    t_hr_gp = np.array([], dtype=float)
                    gpmu_hr = np.array([], dtype=float)
                    gpstddev_hr = np.array([], dtype=float)
                    tvec = like.data.get_vec('t')
                    for i in range(tvec.size):
                        _t_hr_gp = np.linspace(tvec[i] - s, tvec[i] + s, num=kernel_sampling)
                        _gpmu, _gpstddev = like.kernel.realize(pars, xpred=_t_hr_gp, residuals_with_noise=residuals_with_noise, return_kernel_error=True, wavelength=wavelength)
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
                    instnames = like.kernel.get_instnames_for_wave(wavelength)
                    label = '<b>GP ' + '&#x3BB;' + ' = ' + str(int(wavelength)) + ' nm ' + '['
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
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_data, name="<b>" + data.label + "</b>", error_y=_yerr, mode='markers', marker=dict(color=PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], size=12)), row=1, col=1)
                    
                    # Residuals for this instrument after the noise kernel has been removed
                    _residuals = residuals_no_noise[like.model.data_inds[data.label]]
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_residuals, name="<b>" + data.label + "</b>", error_y=_yerr, mode='markers', marker=dict(color=PLOTLY_COLORS[color_index%len(PLOTLY_COLORS)], size=12), showlegend=False), row=2, col=1)
                    
                    # Increase color index
                    color_index += 1
                    
            # RV Color 2
            elif type(like) is pcrvobj.RVChromaticLikelihood2:
                
                # Generate the residuals
                residuals_with_noise = like.residuals_with_noise(pars)
                residuals_no_noise = like.residuals_no_noise(pars)
                
                s = 4 * pars["gp_per"].value
                
                # Plot a GP for each instrument
                for data in like.kernel.data.values():
                    
                    print("Plotting Instrument: " + data.label)
                    
                    # Smartly sample gp
                    t_hr_gp = np.array([], dtype=float)
                    gpmu_hr = np.array([], dtype=float)
                    gpstddev_hr = np.array([], dtype=float)
                    tvec = like.data.get_vec('t')
                    for i in range(tvec.size):
                        _t_hr_gp = np.linspace(tvec[i] - s, tvec[i] + s, num=kernel_sampling)
                        _gpmu, _gpstddev = like.kernel.realize(pars, xpred=_t_hr_gp, residuals_with_noise=residuals_with_noise, return_kernel_error=True, instrument=data.label)
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
                    label = '<b>GP ' + data.label + '</b>'
                    fig.add_trace(plotly.graph_objects.Scatter(x=tt, y=gpmu_hr, line=dict(width=1.2, color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.8)), name=label, showlegend=False), row=1, col=1)
                    fig.add_trace(plotly.graph_objects.Scatter(x=np.concatenate([tt, tt[::-1]]),
                                             y=np.concatenate([gp_upper, gp_lower[::-1]]),
                                             fill='toself',
                                             name=label,
                                             line=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.5), width=1),
                                             fillcolor=pcutils.hex_to_rgba(self.color_map[data.label], a=0.3)))
                    
                # Plot the data on top of the GPs, and the residuals
                data_arr = np.copy(like.data_rv)
                data_arr = like.model.apply_offsets(data_arr, pars)
                for data in like.data.values():
                    
                    # Data errors for this instrument
                    _errors = errors[like.model.data_inds[data.label]]
                    _yerr = dict(array=_errors)
                    
                    # Data on top of the GP, only offset by gammas
                    _data = data_arr[like.model.data_inds[data.label]]
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_data, name="<b>" + data.label + "</b>", error_y=_yerr, mode='markers', marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.95), line=dict(width=2, color='DarkSlateGrey'), size=14)), row=1, col=1)
                    
                    # Residuals for this instrument after the noise kernel has been removed
                    _residuals = residuals_no_noise[like.model.data_inds[data.label]]
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_residuals, name="<b>" + data.label + "</b>", error_y=_yerr, mode='markers', marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.95), line=dict(width=2, color='DarkSlateGrey'), size=14), showlegend=False), row=2, col=1)
                    
            # Standard GP
            elif isinstance(like.kernel, optkernels.GaussianProcess):
                
                # Make hr array for GP
                if 'gp_per' in pars:
                    s = 4 * pars['gp_per'].value
                elif 'gp_decay' in pars:
                    s = pars['gp_decay'].value / 10
                else:
                    s = 10
                t_hr_gp = np.array([], dtype=float)
                gpmu_hr = np.array([], dtype=float)
                gpstddev_hr = np.array([], dtype=float)
                residuals_with_noise = like.residuals_with_noise(pars)
                if like.label == "rvs_HIRES":
                    continue
                for i in range(like.data_t.size):
                    _t_hr_gp = np.linspace(like.data_t[i] - s, like.data_t[i] + s, num=kernel_sampling)
                    _gpmu, _gpstddev = like.kernel.realize(pars, xpred=_t_hr_gp, residuals_with_noise=residuals_with_noise, return_kernel_error=True)
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
                fig.add_trace(plotly.graph_objects.Scatter(x=tt, y=gpmu_hr, line=dict(width=0.8, color=pcutils.hex_to_rgba(self.color_map[list(like.data.keys())[0]], a=0.8)), name=label, showlegend=False), row=1, col=1)
                fig.add_trace(plotly.graph_objects.Scatter(x=np.concatenate([tt, tt[::-1]]),
                                            y=np.concatenate([gp_upper, gp_lower[::-1]]),
                                            fill='toself',
                                            line=dict(color=pcutils.hex_to_rgba(self.color_map[list(like.data.keys())[0]], a=0.5), width=1),
                                            fillcolor=pcutils.hex_to_rgba(self.color_map[list(like.data.keys())[0]], a=0.3),
                                            name=label))
                
                # Generate the residuals without noise
                residuals_no_noise = like.residuals_no_noise(pars)
                
                # For each instrument, plot
                for data in like.data.values():
                    data_arr_offset = like.model.apply_offsets(data.rv, pars, instname=data.label)
                    _errors = errors[like.model.data_inds[data.label]]
                    _residuals = residuals_no_noise[like.model.data_inds[data.label]]
                    _yerr = dict(array=_errors)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=data_arr_offset, name=data.label, error_y=_yerr, mode='markers', marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14)), row=1, col=1)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_residuals, error_y=_yerr, mode='markers', marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.9), line=dict(width=2, color='DarkSlateGrey'), size=14), showlegend=False), row=2, col=1)
                    color_index += 1
                
            # White noise
            else:
                residuals_no_noise = like.residuals_no_noise(pars)
                
                # For each instrument, plot
                for data in like.data.values():
                    data_arr_offset = data.rv - pars['gamma_' + data.label].value
                    _errors = errors[like.model.data_inds[data.label]]
                    _residuals = residuals_no_noise[like.model.data_inds[data.label]]
                    _yerr = dict(array=_errors)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=data_arr_offset, name=data.label, error_y=_yerr, mode='markers', marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.95), line=dict(width=2, color='DarkSlateGrey'), size=12)), row=1, col=1)
                    fig.add_trace(plotly.graph_objects.Scatter(x=data.t - time_offset, y=_residuals, error_y=_yerr, mode='markers', marker=dict(color=pcutils.hex_to_rgba(self.color_map[data.label], a=0.95), size=12, line=dict(width=2, color='DarkSlateGrey')), showlegend=False), row=2, col=1)
                    color_index += 1


        # Labels
        fig.update_xaxes(title_text='<b>BJD - ' + str(time_offset) + '</b>', row=2, col=1)
        fig.update_yaxes(title_text='<b>Residual RV [m/s]</b>', row=2, col=1)
        fig.update_yaxes(title_text='<b>RV [m/s]</b>', row=1, col=1)
        fig.update_xaxes(tickprefix="<b>",ticksuffix ="</b><br>")
        fig.update_yaxes(tickprefix="<b>",ticksuffix ="</b><br>")

        # Limits
        fig.update_xaxes(range=[t_start - dt / 10 - time_offset, t_end + dt / 10 - time_offset], row=1, col=1)
        fig.update_xaxes(range=[t_start - dt / 10 - time_offset, t_end + dt / 10 - time_offset], row=2, col=1)
        
        # Appearance
        fig.update_layout(template="plotly_white")
        fig.update_layout(font=dict(size=20))
        fig.update_layout(width=plot_width, height=plot_height)
        fig.write_html(self.output_path + self.star_name.replace(' ', '_') + '_rvs_full_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.html')
        
        # Return the figure for streamlit
        return fig
        
    def corner_plot(self, mcmc_result):
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
        corner_plot.savefig(self.output_path + self.star_name.replace(' ', '_') + '_corner_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.png')
        return corner_plot
        
    ######################
    #### Periodograms ####
    ######################
        
    def gls_periodogram(self, pmin=None, pmax=None, apply_gp=True, remove_planets=None):
        """Creates a GLS periodogram through external routines.

        Args:
            pmin (float, optional): The min period to test in days. Defaults to 1.1.
            pmax (float, optional): The max period to test in days. Defaults to the time baseline.
            apply_gp (bool, optional): Whether or not to first remove thr best fit GP model with no planets to the data. Defaults to True.
            remove_planets (list, optional): A list of indices (ints) to remove from the model. Defaults to None, empty list.

        Returns:
            GLSPeriodogramResults: The periodogram object results after calling Gls().
        """
        
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
        
        # Cases
        if apply_gp and len(remove_planets) > 0:
            
            # Create new pars and planets dict, keep copies of old
            p0cp = copy.deepcopy(self.p0)
            p0mod = copy.deepcopy(self.p0)
            planets_dict_cp = copy.deepcopy(self.obj.like0.model.planets_dict)
            planets_dict_mod = copy.deepcopy(self.obj.like0.model.planets_dict)
            
            # If the planet is not in remove_planets, we want to disable it from the initial fitting so it stays in the data
            for planet_index in remove_planets:
                if planet_index not in remove_planets:
                    self.like0.model.disable_planet_pars(p0mod, planets_dict_cp, planet_index)
                    del planets_dict_cp[planet_index]
                    
            # Set the planets dict
            for like in self.obj.values():
                like.model.planets_dict = planets_dict_mod
                
            # Set the modified parameters
            self.set_pars(p0mod)
            
            # Perform max like fit
            opt_result = self.mapfit(save=False)
            pbest = opt_result["pbest"]
                
            # Construct the GP for each like and remove from the data
            data_t = np.array([], dtype=float)
            data_rvs = np.array([], dtype=float)
            data_unc = np.array([], dtype=float)
            for like in self.obj.values():
                residuals_with_noise = like.residuals_with_noise(pbest)
                residuals_no_noise = like.residuals_no_noise(pbest)
                errors = like.kernel.compute_data_errors(pbest, include_white_error=True, include_kernel_error=True, residuals_with_noise=residuals_with_noise, kernel_error=kernel_error)
                for data in like.data.values():
                    inds = like.model.data_inds[data.label]
                    data_t = np.concatenate((data_t, data.t))
                    data_rvs = np.concatenate((data_rvs, residuals_no_noise[inds]))
                    data_unc = np.concatenate((data_unc, errors[inds]))
                    
            # Reset parameters dict
            for like in self.obj.values():
                like.model.planets_dict = planets_dict_cp
                
            # Reset parameters
            self.set_pars(p0cp)
        
        elif apply_gp and len(remove_planets) == 0:
            
            # Create new pars and planets dict, keep copies of old
            p0cp = copy.deepcopy(self.p0)
            p0mod = copy.deepcopy(self.p0)
            planets_dict_cp = copy.deepcopy(self.obj.like0.model.planets_dict)
            planets_dict_mod = {}
            
            # Disable all planet parameters
            for planet_index in planets_dict_cp:
               self.like0.model.disable_planet_pars(p0mod, planets_dict_cp, planet_index)
                    
            # Set the planets dict
            for like in self.obj.values():
               like.model.planets_dict = planets_dict_mod
                
            # Set the modified parameters
            self.set_pars(p0mod)

            # Perform max like fit
            opt_result = self.mapfit(save=False)
            pbest = opt_result["pbest"]
                
            # Construct the GP for each like and remove from the data
            # Don't remove any planets.
            data_t = np.array([], dtype=float)
            data_rvs = np.array([], dtype=float)
            data_unc = np.array([], dtype=float)
            for like in self.obj.values():
                residuals_with_noise = like.residuals_with_noise(pbest)
                residuals_no_noise = like.residuals_no_noise(pbest)
                errors = like.kernel.compute_data_errors(pbest, include_white_error=True, include_kernel_error=True, residuals_with_noise=residuals_with_noise)
                for data in like.data.values():
                    inds = like.model.data_inds[data.label]
                    data_t = np.concatenate((data_t, data.t))
                    data_rvs = np.concatenate((data_rvs, residuals_no_noise[inds]))
                    data_unc = np.concatenate((data_unc, errors[inds]))
            
            # Reset parameters and planets_dict
            self.set_pars(p0cp)
            
            # Reset planets dict
            for like in self.obj.values():
                like.model.planets_dict = planets_dict_cp
                
        elif not apply_gp and len(remove_planets) > 0:
            
            # Create new pars and planets dict, keep copies of old
            p0cp = copy.deepcopy(self.p0)
            p0mod = copy.deepcopy(self.p0)
            planets_dict_cp = copy.deepcopy(self.obj.like0.model.planets_dict)
            planets_dict_mod = copy.deepcopy(self.obj.like0.model.planets_dict)
            
            # If the planet is not in remove_planets, we want to disable it from the initial fitting so it stays in the data
            for planet_index in remove_planets:
                if planet_index not in remove_planets:
                    self.like0.model.disable_planet_pars(p0mod, planets_dict_cp, planet_index)
                    del planets_dict_cp[planet_index]
                    
            # Set the planets dict
            for like in self.obj.values():
                like.model.planets_dict = planets_dict_mod
                
            # Set the modified parameters
            self.set_pars(p0mod)
            
            # Perform max like fit
            opt_result = self.mapfit(save=False)
            pbest = opt_result["pbest"]
                
            for like in self.obj.values():
                errors = like.kernel.compute_data_errors(pbest, include_white_error=True, include_kernel_error=True, kernel_error=None, residuals_with_noise=None)
                data_arr = np.copy(like.data_rv)
                data_arr = like.model.apply_offsets(data_arr, pbest)
                for data in like.data.values():
                    inds = self.data.get_inds(data.label)
                    data_rvs[inds] = data_arr[like.model.data_inds[data.label]]
                    data_errors[inds] = errors[like.model.data_inds[data.label]]
                    
            # Reset parameters dict
            for like in self.obj.values():
                like.model.planets_dict = planets_dict_cp
            
            # Construct the best fit planets and remove from the data
            for planet_index in planets_dict_mod:
                data_rvs -= self.obj.like0.model.build_planet(pbest, data_times, planet_index)
            
            # Reset parameters
            self.set_pars(p0cp)
            
        else:
            data_t = np.array([], dtype=float)
            data_rvs = np.array([], dtype=float)
            data_unc = np.array([], dtype=float)
            for like in self.obj.values():
                errors = like.kernel.compute_data_errors(self.p0, include_jit=True, include_gp=True, gp_unc=None, residuals_no_noise=None)
                data_arr = np.copy(like.data_rv)
                data_arr = like.model.apply_offsets(data_arr, self.p0)
                for data in like.data.values():
                    data_t = np.concatenate((data_t, data.t))
                    inds = self.data.get_inds(data.label)
                    data_rvs = np.concatenate((data_rvs, data_arr[like.model.data_inds[data.label]]))
                    data_unc = np.concatenate((data_unc, errors[like.model.data_inds[data.label]]))
        
        # Call GLS
        ss = np.argsort(data_t)
        data_t, data_rvs, data_unc = data_t[ss], data_rvs[ss], data_unc[ss]
        pgram = pyPeriod.Gls((data_t, data_rvs, data_unc), Pbeg=pmin, Pend=pmax)
        
        return pgram

    def rv_period_search(self, pars=None, pmin=None, pmax=None, n_periods=None, n_cores=1, planet_index=1):
        """A brute force period search. A max like fit is run for many locked periods for a specified planet. The remaining model params are subject to what is passed, allowing for flexible brute force searches.

        Args:
            pars (Parameters, optional): The parameters to use. Defaults to None.
            pmin (float, optional): The min period to test in days. Defaults to 1.1.
            pmax (float, optional): The max period to test in days. Defaults to the time baseline.
            n_periods (int, optional): The number of periods to test. Defaults to None.
            n_cores (int, optional): The number of CPU cores to use. Defaults to 1.
            planet_index (int, optional): The planet index to test. Defaults to 1.

        Raises:
            ValueError: [description]

        Returns:
            np.ndarray: The periods tested.
            list: The results for each period tested.
        """
        
        # Resolve parameters
        if pars is None:
            pars = self.p0
        
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
        with open(self.output_path + self.star_name.replace(' ', '_') + '_persearch_results_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.pkl', 'wb') as f:
            pickle.dump({"periods": periods, "persearch_results": persearch_results}, f)
            
        return periods, persearch_results
    
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
            
        wave_vec = self.data.get_wave_vec()
        wave_pairs = self.generate_all_wave_pairs(wave_vec)
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
        wave_vec = self.data.get_wave_vec()
        wave_pairs = self.generate_all_wave_pairs(wave_vec)
        
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
        wave_vec = self.obj.like0.kernel.make_wave_vec()
        times_vec = self.data.get_vec('t')
        rv_vec = self.data.get_vec('rv')
        rv_vec = self.like0.model.apply_offsets(rv_vec, pars)
        unc_vec = self.like0.kernel.compute_data_errors(pars, include_white_error=True, include_kernel_error=False, residuals_with_noise=None, kernel_error=None)
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
        residuals_with_noise = self.like0.residuals_with_noise(pars)
        
        # Compute the coarsely sampled GP for each wavelength.
        gp_mean1_data, gp_stddev1_data = self.like0.kernel.realize(pars, residuals_with_noise, xpred=jds_avg, return_kernel_error=True, wavelength=wave1)
        gp_mean2_data, gp_stddev2_data = self.like0.kernel.realize(pars, residuals_with_noise, xpred=jds_avg, return_kernel_error=True, wavelength=wave2)
        
        # Compute the coarsely sampled GP-color and unc
        gp_color_data = gp_mean1_data - gp_mean2_data
        gp_color_unc_data = np.sqrt(gp_stddev1_data**2 + gp_stddev2_data**2)
        
        # Compute the densely sampled GP-color
        t_gp_hr, gpmean1_hr, gpstddev1_hr = self.gp_smart_sample(pars, self.like0, s=pars["gp_per"].value*2, t=jds_avg, residuals_with_noise=residuals_with_noise, kernel_sampling=200, return_kernel_error=True, wavelength=wave1)
        _, gpmean2_hr, gpstddev2_hr = self.gp_smart_sample(pars, self.like0, s=pars["gp_per"].value*2, t=jds_avg, residuals_with_noise=residuals_with_noise, kernel_sampling=200, return_kernel_error=True, wavelength=wave2)
        
        gpmean_color_hr = gpmean1_hr - gpmean2_hr
        gpstddev_color_hr = np.sqrt(gpstddev1_hr**2 + gpstddev2_hr**2)
                
        # Return a mega dictionary
        out = dict(jds_avg=jds_avg, rv_data1=rv_data1, rv_data2=rv_data2, rv_unc_data1=rv_unc_data1, rv_unc_data2=rv_unc_data2, rvcolor_data=rvcolor_data, unccolor_data=unccolor_data, gp_color_data=gp_color_data, gp_color_unc_data=gp_color_unc_data, t_gp_hr=t_gp_hr, gpmean1_hr=gpmean1_hr, gpstddev1_hr=gpstddev1_hr, gpmean2_hr=gpmean2_hr, gpstddev2_hr=gpstddev2_hr, gpmean_color_hr=gpmean_color_hr, gpstddev_color_hr=gpstddev_color_hr, instnames=instnames, wave1=wave1, wave2=wave2)
        
        return out
    
    def get_simult_obs(self, pars, sep=0.3):
        
        # Parameters
        if pars is None:
            pars = self.p0
            
        # Filter data to only consider each wavelength
        wave_vec = self.obj.like0.kernel.make_wave_vec()
        tel_vec = self.data.make_tel_vec()
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
    #### Additional Tools ####
    ##########################
    
    def model_comparison(self):
        """Runs a model comparison for all combinations of planets.

        Returns:
            list: Each entry is a dict containing the model comp results for each case, and is sorted according to the small sample AIC.
        """
            
        # Store results in a list
        model_comp_results = []
        
        # Alias like0
        like0 = self.obj.like0
        
        # Original planets dict
        planets_dict_cp = copy.deepcopy(like0.model.planets_dict)
        
        # Get all combos
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
                    self.like0.model.disable_planet_pars(p0, planets_dict_cp, planet_index)
            
            # Set planets dict for each model
            for like in _optprob.obj.values():
                like.model.planets_dict = planets_dict
            
            # Pars
            _optprob.set_pars(p0)

            # Run the max like
            opt_result = _optprob.mapfit(save=False)
            
            # Alias best fit params
            pbest = opt_result['pbest']
            
            # Recompute the max like to NOT include any priors to keep things consistent.
            lnL = _optprob.obj.compute_logL(pbest)
            
            # Run the BIC
            bic = _optprob.optimizer.obj.compute_bic(pbest)
            
            # Run the AICc
            aicc = _optprob.optimizer.obj.compute_aicc(pbest)
            
            # Red chi 2
            redchi2 = _optprob.optimizer.obj.compute_redchi2(pbest)
            
            # Store
            model_comp_results.append({'planets_dict': planets_dict, 'lnL': lnL, 'bic': bic, 'aicc': aicc, 'pbest': pbest, 'redchi2': redchi2})
            
            del _optprob
            
        # Get the aicc and bic vals for each model
        aicc_vals = np.array([mcr['aicc'] for mcr in model_comp_results], dtype=float)
        
        # Sort according to aicc val (smaller is "better")
        ss = np.argsort(aicc_vals)
        model_comp_results = [model_comp_results[ss[i]] for i in range(len(ss))]
        
        # Grab the aicc and bic vals again
        aicc_vals = np.array([mcr['aicc'] for mcr in model_comp_results], dtype=float)
        bic_vals = np.array([mcr['bic'] for mcr in model_comp_results], dtype=float)
        
        # Compute aicc and bic vals
        aicc_diffs = np.abs(aicc_vals - np.nanmin(aicc_vals))
        bic_diffs = np.abs(bic_vals - np.nanmin(bic_vals))
        
        # Store diffs
        for i, mcr in enumerate(model_comp_results):
            mcr['delta_aicc'] = aicc_diffs[i]
            mcr['delta_bic'] = bic_diffs[i]
    
        # Save
        fname = self.output_path + self.star_name.replace(' ', '_') + '_modelcomp_' + pcutils.gendatestr(time=True) + "_" + self.tag + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(model_comp_results, f)
        
        return model_comp_results

    def get_components(self, pars):
        return self.obj.get_components(pars)
            
    ######################################
    #### Static Methods (all helpers) ####
    ######################################
    
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
        opt_result = optprob.optimize(save=False)
        return opt_result

    @staticmethod
    def _generate_all_wave_pairs(wave_vec):
        wave_vec_unq = np.sort(np.unique(wave_vec))
        return list(itertools.combinations(wave_vec_unq, 2))

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

    #####################################
    #### Properties (mostly aliases) ####
    #####################################
    
    @property
    def data_t(self):
        t = np.array([], dtype=float)
        for like in self.likes.values():
            t = np.concatenate((t, like.data.get_vec('t')))
        t = np.sort(t)
        return t
    
    @property
    def likes(self):
        return self.obj
    
    @property
    def post(self):
        return self.obj
    
    @property
    def like0(self):
        return self.obj.like0
    
    @property
    def planets_dict(self):
        return self.like0.model.planets_dict