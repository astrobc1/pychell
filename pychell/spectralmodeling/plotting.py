import os

import numpy as np

import matplotlib
#matplotlib.use("AGG")
import matplotlib.pyplot as plt

import pychell.maths as pcmath


def plot_rvs_single_chunk(specrvprob, rvs, iter_index, output_path, time_offset=2450000):
    
    # Plot the rvs, binned rvs, xcorr rvs, xcorr binned rvs
    plot_width, plot_height = 1600, 600
    dpi = 200
    plt.figure(num=1, figsize=(int(plot_width / dpi), int(plot_height / dpi)), dpi=200)
    
    # Individual Forward Model
    plt.plot(rvs["bjds"] - time_offset, rvs["rvsfwm"][:, iter_index] - np.nanmedian(rvs["rvsfwm"][:, iter_index]), marker='o', linewidth=0, color=(0.1, 0.8, 0.1), alpha=0.7, label="Forward model")

    # Individual XC
    if "rvsxc" in rvs:
        plt.plot(rvs["bjds"] - time_offset, rvs["rvsxc"][:, iter_index] - np.nanmedian(rvs["rvsxc"][:, iter_index]), marker='X', linewidth=0, color='black', alpha=0.5, label="XC")
    
    # Plot labels
    plt.title(f"{specrvprob.model.star.star_name.replace('_', ' ')}, {specrvprob.model.sregion.label}, iteration {iter_index + 1}")
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel(f"BJDTDB - {time_offset}")
    plt.ylabel('Relative RV [m/s]')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    fname = f"{output_path}RVs{os.sep}rvs_{specrvprob.model.sregion.label.lower()}_iter{iter_index + 1}.png"
    plt.savefig(fname)
    plt.close()


def quick_bin(rvs, indices):
    rvs_binned = np.full(len(indices), np.nan)
    rverrs_binned = np.full(len(indices), np.nan)
    for i in range(len(indices)):
        rvs_out[i], rverrs_out[i] = pcmath.weighted_combine(rvs[f:l])
    return rvs_binned, rverrs_binned



def plot_spectum_fit(data, model, pars, obj, iter_index, output_path):
    
    # Figure dims for 1 chunk
    fig_width, fig_height = 2000, 720
    dpi = 200
    figsize = int(fig_width / dpi), int(fig_height / dpi)
    
    # Create subplot
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Alias high res grid
    wave_hr = model.templates["wave"]
    
    # Build the model
    wave_data, model_lr = model.build(pars, data)
    
    # The residuals for this iteration
    residuals = data.flux  - model_lr

    # Get the mask
    mask = np.copy(data.mask)
    
    # Ensure known bad pixels are nans in the residuals
    residuals[mask == 0] = np.nan
    
    # Change edges to nans
    if obj.remove_edges > 0:
        good = np.where(mask == 1)[0]
        mask[good[0:obj.remove_edges]] = 0
        mask[good[-obj.remove_edges:]] = 0
        residuals[mask == 0] = np.nan

    # Left and right padding
    good = np.where(np.isfinite(wave_data) & (mask == 1))[0]
    pad = 0.01 * (wave_data[good].max() - wave_data[good].min())
    
    # Data
    plt.plot(wave_data, data.flux, color=(0, 114/255, 189/255), lw=0.8, label="Data")
    
    # Model
    plt.plot(wave_data, model_lr, color=(217/255, 83/255, 25/255), lw=0.8, label="Model")
    
    # Zero line and -0.2 line
    plt.plot(wave_data, np.zeros_like(wave_data), color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':')
    plt.plot(wave_data, np.zeros_like(wave_data) - 0.2, color=(89/255, 23/255, 130/255), lw=0.8, linestyle=':')
    
    # Residuals and worst pixels which were flagged
    plt.plot(wave_data, residuals, color=(255/255, 169/255, 22/255), lw=0.8, label="Residuals")
    
    # Star
    if model.star is not None:
        
        # Initial star
        if not model.star.from_flat and iter_index != 0:
            star_flux = pcmath.doppler_shift_flux(wave_hr, model.templates["star"], pars[model.star.par_names[0]].value)
            if model.lsf is not None:
                star_flux = model.lsf.convolve(star_flux, pars=pars)
            plt.plot(wave_hr, star_flux - 1.2, label='Initial Star', lw=0.8, color='aqua', alpha=0.6)
        
        # Current star
        star_flux = model.star.build(pars, model.templates)
        if model.lsf is not None:
            star_flux = model.lsf.convolve(star_flux, pars=pars)
        plt.plot(wave_hr, star_flux - 1.2, label='Current Star', lw=0.8, color='deeppink', alpha=0.8)
    
    # Tellurics
    if model.tellurics is not None:
        tell_flux = model.tellurics.build(pars, model.templates)
        if model.lsf is not None:
            tell_flux = model.lsf.convolve(tell_flux, pars=pars)
        plt.plot(wave_hr, tell_flux - 1.2, label='Tellurics', lw=0.8, color='indigo', alpha=0.2)
    
    # Gas Cell
    if model.gascell is not None:
        gas_flux = model.gascell.build(pars, model.templates)
        if model.lsf is not None:
            gas_flux = model.lsf.convolve(gas_flux, pars=pars)
        plt.plot(wave_hr, gas_flux - 1.2, label='Gas Cell', lw=0.8, color='green', alpha=0.2)
    
    # X and Y limits
    plt.xlim(model.sregion.wavemin - pad, model.sregion.wavemax + pad)
    plt.ylim(-1.3, 1.2)
        
    # Legend
    plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1.0, 0.5))
        
    # X and Y tick parameters
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=10)
    
    # X and Y axis labels
    plt.xlabel("Wavelength [nm]", fontsize=10)
    plt.ylabel("Norm flux", fontsize=10)
    
    # The title
    plt.title(f"{model.star.star_name.replace('_', ' ')}, {model.sregion.label}, iteration {iter_index + 1}", fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    fname = f"{output_path}Fits{os.sep}{data}_{model.sregion.label.lower()}_iter{iter_index + 1}.png"
    plt.savefig(fname)
    plt.close()