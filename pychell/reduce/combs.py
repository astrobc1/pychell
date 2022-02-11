# Base python
import os

# Maths
import numpy as np
import scipy.interpolate
import scipy.constants as cs
import scipy.signal
from scipy.interpolate import LSQUnivariateSpline

# Graphics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pychell
plt.style.use(os.path.dirname(pychell.__file__) + os.sep + "gadfly_stylesheet.mplstyle")

# pychell deps
import pychell.maths as pcmath

################
#### 1d LSF ####
################

def compute_lsf_width_all(times_sci, times_lfc_cal, wls_cal_scifiber, lfc_cal_scifiber, f0, df, do_orders=None):
    """Wrapper to compute the LSF for all spectra.

    Args:
        times_sci (np.ndarray): The times of the science spectra.
        times_lfc_cal (np.ndarray): The times of the LFC calibration spectra.
        wls_cal_scifiber (np.ndarray): The wavelength grids for the LFC exposures for the science fiber with shape (n_pixels, n_orders, n_spectra)
        lfc_cal_scifiber (np.ndarray): The LFC spectra for LFC exposures for the science fiber with the same shape.
        do_orders (list, optional): Which orders to do. Defaults to all orders.

    Returns:
        np.ndarray: The LSF widths of each order and spectrum for the science spectra (science fiber) with shape=(n_orders, n_spectra).
    """

    # Numbers
    nx, n_orders, n_cal_spec = wls_cal_scifiber.shape
    n_sci_spec = len(times_sci)
    xarr = np.arange(nx).astype(float)
    if do_orders is None:
        do_orders = np.arange(1, n_orders + 1).astype(int).tolist()

    lsfwidths_cal_scifiber = np.full((n_orders, n_cal_spec), np.nan)
    lsfwidths_sci_scifiber = np.full((n_orders, n_sci_spec), np.nan)

    # Loop over orders
    for order_index in range(n_orders):

        if order_index + 1 not in do_orders:
            continue

        # Loop over calibration fiber spectra for sci fiber
        for i in range(n_cal_spec):
            try:
                lsfwidths_cal_scifiber[order_index, i] = compute_lsf_width(wls_cal_scifiber[:, order_index, i], lfc_cal_scifiber[:, order_index, i], f0, df)
            except:
                print(f"Warning! Could not compute LSF for order {order_index+1} cal spectrum {i+1}")

    for order_index in range(n_orders):

        if order_index + 1 not in do_orders:
            continue

        lsfwidths_sci_scifiber[order_index, :] = np.interp(times_sci, times_lfc_cal, lsfwidths_cal_scifiber[order_index, :], left=lsfwidths_cal_scifiber[order_index, 0], right=lsfwidths_cal_scifiber[order_index, -1])
    
    return lsfwidths_sci_scifiber

def compute_lsf_width(lfc_wave, lfc_flux, f0, df):
    """Computes the LSF width for a single order given the LFC spectrum.

    Args:
        lfc_wave (np.ndarray): The wavelength grid.
        lfc_flux (np.ndarray): The LFC flux.

    Returns:
        float: The LSF width.
    """

    # Flag bad pixels
    lfc_flux = flag_bad_pixels(lfc_flux)

    # Pixel grid
    nx = len(lfc_wave)
    xarr = np.arange(nx).astype(float)

    # Generate theoretical LFC peaks
    good = np.where(np.isfinite(lfc_wave) & np.isfinite(lfc_flux))[0]
    wi, wf = lfc_wave[good[0]], lfc_wave[good[-1]]
    integers, lfc_centers_wave_theoretical = gen_theoretical_peaks(wi, wf, f0, df)

    # Remove background flux
    background = estimate_background(lfc_wave, lfc_flux, f0, df)
    lfc_flux_no_bg = lfc_flux - background
    lfc_peak_max = pcmath.weighted_median(lfc_flux_no_bg, percentile=0.75)
    
    # Estimate continuum
    continuum = estimate_continuum(lfc_wave, lfc_flux_no_bg, f0, df)
    lfc_flux_norm = lfc_flux_no_bg / continuum

    # Peak spacing in wavelength space
    peak_spacing = np.polyval(np.polyfit(lfc_centers_wave_theoretical[1:], np.diff(lfc_centers_wave_theoretical), 1), lfc_centers_wave_theoretical)

    # Loop over theoretical peaks and shift
    waves_all = []
    flux_all = []
    for i in range(len(lfc_centers_wave_theoretical)):
        use = np.where((lfc_wave >= lfc_centers_wave_theoretical[i] - peak_spacing[i] / 2) & (lfc_wave < lfc_centers_wave_theoretical[i] + peak_spacing[i] / 2))[0]
        if len(use) >= 5:
            waves_all += list(lfc_wave[use] - lfc_centers_wave_theoretical[i])
            flux_all += list(lfc_flux_no_bg[use] / np.nansum(lfc_flux_no_bg[use]))
    
    # Prep for Gaussian fit
    waves_all = np.array(waves_all, dtype=float)
    flux_all = np.array(flux_all, dtype=float)
    ss = np.argsort(waves_all)
    waves_all = waves_all[ss]
    flux_all = flux_all[ss]
    good = np.where(np.isfinite(waves_all) & np.isfinite(flux_all))[0]
    waves_all = waves_all[good]
    flux_all = flux_all[good]

    p0 = [pcmath.weighted_median(flux_all, percentile=0.95), 0.1]
    bounds = [(0.5 * p0[0], 1.5 * p0[0]), (0.005, 1)]
    opt_result = scipy.optimize.minimize(solve_lsf_model, p0, args=(waves_all, flux_all), method='Nelder-Mead', bounds=bounds)
    pbest = opt_result.x
    sigma = pbest[1]
    #breakpoint()
    #matplotlib.use("MacOSX"); plt.plot(lfc_wave, lfc_flux); plt.vlines(lfc_centers_wave_theoretical, ymin=0, ymax=np.nanmax(lfc_flux), color='red')

    return sigma


#############
#### WLS ####
#############

def compute_wls_all(f0, df, times_sci, times_lfc_cal, wave_estimate_scifiber, wave_estimate_calfiber, lfc_sci_calfiber, lfc_cal_calfiber, lfc_cal_scifiber, do_orders=None, mode="static", poly_order=4, n_cores=1, static_index=0):
    """Wrapper to compute the wavelength solution for all spectra.

    Args:
        times_sci (np.ndarray): The times of the science spectra.
        times_lfc_cal (np.ndarray): The times of the LFC calibration spectra.
        wave_estimate_scifiber (np.ndarray): The approximate wavelength grid for the science fiber with shape (n_pixels, n_orders).
        wave_estimate_calfiber (np.ndarray): The approximate wavelength grid for the calibration fiber with the same shape.
        lfc_sci_calfiber (np.ndarray): The LFC spectra for the science exposures for the calibration fiber with shape (n_pixels, n_orders, n_spectra).
        lfc_cal_calfiber (np.ndarray): The LFC spectra for the LFC exposures for the calibraiton fiber with the same shape.
        lfc_cal_scifiber (np.ndarray): The LFC spectra for the LFC exposures for the science fiber with the same shape.
        do_orders (list, optional): Which orders to do. Defaults to all orders.
        mode (str): How to use the appropriate wavelength solutions compute the science fiber wls. Defaults to "nearest", which uses the closest wavelength solution in time for the science fiber.

    Returns:
        np.ndarray: The wavelength solutions for the science spectra, science fiber with shape (n_pixels, n_orders, n_spectra).
        np.ndarray: The wavelength solutions for the science spectra, calibration fiber with the same shape.
        np.ndarray: The wavelength solutions for the cal spectra, science fiber with the same shape.
        np.ndarray: The wavelength solutions for the cal spectra, cal fiber with the same shape.
    """

    # Numbers
    nx, n_orders = wave_estimate_scifiber.shape
    xarr = np.arange(nx).astype(float)
    if do_orders is None:
        do_orders = np.arange(1, n_orders + 1).astype(int).tolist()

    # Sci
    if times_sci is not None:
        n_sci_spec = len(times_sci)
        wls_sci_scifiber = np.full((nx, n_orders, n_sci_spec), np.nan)
        wls_sci_calfiber = np.full((nx, n_orders, n_sci_spec), np.nan)
    else:
        n_sci_spec = 0
        wls_sci_calfiber = None
        wls_sci_scifiber = None
    
    # Cal
    n_cal_spec = len(times_lfc_cal)
    wls_cal_scifiber = np.full((nx, n_orders, n_cal_spec), np.nan)
    wls_cal_calfiber = np.full((nx, n_orders, n_cal_spec), np.nan)

    # Loop over orders
    for order_index in range(n_orders):

        if order_index + 1 not in do_orders:
            continue

        # Loop over calibration fiber spectra and compute wls for both fibers
        for i in range(n_cal_spec):
            print(f"Computing sci fiber wls for order {order_index+1} cal spectrum {i+1}")
            wls_cal_scifiber[:, order_index, i] = compute_wls(wave_estimate_scifiber[:, order_index], lfc_cal_scifiber[:, order_index, i], f0, df, poly_order)
            print(f"Computing cal fiber wls for order {order_index+1} cal spectrum {i+1}")
            wls_cal_calfiber[:, order_index, i] = compute_wls(wave_estimate_calfiber[:, order_index], lfc_cal_calfiber[:, order_index, i], f0, df, poly_order)

        # Loop over science spectra and compute wls for cal fiber
        if times_sci is not None:
            for i in range(n_sci_spec):
                print(f"Computing cal fiber wls for order {order_index+1} science spectrum {i+1}")
                try:
                    wls_sci_calfiber[:, order_index, i] = compute_wls(wave_estimate_calfiber[:, order_index], lfc_sci_calfiber[:, order_index, i], f0, df, poly_order)
                except:
                    pass

        # Loop over science observations, computer wls for science fiber
        if times_sci is not None:
            for i in range(n_sci_spec):
                print(f"Computing sci fiber wls for order {order_index+1} science spectrum {i+1}")
                if mode == "nearest":
                    k_cal_nearest = np.nanargmin(np.abs(times_sci[i] - times_lfc_cal))
                    wls_sci_scifiber[:, order_index, i] = np.copy(wls_cal_scifiber[:, order_index, k_cal_nearest])
                elif mode == "static":
                    wls_sci_scifiber[:, order_index, i] = np.copy(wls_sci_scifiber[:, order_index, static_index])
                elif mode == "adopt_cal":
                    wls_sci_scifiber[:, order_index, i] = np.copy(wls_sci_calfiber[:, order_index, i])
                else:
                    raise ValueError("mode must be nearest, static, or interp")

    return wls_sci_scifiber, wls_sci_calfiber, wls_cal_scifiber, wls_cal_calfiber

def compute_wls(wave_estimate, lfc_flux, f0, df, poly_order=None, xrange=None):
    """Computes the wavelength solution from the LFC spectrum.

    Args:
        wave_estimate (np.ndarray): The approximate wavelength grid.
        lfc_flux (np.ndarray): The LFC spectrum.
        df (np.ndarray): The frequency of the LFC in GHz.
        f0 (np.ndarray): The frequency of the pump line in GHz.
        poly_order (int, optional): The polynomial order to fit the LFC line centers with.
        xrange (list, optional): The range to consider in pixels for each order. Defaults to the whole order.

    Returns:
        np.ndarray: The wavelength solution.
    """

    # Number of pixels and range to consider for each order
    nx = len(wave_estimate)
    xarr = np.arange(nx)
    if xrange is None:
        xrange = [0, nx-1]
    wave_range = [wave_estimate[xrange[0]], wave_estimate[xrange[1]]]

    # Flag bad pixels
    lfc_flux = flag_bad_pixels(lfc_flux)

    # Compute centers (wave, pix)
    lfc_centers_pix, lfc_centers_wave, _, _ = compute_peaks(wave_estimate, lfc_flux, f0, df, xrange)

    # Polynomial fit to peaks
    wls = fit_peaks(xarr, lfc_centers_pix, lfc_centers_wave, poly_order=poly_order)

    return wls

def estimate_peak_spacing(xi, xf, wi, wf, f0, df):
    """Estimates the peak spacing in detector pixels.

    Args:
        xi (float): The lower pixel.
        xf (float): The upper pixel.
        wi (float): The lower wavelength.
        wf (float): The upper wavelength.
        f0 (float): The frequency of the LFC pump line in GHz.
        df (float): The LFC line spacing in GHz.
    """
    integers, lfc_centers_wave_theoretical = gen_theoretical_peaks(wi, wf, f0, df)
    n_peaks = len(lfc_centers_wave_theoretical)
    peak_spacing = (xf - xi) / n_peaks
    return peak_spacing

def compute_peaks(wave_estimate, lfc_flux, f0, df, xrange, peak_model="gaussian", sigma_guess=[0.2, 1.4, 3.0], mu_guess=[0.5, 1E-2, 0.5]):
    """Computes the pixel and corresponding wavelength values for each LFC spot peak.

    Args:
        wave_estimate (np.ndarray): The approximate wavelength grid.
        lfc_flux (np.ndarray): The LFC flux.
        f0 (float): The LFC pump line GHz.
        df (float): The LFC spacing in GHz.
        xrange (list): The lower and upper pixel bounds to consider.

    Returns:
        np.ndarray: The pixel centers of each peak
        np.ndarray: The wavelength centers of each peak
        np.ndarray: The integers relative to f0.
    """

    # Generate theoretical LFC peaks
    good = np.where(np.isfinite(wave_estimate) & np.isfinite(lfc_flux))[0]
    xi, xf = good[0], good[-1]
    wwi, wwf = wave_estimate[xi], wave_estimate[xf]
    lfc_peak_integers, lfc_centers_wave_theoretical = gen_theoretical_peaks(wwi, wwf, f0, df)
    peak_spacing = estimate_peak_spacing(xi, xf, wwi, wwf, f0, df)
    
    # Number of pixels
    nx = len(wave_estimate)
    xarr = np.arange(nx)
    pfit_estimate = np.polyfit(xarr[good], wave_estimate[good], 4)

    # Remove background flux
    background = estimate_background(wave_estimate, lfc_flux, f0, df)
    lfc_flux_no_bg = lfc_flux - background
    lfc_peak_max = pcmath.weighted_median(lfc_flux_no_bg, percentile=0.99)
    
    # Estimate continuum
    continuum = estimate_continuum(wave_estimate, lfc_flux_no_bg, f0, df)
    lfc_flux_norm = lfc_flux_no_bg / continuum

    # Estimate peaks in pixel space (just indices)
    peaks = scipy.signal.find_peaks(lfc_flux_norm[xrange[0]:xrange[1]+1], height=np.full(xrange[1] - xrange[0] + 1, 0.75), distance=0.8*peak_spacing)[0]
    peaks = np.sort(peaks + xrange[0])

    # Estimate spacing between peaks, assume linear trend across order
    peak_spacing = np.polyval(np.polyfit(peaks[1:], np.diff(peaks), 1), xarr)

    # Only consider peaks with enough flux
    good_peaks = []
    for peak in peaks:
        if lfc_flux_no_bg[peak] >= 0.2 * lfc_peak_max:
            good_peaks.append(peak)
    del good_peaks[0], good_peaks[-1]
    good_peaks = np.array(good_peaks)

    # Fit each peak with a Gaussian
    lfc_centers_pix = np.full(good_peaks.size, np.nan)
    rms_norm = np.full(good_peaks.size, np.nan)
    sigmas = np.full(len(good_peaks), np.nan)
    for i in range(len(good_peaks)):

        # Region to consider
        use = np.where((xarr >= good_peaks[i] - peak_spacing[good_peaks[i]] / 2) & (xarr < good_peaks[i] + peak_spacing[good_peaks[i]] / 2))[0]

        # Crop data
        xx, yy = np.copy(xarr[use]), np.copy(lfc_flux[use])

        # Normalize lfc flux to max
        yy -= np.nanmin(yy)
        yy /= np.nanmax(yy)

        # Pars and bounds
        p0 = []
        bounds = []

        # Amp
        p0.append(1.0)
        bounds.append((0.5, 1.5))

        # Mu
        p0.append(good_peaks[i] + mu_guess[1])
        bounds.append((good_peaks[i] - mu_guess[0], good_peaks[i] + mu_guess[2]))

        # Sigma or fwhm
        if peak_model.lower() == "gaussian":
            p0.append(sigma_guess[1])
            bounds.append((sigma_guess[0], sigma_guess[2]))
        else:
            p0.append(sigma_guess[1] * 2.355)
            bounds.append((sigma_guess[0] * 2.355, sigma_guess[2] * 2.355))

        # Offset
        p0.append(0.1)
        bounds.append((-0.2, 0.2))

        # Fit
        if peak_model.lower() == "gaussian":
            opt_result = scipy.optimize.minimize(solve_fit_peak1d_gaussian, p0, args=(xx, yy), method='Nelder-Mead', bounds=bounds, tol=1E-8)
        else:
            opt_result = scipy.optimize.minimize(solve_fit_peak1d_lorentz, p0, args=(xx, yy), method='Nelder-Mead', bounds=bounds, tol=1E-8)

        # Results
        pbest = opt_result.x
        lfc_centers_pix[i] = pbest[1]
        sigmas[i] = pbest[2]
        rms_norm[i] = opt_result.fun

    # Determine which LFC spot matches each peak
    lfc_centers_wave = []
    peak_integers = []
    for i in range(len(lfc_centers_pix)):
        diffs = np.abs(np.polyval(pfit_estimate, lfc_centers_pix[i]) - lfc_centers_wave_theoretical)
        k = np.argmin(diffs)
        lfc_centers_wave.append(lfc_centers_wave_theoretical[k])
        peak_integers.append(lfc_peak_integers[k])
    lfc_centers_wave = np.array(lfc_centers_wave)
    peak_integers = np.array(peak_integers)
    return lfc_centers_pix, lfc_centers_wave, rms_norm, peak_integers

def gen_theoretical_peaks(wi, wf, f0, df):
    """Generates the theoretical wavelengths given a range of wavelengths.

    Args:
        wi (float): The lower wavelength.
        wf (float): The upper wavelength.
        f0 (float): The frequency of the pump line in Hz.
        df (float): The line spacing in Hz.

    Returns:
        np.ndarray: The LFC line centers in wavelength.
    """

    # Generate the frequency grid
    n_left, n_right = 10_000, 10_000
    lfc_centers_freq_theoretical = np.arange(f0 - n_left * df, f0 + (n_right + 1) * df, df)
    integers = np.arange(-n_left, n_right + 1).astype(int)

    # Convert to wavelength
    lfc_centers_wave_theoretical = cs.c / lfc_centers_freq_theoretical
    lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[::-1] * 1E10
    integers = integers[::-1]

    # Only peaks within the bounds
    good = np.where((lfc_centers_wave_theoretical > wi - 0.1) & (lfc_centers_wave_theoretical < wf + 0.1))[0]
    lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[good]
    integers = integers[good]

    return integers, lfc_centers_wave_theoretical

def estimate_background(lfc_wave, lfc_flux, f0, df):
    """Estimates the background of the LFC spectrum through a median filter.

    Args:
        lfc_wave (np.ndarray): The LFC wave grid.
        lfc_flux (np.ndarray): The LFC flux grid.

    Returns:
        np.ndarray: The background
    """
    good = np.where(np.isfinite(lfc_wave) & np.isfinite(lfc_flux))[0]
    xi, xf = good[0], good[-1]
    wi, wf = lfc_wave[xi], lfc_wave[xf]
    peak_spacing = estimate_peak_spacing(xi, xf, wi, wf, f0, df)
    background = pcmath.generalized_median_filter1d(lfc_flux, width=int(2 * peak_spacing), percentile=0.01)
    return background

def estimate_continuum(lfc_wave, lfc_flux, f0, df):
    """Estimates the continuum of the LFC spectrum through a median filter.

    Args:
        lfc_wave (np.ndarray): The LFC wave grid.
        lfc_flux (np.ndarray): The LFC flux grid.

    Returns:
        np.ndarray: The continuum
    """
    good = np.where(np.isfinite(lfc_wave) & np.isfinite(lfc_wave))[0]
    xi, xf = good[0], good[-1]
    wi, wf = lfc_wave[xi], lfc_wave[xf]
    peak_spacing = estimate_peak_spacing(xi, xf, wi, wf, f0, df)
    continuum = pcmath.generalized_median_filter1d(lfc_flux, width=int(2 * peak_spacing), percentile=0.99)
    return continuum

def flag_bad_pixels(lfc_flux, smooth_width=3, sigma_thresh=9):
    """Wrapper to flag bad pixels in the LFC spectrum.

    Args:
        lfc_flux (np.ndarray): The LFC flux.
        smooth_width (int, optional): The width of the median filter (in pixels) to smooth the LFC with. Defaults to 3.
        sigma_thresh (int, optional): The threshold for bad pixels. Defaults to 9.

    Returns:
        np.ndarray: The LFC flux with bad pixels flagged as np.nan.
    """

    # First remove negatives
    lfc_flux_out = np.copy(lfc_flux)
    bad = np.where(lfc_flux_out < 0)[0]
    if bad.size > 0:
        lfc_flux_out[bad] = np.nan

    # Identify bad pixels in lfc flux
    lfc_flux_smooth = pcmath.median_filter1d(lfc_flux_out, width=smooth_width)
    rel_errors = (lfc_flux_out - lfc_flux_smooth) / lfc_flux_smooth
    bad = np.where(np.abs(rel_errors) > sigma_thresh * np.nanstd(rel_errors))[0]
    if bad.size > 0:
        lfc_flux_out[bad] = np.nan
    return lfc_flux_out

######################
#### PEAK FITTING ####
######################

def fit_peaks(pixel_arr, lfc_centers_pix, lfc_centers_wave, poly_order=6):
    """Simple wrapper to fit a polynomial to the lfc centers for a single order.

    Args:
        pixel_arr ([type]): The pixel array to return the polynomial wls on.
        lfc_centers_pix ([type]): The LFC line centers in pixel space.
        lfc_centers_wave ([type]): The corresponding LFC line centers in wavelength space.
        poly_order (int, optional): The order of the polynomial to fit the LFC line centers with. Defaults to 6.

    Returns:
        np.ndarray: The nominal wls from the polynomial fit on the pixel_arr grid.
    """
    pfit = np.polyfit(lfc_centers_pix, lfc_centers_wave, poly_order)
    wls = np.polyval(pfit, pixel_arr)
    return wls

def solve_fit_peak1d_gaussian(pars, x, data):
    model = pcmath.gauss(x, *pars[0:3]) + pars[3]
    rms = pcmath.rmsloss(data, model)
    return rms

def solve_fit_peak1d_lorentz(pars, x, data):
    model = pcmath.lorentz(x, *pars[0:3]) + pars[3]
    rms = pcmath.rmsloss(data, model)
    return rms

def solve_lsf_model(pars, x, data):
    model = pcmath.gauss(x, pars[0], 0, pars[1])
    rms = pcmath.rmsloss(data, model)
    return rms

def solve_fit_peak2d(pars, xarr, yarr, image):
    model = peak_model2d(pars, xarr, yarr)
    rms = pcmath.rmsloss(image.flatten(), model.flatten())
    return rms

def peak_model2d(pars, xarr, yarr):

    model = np.full((len(yarr), len(xarr)), np.nan)
    
    amp = pars[0]
    xmu = pars[1]
    ymu = pars[2]
    sigma = pars[3]
    q = pars[4]
    theta = pars[5]
    offset = pars[6]

    # Compute PSF
    for i in range(len(yarr)):
        for j in range(len(xarr)):
            xp = (xarr[j] - xmu) * np.sin(theta) - (yarr[i] - ymu) * np.cos(theta)
            yp = (xarr[j] - xmu) * np.cos(theta) + (yarr[i] - ymu) * np.sin(theta)
            model[i, j] = amp * np.exp(-0.5 * ((xp / sigma)**2 + (yp / (sigma * q))**2)) + offset

    # Return model
    return model