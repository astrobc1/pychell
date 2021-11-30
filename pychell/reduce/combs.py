# Maths
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.constants as cs
import scipy.signal
from scipy.interpolate import LSQUnivariateSpline

# pychell deps
import pychell.maths as pcmath

#########################
#### PRIMARY METHODS ####
#########################

def compute_psf2d(image, badpix_mask, trace_positions, wave_estimate, f0, df, peak_thresh):
    """Fits 2d tilted Gaussian to each peak in the 2d image.

    Args:
        image (np.ndarray): The image.
        trace_dict (dict): The trace dictionary.
        wave_estimate (np.ndarray): An estimate of the wavelength grid for this order.
        f0 (float): The frequency of the pump line.
        df (float): The comb line spacing.
    """


    ny, nx = image.shape
    yarr = np.arange(ny)
    xarr = np.arange(nx)

    # Generate theoretical LFC peaks
    good = np.where(np.isfinite(wave_estimate))[0]
    xi, xf = good[0], good[-1]
    wi, wf = wave_estimate[xi], wave_estimate[xf]
    lfc_peak_integers, lfc_centers_wave_theoretical = gen_theoretical_peaks(wi, wf, f0, df)
    n_spots = len(lfc_centers_wave_theoretical)
    peak_spacing = estimate_peak_spacing(xi, xf, wi, wf, f0, df)
    
    # Number of pixels
    nx = len(wave_estimate)
    xarr = np.arange(nx)
    pfit_estimate = np.polyfit(xarr[good], wave_estimate[good], 4)

    # Estimate spec1d
    lfc_flux = simple_extract(image, badpix_mask, trace_positions)

    # Remove background flux
    background = estimate_background(wave_estimate, lfc_flux, f0, df)
    lfc_flux_no_bg = lfc_flux - background
    lfc_peak_max = pcmath.weighted_median(lfc_flux_no_bg, percentile=0.99)
    
    # Estimate continuum
    continuum = estimate_continuum(wave_estimate, lfc_flux_no_bg, f0, df)
    lfc_flux_norm = lfc_flux_no_bg / continuum

    # Estimate peaks in pixel space (just indices)
    peaks = scipy.signal.find_peaks(lfc_flux_norm, height=np.full(nx, 0.75), distance=0.8*peak_spacing)[0]
    peaks = np.sort(peaks)

    # Estimate spacing between peaks, assume linear trend across order
    peak_spacing = np.polyval(np.polyfit(peaks[1:], np.diff(peaks), 1), xarr)

    # Only consider peaks with enough flux
    good_peaks = []
    for peak in peaks:
        if np.nanmax(image[:, peak-1:peak+2]) >= peak_thresh:
            good_peaks.append(peak)
    good_peaks = np.array(good_peaks)

    # Storage arrays for best fit params
    amps = np.full(good_peaks.size, np.nan) # 0
    lfc_centers_pix_x = np.full(good_peaks.size, np.nan) # 1
    sigmas = np.full(good_peaks.size, np.nan) # 3
    qs = np.full(good_peaks.size, np.nan) # 4
    thetas = np.full(good_peaks.size, np.nan) # 5
    rms_norm = np.full(good_peaks.size, np.nan)

    # Fit each peak with a Gaussian
    for i in range(len(good_peaks)):
        print(i, len(good_peaks))

        # Window
        usex = np.where((xarr >= good_peaks[i] - peak_spacing[good_peaks[i]] / 2) & (xarr < good_peaks[i] + peak_spacing[good_peaks[i]] / 2))[0]
        xx = np.copy(xarr[usex])
        yy = np.arange(trace_positions[good_peaks[i]] - int(len(usex) / 2), trace_positions[good_peaks[i]] + int(len(usex) / 2) + 1).astype(int)

        # Data
        zz = image[yy.min():yy.max()+1, usex.min():usex.max()+1]

        # Initial pars
        p0 = [np.nanmax(zz), # amp
              xarr[good_peaks[i]], # mux
              trace_positions[good_peaks[i]], # muy
              1.4, # sigma
              0.8, # q
              np.pi/4, # theta
              3] # offset
        
        # Bounds
        bounds = [(0.5 * p0[0], 1.5 * p0[0]), # amp
                  (p0[1] - 1, p0[1] + 1), # mux
                  (p0[2] - 1, p0[2] + 1), # muy
                  (0.5, 4), # sigma
                  (0.4, 0.9), # q
                  (0, np.pi/2), # theta
                  (0, 10)] # offset

        # Fit
        opt_result = scipy.optimize.minimize(fit_peak2d, p0, args=(xx, yy, zz), method='Nelder-Mead', bounds=bounds)

        # Get best fit params
        pbest = opt_result.x
        lfc_centers_pix_x[i] = pbest[1]
        sigmas[i] = pbest[3]
        qs[i] = pbest[4]
        thetas[i] = pbest[5]

        # Store rms of fit
        rms_norm[i] = opt_result.fun / pbest[0]

    return rms_norm, lfc_centers_pix_x, sigmas, thetas, qs

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
    bounds = [(0.5 * p0[0], 1.5 * p0[0]), (0.01, 1)]
    opt_result = scipy.optimize.minimize(fit_lsf, p0, args=(waves_all, flux_all), method='Nelder-Mead', bounds=bounds)
    pbest = opt_result.x
    sigma = pbest[1]

    return sigma

def compute_wls_all(f0, df, times_sci, times_lfc_cal, wave_estimate_scifiber, wave_estimate_calfiber, lfc_sci_calfiber, lfc_cal_calfiber, lfc_cal_scifiber, do_orders=None, method="nearest", poly_order=6):
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
        method (str): How to use the appropriate wavelength solutions compute the science fiber wls. Defaults to "nearest", which uses the closest wavelength solution in time for the science fiber.

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
            print(f"Computing cal fiber wls for order {order_index+1} cal spectrum {i+1}")
            try:
                wls_cal_scifiber[:, order_index, i] = compute_wls(wave_estimate_scifiber[:, order_index], lfc_cal_scifiber[:, order_index, i], df, f0, poly_order)
            except:
                print(f"Warning! Could not compute wls for order {order_index+1} cal spectrum {i+1}")
            
            try:
                wls_cal_calfiber[:, order_index, i] = compute_wls(wave_estimate_calfiber[:, order_index], lfc_cal_calfiber[:, order_index, i], df, f0, poly_order)
            except:
                print(f"Warning! Could not compute wls for order {order_index+1} cal spectrum {i+1}")

        # Loop over science spectra and compute wls for cal fiber
        if times_sci is not None:
            for i in range(n_sci_spec):
                print(f"Computing cal fiber wls for order {order_index+1} science spectrum {i+1}")
                try:
                    wls_sci_calfiber[:, order_index, i] = compute_wls(wave_estimate_calfiber[:, order_index], lfc_sci_calfiber[:, order_index, i], df, f0, poly_order)
                except:
                    print(f"Warning! Could not compute wls for order {order_index+1} science spectrum {i+1}")

        # Loop over science observations, computer wls for science fiber
        if times_sci is not None:
            for i in range(n_sci_spec):
                print(f"Computing sci fiber wls for order {order_index+1} science spectrum {i+1}")
                if method == "interp":
                    for x in range(nx):
                        wls_sci_scifiber[x, order_index, i] = np.interp(times_sci[i], times_lfc_cal, wls_cal_scifiber[x, order_index, :], left=wls_cal_scifiber[x, order_index, 0], right=wls_cal_scifiber[x, order_index, -1])
                elif method == "nearest":
                    k_cal_nearest = np.nanargmin(np.abs(times_sci[i] - times_lfc_cal))
                    wls_sci_scifiber[:, order_index, i] = np.copy(wls_cal_scifiber[:, order_index, k_cal_nearest])
                else:
                    raise ValueError("method must be nearest or interp")

    return wls_sci_scifiber, wls_sci_calfiber, wls_cal_scifiber, wls_cal_calfiber

def compute_wls(wave_estimate, lfc_flux, df, f0, poly_order=None):
    """Computes the wavelength solution from the LFC spectrum.

    Args:
        wave_estimate (np.ndarray): The approximate wavelength grid.
        lfc_flux (np.ndarray): The LFC spectrum.
        df (np.ndarray): The frequency of the LFC in GHz.
        f0 (np.ndarray): The frequency of the pump line in GHz.
        poly_order (int, optional): The polynomial order to fit the LFC line centers with.

    Returns:
        np.ndarray: The wavelength solution.
    """

    # Number of pixels
    nx = len(wave_estimate)
    xarr = np.arange(nx)

    # Flag bad pixels
    lfc_flux = flag_bad_pixels(lfc_flux)

    # Compute centers (wave, pix)
    lfc_centers_pix, lfc_centers_wave, _ = compute_peaks(wave_estimate, lfc_flux, f0, df)

    # Polynomial fit to peaks
    wls = fit_peaks(xarr, lfc_centers_pix, lfc_centers_wave, poly_order=poly_order)

    return wls


def simple_extract(image, badpix_mask, trace_positions):

    ny, nx = image.shape

    yarr = np.arange(ny)


    yarr_zero_center = np.arange(int(np.floor(-ny / 2)), int(np.ceil(ny / 2)) + 1)
    image_rect = np.full((len(yarr_zero_center), nx), np.nan)

    # Rectify
    for x in range(nx):
        image_rect[:, x] = pcmath.lin_interp(yarr - trace_positions[x], image[:, x], yarr_zero_center)
        image_rect[:, x] /= np.nansum(image_rect[:, x])

    # Trace profile crude estimate
    trace_profile = np.nanmedian(image_rect, axis=1)

    # Simple boxcar extract
    spec1d = np.full(nx, np.nan)
    for x in range(nx):
        weights = badpix_mask[:, x]
        tp = pcmath.lin_interp(yarr_zero_center + trace_positions[x], trace_profile, yarr)
        spec1d[x] = np.nansum(image[:, x]) / np.nansum(weights) / np.nansum(tp * weights, axis=0)

    bad = np.where(~np.isfinite(spec1d) | (spec1d == 0))[0]
    spec1d[bad] = np.nan

    return spec1d


######################
#### PEAK FITTING ####
######################

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

def compute_peaks(wave_estimate, lfc_flux, f0, df):
    """Computes the pixel and corresponding wavelength values for each LFC spot peak.

    Args:
        wave_estimate (np.ndarray): The approximate wavelength grid.
        lfc_flux (np.ndarray): The LFC flux.
        f0 (float): The LFC pump line GHz.
        df (float): The LFC spacing in GHz.

    Returns:
        np.ndarray: The pixel centers of each peak
        np.ndarray: The wavelength centers of each peak
        np.ndarray: The integers relative to f0.
    """

    # Generate theoretical LFC peaks
    good = np.where(np.isfinite(wave_estimate) & np.isfinite(lfc_flux))[0]
    xi, xf = good[0], good[-1]
    wi, wf = wave_estimate[xi], wave_estimate[xf]
    lfc_peak_integers, lfc_centers_wave_theoretical = gen_theoretical_peaks(wi, wf, f0, df)
    peak_spacing = estimate_peak_spacing(xi, xf, wi, wf, f0, df)
    
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
    peaks = scipy.signal.find_peaks(lfc_flux_norm, height=np.full(nx, 0.75), distance=0.8*peak_spacing)[0]
    peaks = np.sort(peaks)

    # Estimate spacing between peaks, assume linear trend across order
    peak_spacing = np.polyval(np.polyfit(peaks[1:], np.diff(peaks), 1), xarr)

    # Only consider peaks with enough flux
    good_peaks = []
    for peak in peaks:
        if lfc_flux_no_bg[peak] >= 0.2 * lfc_peak_max:
            good_peaks.append(peak)
    good_peaks = np.array(good_peaks)

    # Fit each peak with a Gaussian
    lfc_centers_pix = np.full(good_peaks.size, np.nan)
    rms_norm = np.full(good_peaks.size, np.nan)
    for i in range(len(good_peaks)):
        use = np.where((xarr >= good_peaks[i] - peak_spacing[good_peaks[i]] / 2) & (xarr < good_peaks[i] + peak_spacing[good_peaks[i]] / 2))[0]
        xx, yy = np.copy(xarr[use]), np.copy(lfc_flux[use])
        yy /= np.nanmax(yy)
        p0 = np.array([1.0, # amp
                       good_peaks[i], # mu
                       len(use) / 4, # sigma
                       np.nanmin(yy)]) # offset
        bounds = [(0.8, 1.5), (p0[1] - np.nanmean(peak_spacing[use]) / 2, p0[1] + np.nanmean(peak_spacing[use]) / 2), (0.25 * p0[2], 4*p0[2]), (0, p0[3] * 2)]
        opt_result = scipy.optimize.minimize(fit_peak1d, p0, args=(xx, yy), method='Nelder-Mead', bounds=bounds)
        pbest = opt_result.x
        lfc_centers_pix[i] = pbest[1]
        rms_norm[i] = opt_result.fun / pbest[0]

    # Flag bad fits
    good_rms = pcmath.weighted_median(rms_norm, percentile=0.75)
    good = np.where(rms_norm < 3*good_rms)[0]
    lfc_centers_pix = lfc_centers_pix[good]

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

    return lfc_centers_pix, lfc_centers_wave, peak_integers

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

def fit_peak1d(pars, x, data):
    model = pcmath.gauss(x, *pars[0:3]) + pars[3]
    rms = pcmath.rmsloss(data, model)
    return rms

def fit_lsf(pars, x, data):
    model = pcmath.gauss(x, pars[0], 0, pars[1])
    rms = pcmath.rmsloss(data, model)
    return rms

def fit_peak2d(pars, xarr, yarr, image):
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