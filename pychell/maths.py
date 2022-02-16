#### A helper file containing math routines
# Default python modules
# Debugging
import warnings
import sys

# Science/Math
import scipy.interpolate # spline interpolation
from scipy import constants as cs # cs.c = speed of light in m/s
import scipy.stats
import numpy as np
import scipy.ndimage.filters
import numpy.polynomial.chebyshev

from astropy.coordinates import SkyCoord
import astropy.modeling.functional_models
import astropy.units as units

# Graphics for debugging
import matplotlib.pyplot as plt

# LLVM
from numba import jit, njit, prange
import numba
import numba.types as nt
from llc import jit_filter_function

def compute_R2_stat(ydata, ymodel, w=None):
    """Computes the weighted R2 stat.

    Args:
        ydata (np.ndarray): The observations.
        ymodel (np.ndarray): The model.
        w (np.ndarray, optional): The weights. Defaults to uniform weights (so unweighted).

    Returns:
        float: The weighted R2 stat.
    """
    
    # Weights
    if w is None:
        w = np.ones_like(ydata)
    w = w / np.nansum(w)
    
    # Weighted mean of the data
    ybardata = weighted_mean(ydata, w)
    
    # SStot (total sum of squares)
    sstot = np.nansum(w * (ydata - ybardata)**2)
    
    # Residuals
    resid = ydata - ymodel
    
    # SSres
    ssres = np.nansum(w * resid**2)
    
    # R2 stat
    r2 = 1 - (ssres / sstot)
    
    return r2

@njit(nogil=True)
def outer_diff(x, y):
    """Computes the matrix D_ij = abs(xi-yj)

    Args:
        x (np.ndarray): The x variable.
        y (np.ndarray): The y variable.

    Returns:
        np.ndarray: The distance matrix D_ij.
    """
    n1 = len(x)
    n2 = len(y)
    out = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            out[i, j] = np.abs(x[i] - y[j])
    return out

@njit
def outer_fun(fun, x, y):
    """Generalized outer function (like an outer product). Creates the matrix D_ij = fun(xi, yj)

    Args:
        fun (function): A function decorated with numba.jit.
        x (np.ndarray): The x variable.
        y (np.ndarray): The y variable.

    Returns:
        np.ndarray: The matrix D_ij.
    """
    n1 = len(x)
    n2 = len(x)
    out = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            out[i, j] = fun(x[i], y[j])
    return out


def rmsloss(x, y, weights=None, flag_worst=0, remove_edges=0):
    """Convenient method to compute the weighted RMS between two vectors x and y.

    Args:
        x (np.ndarray): The x variable (1d).
        y (np.ndarray): The y variable (1d).
        weights (np.ndarray, optional): The weights. Defaults to uniform weights.
        flag_worst (int, optional): Flag the largest outliers (with weights applied). Defaults to 0.
        remove_edges (int, optional): Ignore the edges. Defaults to 0.

    Returns:
        float: The weighted RMS.
    """
    
    # Compute diffs2
    if weights is not None:
        good = np.where(np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0))[0]
        xx, yy, ww = x[good], y[good], weights[good]
        diffs2 = ww * (xx - yy)**2
    else:
        good = np.where(np.isfinite(y))[0]
        xx, yy = x[good], y[good]
        diffs2 = (xx - yy)**2
    
    # Ignore worst N pixels
    if flag_worst > 0:
        ss = np.argsort(diffs2)
        diffs2[ss[-1*flag_worst:]] = np.nan
        if weights is not None:
            ww[ss[-1*flag_worst:]] = 0
                
    # Remove edges
    if remove_edges > 0:
        diffs2[0:remove_edges] = 0
        diffs2[-remove_edges:] = 0
        if weights is not None:
            ww[0:remove_edges] = 0
            ww[-remove_edges:] = 0
        
    # Compute rms
    if weights is not None:
        rms = np.sqrt(np.nansum(diffs2) / np.nansum(ww))
    else:
        n_good = diffs2.size
        rms = np.sqrt(np.nansum(diffs2) / n_good)

    return rms

def measure_fwhm(x, y):
    """Measures the fwhm of a signal.

    Args:
        x (np.ndarray): The grid the signal is sampled on.
        y (np.ndarray): The signal.

    Returns:
        float: The full width half max of the signal in units of x.
    """
    
    max_loc = np.nanargmax(y)
    max_val = np.nanmax(y)
    
    left = np.where((x < x[max_loc]) & (y < 0.7 * max_val))[0]
    right = np.where((x > x[max_loc]) & (y < 0.7 * max_val))[0]
    
    left_x = intersection(x[left], y[left], 0.5 * max_val, precision = 1000)[0]
    right_x = intersection(x[right], y[right], 0.5 * max_val, precision = 1000)[0]
    fwhm = right_x - left_x
    
    return fwhm

def sigmatofwhm(sigma):
    """Converts sigma to fwhm assuming a normal distribution.

    Args:
        sigma (float): The standard deviation of a distribution.

    Returns:
        float: The full width half max of the distribution.
    """
    return sigma * np.sqrt(8 * np.log(2))

def fwhmtosigma(fwhm):
    """Converts fwhm to sigma assuming a normal distribution.

    Args:
        fwhm (float): The full width half max of the distribution.

    Returns:
        (float): The standard deviation of a distribution.
    """
    return fwhm / np.sqrt(8 * np.log(2))

def doppler_shift_wave(wave, vel):
    wave_shifted = doppler_shift_SR(wave, vel)
    return wave_shifted

def doppler_shift_flux(wave, flux, vel, wave_out=None):
    """Doppler shifts a signal according to the standard SR Doppler formula (radial only) or a differential version of the classical Doppler equation.

    Args:
        wave (np.ndarray): The initial wavelengths.
        vel (float): The velocity in m/s.
        flux (np.ndarray, optional): The spectrum corresponding to wave. Defaults to None.

    Returns:
        np.ndarray: The Doppler shifted wave vector if flux is None, or the Doppler shifted flux on the wave_out grid otherwise.
    """

    # The shifted wave
    wave_shifted = doppler_shift_wave(wave, vel)

    # Interpolate the flux
    if wave_out is None:
        wave_out = wave_shifted
        flux_out = flux
    else:
        flux_out = cspline_interp(wave_shifted, flux, wave_out)

    # Return
    return wave_out, flux_out
    
def lin_interp(x, y, xnew):
    """Alias for np.interp with np.nan as left and right values.

    Args:
        x (np.ndarray): The x grid of the current signal.
        y (np.ndarray): The y grid of the current signal.
        xnew (np.ndarray): The new grid to sample y on.

    Returns:
        np.ndarray: The signal y sampled on xnew.
    """
    return np.interp(xnew, x, y, left=np.nan, right=np.nan)

def cspline_interp(x, y, xnew):
    """Alias for scipy.interpolate.CubicSpline.

    Args:
        x (np.ndarray): The x grid of the current signal.
        y (np.ndarray): The y grid of the current signal.
        xnew (np.ndarray): The new grid to sample y on.

    Returns:
        np.ndarray: The signal y sampled on xnew.
    """
    good = np.where(np.isfinite(x) & np.isfinite(y))[0]
    return scipy.interpolate.CubicSpline(x[good], y[good], extrapolate=False)(xnew)

def cspline_fit(x, y, knots, weights=None):
    """Fits a signal with scipy.interpolate.LSQUnivariateSpline.

    Args:
        x (np.ndarray): The x grid.
        y (np.ndarray): The y grid.
        knots (np.ndarray): The knots.
        weights (np.ndarray, optional): The weights. Defaults to uniform weights.

    Returns:
        LSQUnivariateSpline: The nominal spline fit.
    """
    if weights is None:
        weights = np.ones_like(y)
    good = np.where(np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0))[0]
    xx, yy, ww = x[good], y[good], weights[good]
    _cspline_fit = scipy.interpolate.LSQUnivariateSpline(xx, yy, t=knots, w=ww, k=3, ext=1)
    return _cspline_fit

def doppler_shift_SR(wave, vel):
    """Doppler-shift according to the SR equation.

    Args:
        wave (np.ndarray or float): The input wavelengths.
        vel (float): The velocity in m/s.

    Returns:
        np.ndarray or float: The Doppler-shifted wavelength.
    """
    beta = vel / cs.c
    return wave * np.sqrt((1 + beta) / (1 - beta))

def doppler_shift_CM(wave, vel):
    """Doppler-shift according to the classical eq.

    Args:
        wave (np.ndarray or float): The input wavelengths.
        vel (float): The velocity in m/s.

    Returns:
        np.ndarray or float: The Doppler-shifted wavelength.
    """
    return wave * np.exp(vel / cs.c)

@jit_filter_function
def fmedian(x):
    """Fast median calculation for median filtering arrays, called by generic_filter.

    Args:
        x (np.ndarray): The window.

    Returns:
        float: The filtered value given this window.
    """
    if np.sum(np.isfinite(x)) == 0:
        return np.nan
    else:
        return np.nanmedian(x)

def median_filter1d(x, width, preserve_nans=True):
    """Computes a median 1d filter.

    Args:
        x (np.ndarray): The array to filter.
        width (int): The width of the filter.
        preserve_nans (bool, optional): Whether or not to preserve any nans or infs which may get overwritten. Defaults to True.

    Returns:
        np.ndarray: The filtered array.
    """
    
    bad = np.where(~np.isfinite(x))[0]
    good = np.where(np.isfinite(x))[0]
    
    if good.size == 0:
        return np.full(x.size, fill_value=np.nan)
    else:
        out = scipy.ndimage.filters.generic_filter(x, fmedian, size=width, cval=np.nan, mode='constant')    
        
    if preserve_nans:
        out[bad] = np.nan # If a nan is overwritten with a new value, rewrite with nan
        
    return out

#@jit
def generalized_median_filter1d(x, width, percentile=0.5):
    nx = len(x)
    y = np.full(nx, np.nan)
    for i in range(nx):
        ilow = int(np.max([0, i - np.ceil(width / 2)]))
        ihigh = int(np.min([i + np.floor(width / 2), nx - 1]))
        if np.where(np.isfinite(x[ilow:ihigh+1]))[0].size > 0:
            y[i] = weighted_median(x[ilow:ihigh+1], percentile=percentile)
    return y

@njit
def gauss(x, amp, mu, sigma):
    """Constructs a standard Gaussian

    Args:
        x (np.ndarray): The independent variable.
        amp (float): The amplitude (height).
        mu (float): The mean (center point).
        sigma (float): The standard deviation.

    Returns:
        np.ndarray: The Gaussian
    """
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def robust_stddev(x):
    """Computes the robust standard deviation of a data set.

    Args:
        x (np.ndarray): The input array

    Returns:
        float: The robust standard deviation
    """
    eps = 1E-20
    x0 = np.nanmedian(x)

    # First, the median absolute deviation MAD about the median:
    mad_ = mad(x)

    if mad_ < eps:
        mad_ = np.nanmean(np.abs(x - x0)) / 0.8

    if mad_ < eps:
        return 0

    # Now the biweighted value:
    u = (x - x0) / (6 * mad_)
    uu = u * u
    q = np.where(uu <= 1)[0]
    count = q.size
    if count < 3:
        return -1

    numerator = np.nansum((x[q] - x0)**2 * (1 - uu[q])**4)
    n = x.size
    den1 = np.nansum((1 - uu[q]) * (1 - 5 * uu[q]))
    sigma = n * numerator / (den1 * (den1 - 1))

    if sigma > 0:
        return np.sqrt(sigma)
    else:
        return 0

def median_filter2d(x, width, preserve_nans=True):
    """Computes a median 2d filter.

    Args:
        x (np.ndarray): The array to filter.
        width (int): The width of the filter.
        preserve_nans (bool, optional): Whether or not to preserve any nans or infs which may get overwritten. Defaults to True.

    Returns:
        np.ndarray: The filtered array.
    """
    bad = np.where(~np.isfinite(x))
    good = np.where(np.isfinite(x))
    
    if good[0].size == 0:
        return np.full(x.shape, fill_value=np.nan)
    else:
        out = scipy.ndimage.filters.generic_filter(x, fmedian, size=width, cval=np.nan, mode='constant')
    
    if preserve_nans:
        out[bad] = np.nan # If a nan is overwritten with a new value, rewrite with nan
        
    return out

@njit
def convolve1d(x, k, padleft, padright):
    """Numerical convolution for a 1d signal.

    Args:
        x (np.ndarray): The input signal, must be uniform spacing.
        k (np.ndarray): The kernel, must have the same grid spacing as x.
        padleft (int): The left pad.
        padright (int): The right pad.

    Returns:
        np.ndarray: The convolved signal.
    """

    # Length of each
    nx = len(x)
    nk = len(k)

    # The left and right padding
    n_pad = numba.int64(nk / 2)

    # Output vector
    out = np.zeros(x.shape)

    # Flip the kernel (just a view, no new vector is allocated)
    kf = k[::-1]

    # Left values
    for i in range(n_pad):
        s = 0.0
        for j in range(nk):
            ii = i - n_pad + j
            if ii < 0:
                s += padleft * kf[j]
            else:
                s += x[ii] * kf[j]
        out[i] = s

    # Middle values
    for i in range(n_pad, nx-n_pad):
        s = 0.0
        for j in range(nk):
            s += x[i - n_pad + j] * kf[j]
        out[i] = s

    # Right values
    for i in range(nx-n_pad, nx):
        s = 0.0
        for j in range(nk):
            ii = i - n_pad + j
            if ii > nx - 1:
                s += padleft * kf[j]
            else:
                s += x[ii] * kf[j]
        out[i] = s

    # Return out
    return out

def convolve_flux(wave, flux, R=None, width=None, interp=None, lsf=None, croplsf=False):
    """Convolves flux.

    Args:
        wave (np.ndarray): The wavelength grid.
        flux (np.ndarray): The corresponding flux to convolve.
        R (float, optional): The resolution to convolve down to. Defaults to None.
        width (float, optional): The LSF width in units of wave. Defaults to None.
        compress (int, optional): The number of LSF points is only int(nwave / compress). Defaults to 64.
        uniform (bool, optional): Whether or not the wave grid is already uniform, which is necessary for standard convolution. Defaults to False.
        lsf (np.ndarray, optional): The LSF to convolve with.
    Returns:
        np.ndarray: The convolved flux
    """
    
    # Get good initial points
    if wave is not None:
        good = np.where(np.isfinite(wave) & np.isfinite(flux))[0]
        wavegood, fluxgood = wave[good], flux[good]
    else:
        good = np.where(np.isfinite(flux))[0]
        wavegood, fluxgood = None, flux[good]
        
    # Whether or not to interpolate onto a uniform grid
    if interp or interp is None and wavegood is not None:
        interp = np.unique(np.diff(wavegood)).size > 1 or good.size < wave.size
    else:
        interp = False
    
    # Interpolate onto a uniform grid if set or values were masked
    if interp:
        wavelin = np.linspace(wavegood[0], wavegood[-1], num=good.size)
        fluxlin = scipy.interpolate.CubicSpline(wavegood, fluxgood, extrapolate=False)(wavelin)
    else:
        wavelin, fluxlin = wavegood, fluxgood
    
    # Derive LSF from width or R
    if lsf is None:
        
        # The grid spacing
        dl = wavelin[1] - wavelin[0]
    
        # The mean wavelength
        ml = np.nanmean(wavelin)
        
        # Sigma
        sig = width_from_R(R, ml) if R is not None else width
    
        # Initial LSF grid that's way too big
        nlsf = int(0.1 * wavelin.size)
        xlsf = np.arange(np.floor(-nlsf / 2), np.floor(nlsf / 2) + 1) * dl

        # Construct and max-normalize LSF
        lsf = np.exp(-0.5 * (xlsf / sig)**2)
        lsf /= np.nanmax(lsf)
        
        # Only consider > 1E-10
        goodlsf = np.where(lsf > 1E-10)[0]
        nlsf = goodlsf.size
        if nlsf % 2 == 0:
            nlsf += 1
        xlsf = np.arange(-np.floor(nlsf / 2), np.floor(nlsf / 2) + 1) * dl

        # Construct and sum-normalize LSF
        lsf = np.exp(-0.5 * (xlsf / sig)**2)
        lsf /= np.nansum(lsf)
    
    else:
        
        # Get the approx index of the max of the LSF
        if croplsf:
            lsf = lsf / np.nanmax(lsf)
            max_ind = np.nanargmax(lsf)
            goodlsf = np.where(lsf > 1E-10)[0]
            nlsf = goodlsf.size
            if nlsf % 2 == 0:
                nlsf += 1
            f, l = goodlsf[0], goodlsf[-1]
            k = np.max([np.abs(f - max_loc), np.abs(l - max_loc)])
            nlsf = 2 * k + 1
            lsf = lsf[(max_loc - k):(max_loc + k + 1)]
            
        else:
            
            nlsf = lsf.size
            
    # Ensure the lsf size is odd
    assert lsf.size % 2 == 1
        
    # Pad
    fluxlinp = np.pad(fluxlin, pad_width=(int(nlsf / 2), int(nlsf / 2)), mode='constant', constant_values=(fluxlin[0], fluxlin[-1]))

    # Convolve
    fluxlinc = np.convolve(fluxlinp, lsf, mode='valid')
    #fluxlinc = convolve1d(fluxlin, lsf, fluxlin[0], fluxlin[-1])
    
    # Interpolate back to the default grid
    if interp:
        goodlinc = np.where(np.isfinite(fluxlinc))[0]
        fluxc = scipy.interpolate.CubicSpline(wavelin[goodlinc], fluxlinc[goodlinc], extrapolate=False)(wave)
    elif flux.size > fluxlinc.size:
        fluxc = np.full(flux.size, fill_value=np.nan)
        fluxc[good] = fluxlinc
    else:
        fluxc = fluxlinc
    
    return fluxc


def hermfun(x, deg):
    """Computes Hermite Gaussian Functions via the recursion relation.

    Args:
        x (np.ndarray): The independent variable grid.
        deg (int): The degree of the Hermite functions. deg=0 is just a Gaussian.

    Returns:
        np.ndarray: The individual Hermite-Gaussian functions as column vectors.
    """
    herm0 = np.pi**-0.25 * np.exp(-0.5 * x**2)
    herm1 = np.sqrt(2) * herm0 * x
    if deg == 0:
        herm = herm0
        return herm
    elif deg == 1:
        return np.array([herm0, herm1], dtype=float).T
    else:
        herm = np.zeros(shape=(x.size, deg+1), dtype=float)
        herm[:, 0] = herm0
        herm[:, 1] = herm1
        for k in range(2, deg+1):
            herm[:, k] = np.sqrt(2 / k) * (x * herm[:, k-1] - np.sqrt((k - 1) / 2) * herm[:, k-2])
        return herm

@njit
def mad(x):
    """Computes the true median absolute deviation.

    Args:
        x np.ndarray: The input array.

    Returns:
        float: The M.A.D. of the input.
    """
    return np.nanmedian(np.abs(x - np.nanmedian(x)))

#@jit
def weighted_median(data, weights=None, percentile=0.5):
    """Computes the weighted percentile of a data set

    Args:
        data (np.ndarray): The input data.
        weights (np.ndarray, optional): How to weight the data. Defaults to uniform weights.
        percentile (float, optional): The desired percentile. Defaults to 0.5.

    Returns:
        float: The weighted percentile of the data.
    """
    if weights is None:
        weights = np.ones(shape=data.shape, dtype=np.float64)
    bad = np.where(~np.isfinite(data))
    if bad[0].size == data.size:
        return np.nan
    if bad[0].size > 0:
        weights[bad] = 0
    data = data.flatten()
    weights = weights.flatten()
    inds = np.argsort(data)
    data_s = data[inds]
    weights_s = weights[inds]
    percentile = percentile * np.nansum(weights)
    if np.any(weights > percentile):
        good = np.where(weights_s == np.nanmax(weights_s))[0][0]
        w_median = data_s[good]
    else:
        cs_weights = np.nancumsum(weights_s)
        idx = np.where(cs_weights <= percentile)[0][-1]
        if weights_s[idx] == percentile:
            w_median = np.nanmean(data_s[idx:idx+2])
        else:
            w_median = data_s[idx+1]

    return w_median

#@njit
def weighted_stddev(x, w):
    """Computes the weighted standard deviation of a dataset with bias correction.

    Args:
        x (np.ndarray): The input array.
        w (np.ndarray): The weights.

    Returns:
        float: The weighted standard deviation.
    """
    weights = w / np.nansum(w)
    wm = weighted_mean(x, w)
    dev = x - wm
    bias_estimator = 1.0 - np.nansum(weights ** 2) / np.nansum(weights) ** 2
    var = np.nansum(dev ** 2 * weights) / bias_estimator
    return np.sqrt(var)

def weighted_mean(x, w, axis=None):
    """Computes the weighted mean of a dataset.

    Args:
        x (np.ndarray): The input array.
        w (np.ndarray): The weights.

    Returns:
        float: The weighted mean.
    """
    return np.nansum(x * w, axis=axis) / np.nansum(w, axis=axis)

def weighted_combine(y, w, yerr=None, err_type="empirical"):
    """Performs a weighted coadd.

    Args:
        y (np.ndarray): The data to coadd.
        w (np.ndarray): The weights.
        yerr (np.ndarray): The correspinding error bars. Defaults to None.
        err_type (str): The method to determine error bars. If "empirical", error bars are computed by considering the weighted stddev of the appropriate data points. If "poisson", error bars are computed by taking the mean of the appropriate data errors divided by the sqrt(N), where N is the number of data points. If there are only two data points, the poisson method is used regardless.
    """
    
    # Determine how many good data points.
    good = np.where((w > 0) & np.isfinite(w))[0]
    n_good = len(good)

    # If none are good, return nan
    if n_good == 0:
        yc, yc_unc = np.nan, np.nan
    elif n_good == 1:
        # If only one is good, the mean is the only good value, and the error is the only good error.
        # If no error is provided, nan is returned.
        yc = y[good[0]]
        if yerr is not None:
            yc_unc = yerr[good[0]]
        else:
            yc_unc = np.nan
    elif n_good == 2:
        # If only two are good, the mean is the weighted mean.
        # Error is computed from existing errors if provided, or empirically otherwise.
        yc = weighted_mean(y[good].flatten(), w[good].flatten())
        if yerr is not None:
            yc_unc = np.nanmean(yerr[good].flatten()) /  np.sqrt(2)
        else:
            yc_unc = weighted_stddev(y[good].flatten(), w[good].flatten()) / np.sqrt(2)
    else:
        # With >= 3 useful points, the mean is the weighted mean.
        # Error is computed from err_type
        yc = weighted_mean(y[good].flatten(), w[good].flatten())
        yc_unc = weighted_stddev(y[good].flatten(), w[good].flatten()) / np.sqrt(n_good)
        
    return yc, yc_unc

def cross_correlate(x1, y1, x2, y2, lags, kind="rms"):
    """Cross-correlation in "pixel" space.

    Args:
        y1 (np.ndarray): The array to cross-correlate.
        y2 (np.ndarray): The array to cross-correlate against.
        lags (np.ndarray): An array of lags (shifts), must be integers.
        kind (str): Which kind of XC to perform. "rms" computes the rms loss at each lag. Otherwise a standard XC is performed. Note this implies the ccf is flipped for "rms"

    Returns:
        np.ndarray: The cross-correlation function
    """
    # Shifts y2 and compares it to y1
    n1 = y1.size
    n2 = y2.size
  
    nlags = lags.size
    
    corrfun = np.zeros(nlags, dtype=float)
    corrfun[:] = np.nan
    for i in range(nlags):
        y2_shifted = np.interp(x1, x2 + lags[i], y2, left=np.nan, right=np.nan)
        good = np.where(np.isfinite(y2_shifted) & np.isfinite(y1))[0]
        if good.size < 3:
            continue
        vec_cross = y1 * y2_shifted
        weights = np.ones(n1, dtype=np.float64)
        bad = np.where(~np.isfinite(vec_cross))[0]
        if bad.size > 0:
            weights[bad] = 0
        if kind.lower() == "rms":
            corrfun[i] = rmsloss(y1, y2_shifted, weights=weights)
        else:
            corrfun[i] = np.nansum(vec_cross * weights) / np.nansum(weights)

    return corrfun


def cross_correlate_doppler(x1, y1, x2, y2, vels, kind="rms"):
    """Cross-correlation in "pixel" space.

    Args:
        y1 (np.ndarray): The array to cross-correlate.
        y2 (np.ndarray): The array to cross-correlate against.
        lags (np.ndarray): An array of lags (shifts), must be integers.
        kind (str): Which kind of XC to perform. "rms" computes the rms loss at each lag. Otherwise a standard XC is performed. Note this implies the ccf is flipped for "rms"

    Returns:
        np.ndarray: The cross-correlation function
    """
    # Shifts y2 and compares it to y1
    n1 = y1.size
    n2 = y2.size
  
    nvels = vels.size
    
    corrfun = np.zeros(nvels, dtype=float)
    corrfun[:] = np.nan
    for i in range(nvels):
        y2_shifted = np.interp(x1, doppler_shift_wave(x2, vels[i]), y2, left=np.nan, right=np.nan)
        good = np.where(np.isfinite(y2_shifted) & np.isfinite(y1))[0]
        if good.size < 3:
            continue
        vec_cross = y1 * y2_shifted
        weights = np.ones(n1, dtype=np.float64)
        bad = np.where(~np.isfinite(vec_cross))[0]
        if bad.size > 0:
            weights[bad] = 0
        if kind.lower() == "rms":
            corrfun[i] = rmsloss(y1, y2_shifted, weights=weights)
        else:
            corrfun[i] = np.nansum(vec_cross * weights) / np.nansum(weights)

    return corrfun

def intersection(x, y, yval, precision=1):
    """Computes the intersection (x value) of signal y with yval.

    Args:
        x (np.ndarray): The x vector.
        y (np.ndarray): The y vector.
        yval (float): The value to intersect.
        precision (int, optional): The precision of the intersection, effectively oversampling (not accuracy). Defaults to 1.

    Returns:
        float: The x value that corresponds to the intersection with yval.
    """
    
    if precision > 1:
        dx = np.nanmedian(np.abs(np.diff(x)))
        dxhr = dx / precision
        xhr = np.arange(np.nanmin(x), np.nanmax(x) + 1, dxhr)
        good = np.where(np.isfinite(x) & np.isfinite(y))[0]
        cspline = scipy.interpolate.CubicSpline(x[good], y[good], extrapolate=False)
        yhr = cspline(xhr)
    
        index, _ = find_closest(yhr, yval)
        
        return xhr[index], yhr[index]
        
    else:
        
        index, _ = find_closest(y, yval)
        
        return x[index], y[index]

def legpoly_coeffs(x, y, deg=None):
    """Computes the Legendre polynomial coefficients given a set of x and y points.

    Args:
        x (np.ndarray): The x vector.
        y (np.ndarray): The y vector.
        deg (int, optional): [description]. Defaults to len(x) - 1.

    Returns:
        np.ndarray: The nominal Legendre polynomial coefficients.
    """
    if deg is None:
        deg = len(x) - 1
    V = np.polynomial.legendre.legvander(x, deg)
    coeffs = np.linalg.solve(V, y)
    return coeffs

def poly_coeffs(x, y):
    """Computes the polynomial coefficients (of order len(x)) given a set of x and y points.

    Args:
        x (np.ndarray): The x vector.
        y (np.ndarray): The y vector.

    Returns:
        np.ndarray: The nominal polynomial coefficients.
    """
    V = np.vander(x)
    coeffs = np.linalg.solve(V, y)
    return coeffs

def gauss_modified(x, amp, mu, sigma, p):
    """Constructs a modified Gaussian (variable exponent p)

    Args:
        x (np.ndarray): The independent variable.
        amp (float): The amplitude (height).
        mu (float): The mean (center point).
        sigma (float): The standard deviation.
        d (float): The modified exponent (2 for a standard Gaussian)

    Returns:
        np.ndarray: The modified Gaussian
    """
    return amp * np.exp(-0.5 * (np.abs((x - mu) / sigma))**p)

def shiftint1d(x, n, cval=np.nan):
    result = np.empty(x.size)
    if n > 0:
        result[:n] = cval
        result[n:] = x[:-n]
    elif n < 0:
        result[n:] = cval
        result[:n] = x[-n:]
    else:
        result[:] = x
    return result

def voigt(x, amp, mu, sigma, fwhm_L):
    """Alias for astropy.modeling.functional_models.Voigt1D.

    Args:
        x (np.ndarray): The x vector.
        amp_L (float): The amplitude of the Lorentzian.
        mu_L (float): The mean of the Lorentzian.
        fwhm_L (float): The full width half max of the Lorentzian.

    Returns:
        np.ndarray: The Voigt profile.
    """
    return astropy.modeling.functional_models.Voigt1D(x_0=mu, amplitude_L=amp, fwhm_G=sigmatofwhm(sigma), fwhm_L=fwhm_L)(x)


# lorentz convolved with an arbitrary kernel
def generalized_voigt(x, amp, mu_L, fwhm_L, kernel):
    """Generalized Voigt profile a Lorentzian convolved with an arbitrary kernel.

    Args:
        x (np.ndarray): The x vector.
        amp_L (float): The amplitude of the Lorentzian.
        mu_L (float): The mean of the Lorentzian.
        fwhm_L (float): The full width half max of the Lorentzian.
        kernel (np.ndarray): The kernel to convolve with.

    Returns:
        np.ndarray: The generalized Voigt signal.
    """
    
    # Normalize kernel
    kernel = np.nansum(kernel)

    # Build lorentz
    _lorentz = lorentz(x, amp, mu, fwhm_L)

    # Convolve with kernel
    out = convolve_flux(x, _lorentz, lsf=kernel)

    return out

def normal(x):
    return gauss(x, 1 / np.sqrt(2 * np.pi), 0, 1)


def skew_normal(x, loc, scale, alpha):
    xx = (x - loc) / scale
    return (2 / scale) * normal(xx) * scipy.stats.norm.cdf(alpha * xx)

@njit
def lorentz(x, amp, mu, fwhm):
    """Computes a Lorentzian function.

    Args:
        x (np.ndarray or float): The x variable.
        amp ([type]): The amplitude.
        mu ([type]): The mean.
        fwhm ([type]): The full width half max.

    Returns:
        np.ndarray: The Lorentzian.
    """
    xx = (x - mu) / (fwhm / 2)
    return amp / (1 + xx**2)

def poly_filter(y, width, poly_order=3):
    """Filters a signal with polynomials.

    Args:
        y (np.ndarray): The signal to filter.
        width (int): The rolling width in pixels, must be odd.
        poly_order (int, optional): The polynomial order. Defaults to 3.

    Returns:
        np.ndarray: The filtered array.
    """
    width = int(width)
    assert width > poly_order
    nx = len(y)
    x = np.arange(nx).astype(int)
    window_arr = np.arange(width)
    y_out = np.full(nx, np.nan)
    for i in range(nx):
        ilow = int(np.max([0, np.ceil(i - width / 2)]))
        ihigh = int(np.min([np.floor(i + width / 2), nx - 1]))
        good = np.where(np.isfinite(y[ilow:ihigh + 1]))[0]
        if good.size < poly_order + 1:
            continue
        xx, yy = x[ilow:ihigh + 1][good], y[ilow:ihigh + 1][good]
        pfit = np.polyfit(xx, yy, poly_order)
        y_out[i] = np.polyval(pfit, x[i])
    good = np.where(np.isfinite(y))[0]
    ilow = np.min(good)
    ihigh = np.max(good)
    y_out[0:ilow] = np.nan
    y_out[ihigh + 1:] = np.nan
    return y_out

def poly_filter2(x, y, width, poly_order=3):
    """Filters a signal with polynomials.

    Args:
        x (np.ndarray): The grid for the signal to filter.
        y (np.ndarray): The signal to filter.
        width (int): The rolling width in pixels, must be odd.
        poly_order (int, optional): The polynomial order. Defaults to 3.

    Returns:
        np.ndarray: The filtered array.
    """
    nx = len(x)
    y_out = np.full(nx, np.nan)
    for i in range(nx):
        good = np.where((x > x[i] - width/2) & (x < x[i] + width/2))[0]
        if good.size < poly_order + 1:
            continue
        xx, yy = x[good], y[good]
        good = np.where(np.isfinite(xx) & np.isfinite(yy))[0]
        pfit = np.polyfit(xx[good], yy[good], poly_order)
        y_out[i] = np.polyval(pfit, x[i])
    return y_out
        
#@jit
def normalize_image(image, height, order_spacing, percentile=0.99, downsample=4):
    """Normalizes the traces of an echellogram which are roughly aligned with detector rows.

    Args:
        image (np.ndarray): The echellogram.
        window (int, optional): The size of the rolling window. Defaults to 5.
        percentile (float, optional): The percentile of the continuum in the rolling window. Defaults to 0.99.
        downsample (int, optional): How many columns to group together (higher is faster but less precise). Defaults to 8.

    Returns:
        np.ndarray: The normalized image.
    """
    out = np.full_like(image, np.nan)
    ny, nx = out.shape
    xx = np.arange(nx)
    for i in range(0, nx, downsample):
        good = np.where(np.isfinite(image[:, i]))[0]
        if good.size == 0:
            continue
        x_low = np.max([0, i - downsample / 2])
        x_high = np.min([i + downsample / 2, nx - 1])
        inds = np.arange(x_low, x_high + 1).astype(int)
        continuum_col = generalized_median_filter1d(image[:, i], width=2.5 * order_spacing + height, percentile=percentile)
        for j in inds:
            out[:, j] = image[:, j] / continuum_col
    bad = np.where(out < 0)
    if bad[0].size > 0:
        out[bad] = np.nan
    return out



def cspline_fit_fancy(x, y, window=None, n_knots=50, percentile=0.99):
    """Robust at fitting certain signals present in a dataset.

    Args:
        x (np.ndarray): The x vector.
        y (np.ndarray): The y vector.
        window (float, optional): The window size. Defaults to (x_max - x_min) / (4 * n_knots).
        n_knots (int, optional): The number of spline knots to fit the median signal by. Defaults to 50.
        percentile (float, optional): The percentile of the signal in the rolling window. Defaults to 0.99.

    Returns:
        np.ndarray: The desired signal.
    """
    nx = len(x)
    y_out_init = np.full_like(y, np.nan)
    good = np.where(np.isfinite(x) & np.isfinite(y))[0]
    x_min, x_max = np.min(x[good]), np.max(x[good])
    knots = np.linspace(x_min + 1E-5, x_max - 1E-5, num=n_knots)
    if window is None:
        window = (x_max - x_min) / (4 * n_knots)
    for i in range(nx):
        x_mid = x[i]
        use = np.where((x > x_mid - window / 2) & (x < x_mid + window / 2))[0]
        if use.size == 0:
            continue
        y_out_init[i] = weighted_median(y[use], percentile=percentile)
    cspline = cspline_fit(x, y_out_init, knots=knots)
    y_out = cspline(x)
    return y_out

# lf = l0 * e^(dv/c)
# dl = l0 * (e^(dv/c) - 1)
# dl / l0 = (e^(dv/c) - 1)
# dl / l0 + 1 = e^(dv/c)
# ln(dl / l0 + 1) = dv/c
# c * ln(dl / l0 - 1) = dv
@jit
def dl_to_dv(dl, l):
    return cs.c * np.log(dl / l + 1)

# dl = l0 * (e^(dv/c) - 1)
@jit
def dv_to_dl(dv, l):
    return l * (np.exp(dv / cs.c) - 1)

def flatten_jagged_list(x):
    x_out = np.array([], dtype=float)
    inds = np.array([], dtype=int)
    for i in range(len(x)):
        j_start = len(x_out)
        x_out = np.concatenate((x_out, x[i]))
        inds = np.concatenate((inds, np.arange(len(x[i]))))
    return x_out, inds

def chebyval2d(pcoeffs, echelle_order, norm_pixel, norm_order, poly_order_inter_order, poly_order_intra_order):
    if len(pcoeffs.shape) == 1:
        pcoeffs = pcoeffs.reshape((poly_order_inter_order+1, poly_order_intra_order+1))
    #return 1 / echelle_order * numpy.polynomial.chebyshev.chebval2d(norm_pixel, norm_order, pcoeffs)
    return 1 / norm_order * numpy.polynomial.chebyshev.chebval2d(norm_pixel, norm_order, pcoeffs)

def chebyval2d_model_builder(pcoeffs, echelle_orders_flat, pixels_flat, nx, max_order, poly_order_inter_order, poly_order_intra_order):
    model = chebyval2d(pcoeffs, echelle_orders_flat, pixels_flat / nx, echelle_orders_flat / max_order, poly_order_inter_order, poly_order_intra_order)
    return model

def solve_poly2d_echelle(pcoeffs, pixels_flat, y_flat, weights_flat, echelle_orders_flat, nx, max_order, poly_order_inter_order, poly_order_intra_order):
    model_flat = chebyval2d_model_builder(pcoeffs, echelle_orders_flat, pixels_flat, nx, max_order, poly_order_inter_order, poly_order_intra_order)
    good = np.where((weights_flat > 0) & np.isfinite(weights_flat))[0]
    w = weights_flat[good]
    m = model_flat[good]
    s = y_flat[good]
    w = w / np.nansum(w)
    wres = w * (m - s)
    return wres
