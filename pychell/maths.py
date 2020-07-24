
#### A helper file containing math routines
# Default python modules
# Debugging
from pdb import set_trace as stop
import warnings
import sys

# Science/Math
import scipy.interpolate # spline interpolation
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np
import scipy.ndimage.filters
try:
    import torch
except:
    warnings.warn("Could not import pytorch!")
from astropy.coordinates import SkyCoord
import astropy.units as units

# LLVM
from numba import jit, njit, prange
import numba
import numba.types as nt
from llc import jit_filter_function

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

# Fast median filter 1d over a fixed box width in "pixels"
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

# Returns a gaussian
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


# Compatible with Parameters objects
def gauss_solver(pars, x, data):
    """Wrapper to optimize a Gaussian.

    Args:
        pars (Parameters): The parameters object.
        x ([type]): The independent variable.
        data ([type]): The data to model.

    Returns:
        float: The RMS between the data and model. This is the value to minimize.
        cons: Returns 1 since there are no additional constraints to this problem.
    """
    model = gauss(x, pars['amp'].value, pars['mu'].value, pars['sigma'].value)
    good = np.where(np.isfinite(data))[0]
    rms = np.sqrt(np.nansum((data - model)**2) / good.size)
    return rms, 1


def fix_nans(x, y):
    """A crude method of replacing bad pixels. Only pixels with nan or inf and bounded on both ends by at least one "good" data point are replaced.

    Args:
        x (np.ndarray): The independent variable.
        y (np.ndarray): The dependent variable containing bad pixels as either nan or inf.

    Returns:
        np.ndarray: The fixed dependent variable.
    """
    bad_all = np.where(~np.isfinite(y))[0]
    good = np.where(np.isfinite(y))[0]
    y_fixed = np.copy(y)
    f, l = good[0], good[-1]
    actual_bad = np.where((bad_all > f) & (bad_all < l))[0]
    if actual_bad.size > 0:
        actual_bad = bad_all[actual_bad]
    y_fixed[actual_bad] = scipy.interpolate.CubicSpline(x[good], y[good], extrapolate=False)(x[actual_bad])
    
    return y_fixed

# Robust stddev (almost the mad)
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

def convolve_flux(wave, flux, R, compress=64):
    
    good = np.where(np.isfinite(wave) & np.isfinite(flux))[0]
    ng = good.size
    
    # Interpolate onto a linearly spaced grid
    cspline = scipy.interpolate.CubicSpline(wave[good], flux[good], extrapolate=True)
    lingrid = np.linspace(wave[good[0]], wave[good[-1]], num=ng)
    fluxlin = cspline(lingrid)
    dl = lingrid[1] - lingrid[0]
    
    # The mean wavelength
    ml = np.nanmean(wave[good])
    
    # The number of points in the lsf grid
    nlsf = int(ng / compress)
    
    xlsf = np.arange(-(int(nlsf / 2) - 1), int(nlsf / 2) + 1, 1) * dl
    fluxlinp = np.pad(fluxlin, pad_width=(int(nlsf / 2 - 1), int(nlsf / 2)), mode='constant', constant_values=(fluxlin[0], fluxlin[-1]))

    sig = ml / (2 * np.sqrt(2 * np.log(2)) * R)
    lsf = np.exp(-0.5 * (xlsf / sig)**2)
    lsf /= np.sum(lsf)

    fluxlinc = np.convolve(fluxlinp, lsf, mode='valid')
    goodlinc = np.where(np.isfinite(fluxlinc))[0]
    fluxc = scipy.interpolate.CubicSpline(lingrid, fluxlinc, extrapolate=False)(wave)
    
    return fluxc


# Returns the hermite polynomial of degree deg over the variable x
def hermfun(x, deg):
    """Computes Hermite Gaussian Functions via the recursion relation.

    Args:
        x (np.ndarray): The independent variable grid.
        deg (int): The degree of the Hermite functions. deg=0 is just a Gaussian.

    Returns:
        np.ndarray: The individual Hermite-Gaussian functions as column vectors.
    """
    herm0 = np.pi**-0.25 * np.exp(-1.0 * x**2 / 2.0)
    herm1 = np.sqrt(2) * herm0 * x
    if deg == 0:
        herm = herm0
    elif deg == 1:
        herm = np.array([herm0, herm1]).T
    else:
        herm = np.zeros(shape=(x.size, deg+1), dtype=float)
        herm[:, 0] = herm0
        herm[:, 1] = herm1
        for k in range(2, deg+1):
            herm[:, k] = np.sqrt(2 / k) * (x * herm[:, k-1] - np.sqrt((k - 1) / 2) * herm[:, k-2])
    return herm

# Interpolation but fix nans
def interpolate_fix_nans(x, y, xnew):

    # Identify bad pixels and remove them before interpolation
    bad_init = np.where(~np.isfinite(y) | ~np.isfinite(np.roll(y, 1, axis=0)) | ~np.isfinite(np.roll(y, -1, axis=0)))[0]
    nbad_init = bad_init.size
    xx = np.delete(x, bad_init)
    yy = np.delete(y, bad_init)

    if xx.size == 0:
        return np.full(xnew.size, fill_value=np.nan)

    eqs = scipy.interpolate.interp1d(xx, yy, kind='linear', fill_value=np.nan, assume_sorted=True, bounds_error=False)
    ynew = eqs(xnew)
    bad = np.where((xnew > np.nanmax(x)) | (xnew < np.nanmin(x)))[0]
    nbad = bad.size

    for i in range(nbad_init):
        
        # If bad array element is first input array position
        if bad_init[i] == 0:
            bad_i = np.where(xnew <= x[bad_init[i]])[0]
            nbad_i = bad_i.size
            if nbad_i != 0:
                ynew[bad_i] = np.nan
            continue
        # If bad array element is last input array position
        if bad_init[i] == x.size-1:
            bad_i = np.where(xnew > x[bad_init[i]])[0]
            nbad_i = bad_i.size
            if nbad_i != 0:
                ynew[bad_i] = np.nan
            continue
        # If bad array element is anywhere else
        bad_i = np.where((xnew >= x[bad_init[i]-1]) & (xnew <= x[bad_init[i]+1]))[0]
        nbad_i = bad_i.size
        if nbad_i != 0:
            ynew[bad_i] = np.nan
    
    if nbad != 0:
        ynew[bad] = np.nan

    return ynew


# This calculates the median absolute deviation of array x
def mad(x):
    """Computes the true median absolute deviation.

    Args:
        x np.ndarray: The input array.

    Returns:
        float: The M.A.D. of the input.
    """
    return np.nanmedian(np.abs(x - np.nanmedian(x)))

# This calculates the weighted percentile of a data set.
def weighted_median(data, weights=None, med_val=0.5):
    """Computes the weighted percentile of a data set

    Args:
        data (np.ndarray): The input data.
        weights (np.ndarray, optional): How to weight the data. Defaults to uniform weights.
        med_val (float, optional): The desired percentile. Defaults to 0.5.

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
    med_val = med_val * np.nansum(weights)
    if np.any(weights > med_val):
        good = np.where(weights == np.nanmax(weights))[0][0]
        w_median = data[good]
    else:
        cs_weights = np.nancumsum(weights_s)
        idx = np.where(cs_weights <= med_val)[0][-1]
        if weights_s[idx] == med_val:
            w_median = np.nanmean(data_s[idx:idx+2])
        else:
            w_median = data_s[idx+1]
    return w_median

# This calculates the unbiased weighted standard deviation of array x with weights w
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

# This calculates the weighted mean of array x with weights w
@jit
def weighted_mean(x, w):
    """Computes the weighted mean of a dataset.

    Args:
        x (np.ndarray): The input array.
        w (np.ndarray): The weights.

    Returns:
        float: The weighted mean.
    """
    return np.nansum(x * w) / np.nansum(w)

# Rolling function f over a window given w of y given the independent variable x
def rolling_fun_true_window(f, x, y, w):
    """Computes a filter over the data using windows determined from a proper independent variable.

    Args:
        f (function): The desired filter, must take a numpy array as input.
        x (np.ndarray): The independent variable.
        y (np.ndarray): The dependent variable.
        w (float): Window size (in units of x)

    Returns:
        np.ndarray: The filtered array.
    """
    output = np.empty(x.size, dtype=np.float64)

    for i in range(output.size):
        locs = np.where((x > x[i] - w/2) & (x <= x[i] + w/2))
        if len(locs) == 0:
            output[i] = np.nan
        else:
            output[i] = f(y[locs])

    return output


# Rolling function f over a window given w of y given the independent variable x
def rolling_stddev_overcols(image, nbins):
    """Computes a filter over the data using windows determined from a proper independent variable.

    Args:
        f (function): The desired filter, must take a numpy array as input.
        x (np.ndarray): The independent variable.
        y (np.ndarray): The dependent variable.
        w (float): Window size (in units of x)

    Returns:
        np.ndarray: The filtered array.
    """
    
    ny, nx = image.shape
    
    xarr = np.arange(nx) # redundant and to remind me of potential modifications
    goody, goodx = np.where(np.isfinite(image))
    f, l = xarr[goodx[0]], xarr[goodx[-1]]
    bins = np.linspace(f, l, num=nbins + 1)
    output = np.empty(nbins, dtype=np.float64) + np.nan
    for i in range(nbins):
        locs = np.where((xarr >= bins[i]) & (xarr < bins[i+1]))[0]
        if len(locs) == 0:
            output[i] = np.nan
        else:
            output[i] = np.nanstd(image[:, locs])

    return output, bins



# Locates the closest value to a given value in an array
# Returns the value and index.
def find_closest(x, val):
    """Finds the index and corresponding value in x which is closest to some val.

    Args:
        x (np.ndarray): The array to search.
        val (float): The value to be closest to.

    Returns:
        int: The index of the closest member
        float: The value at that index.
    """
    diffs = np.abs(x - val)
    loc = np.nanargmin(diffs)
    return loc, x[loc]

# Cross correlation for trace profile
def cross_correlate1(y1, y2, lags):
    """Cross-correlation in "pixel" space.

    Args:
        y1 (np.ndarray): The array to cross-correlate.
        y2 (np.ndarray): The array to cross-correlate against.
        lags (np.ndarray): An array of lags (shifts), must be integers.

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
        #ylag = shiftint1d(y2, -1*lags[i], cval=np.nan)
        ylag = np.roll(y2, -1*lags[i])
        vec_cross = y1 * ylag
        weights = np.ones(n1, dtype=np.float64)
        bad = np.where(~np.isfinite(vec_cross))[0]
        if bad.size > 0:
            weights[bad] = 0
        corrfun[i] = np.nansum(vec_cross * weights) / np.nansum(weights)

    return corrfun - np.nanmin(corrfun)

def cross_correlate2(x1, y1, x2, y2, lags):
    """Cross-correlation in "pixel" space.

    Args:
        y1 (np.ndarray): The array to cross-correlate.
        y2 (np.ndarray): The array to cross-correlate against.
        lags (np.ndarray): An array of lags (shifts), must be integers.

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
        corrfun[i] = np.nansum(vec_cross * weights) / np.nansum(weights)

    return corrfun



def intersection(x, y, yval, precision=1000):
    
    dx = np.nanmedian(np.abs(np.diff(x)))
    dxhr = dx / precision
    xhr = np.arange(np.nanmin(x), np.nanmax(x) + 1, dxhr)
    good = np.where(np.isfinite(x) & np.isfinite(y))[0]
    cspline = scipy.interpolate.CubicSpline(x[good], y[good], extrapolate=False)
    yhr = cspline(xhr)
    
    index, _ = find_closest(yhr, yval)
    
    return xhr[index], yhr[index]

# Calculates the reduced chi squared
def reduced_chi_square(x, err):
    """Computes the reduced chi-square statistic

    Args:
        x (np.ndarray): The input array.
        err (np.ndarray): The associated errors.

    Returns:
        float: The reduced chi-square statistic.
    """
    # Define the weights as 1 over the square of the error bars. 
    weights = 1.0 / err**2

    # Calculate the reduced chi square defined around the weighted mean, 
    # assuming we are not fitting to any parameters.
    redchisq = (1.0 / (x.size-1)) * np.nansum((x - weighted_mean(x, weights))**2 / err**2)

    return redchisq

# Given 3 data points this returns the polynomial coefficients via matrix inversion, effectively
# In theory equivalent to np.polyval(x, y, deg=2)
@jit
def poly_coeffs(x, y):
    """Computes quadratic coefficients given three points.

    Args:
        x (np.ndarray): The x points.
        y (np.ndarray): The corresponding y points.

    Returns:
        np.ndarray: The quadratic coefficients [quad, lin, zero]
    """
    a0 = (-x[2] * y[1] * x[0]**2 + x[1] * y[2] * x[0]**2 + x[2]**2 * y[1] * x[0] - x[1]**2 * y[2] * x[0] - x[1] * x[2]**2 * y[0] + x[1]**2 * x[2] * y[0])/((x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2]))
    a1 = (y[1] * x[0]**2 - y[2] * x[0]**2 - x[1]**2 * y[0] + x[2]**2 * y[0] - x[2]**2 * y[1] + x[1]**2 * y[2]) / ((x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2]))
    a2 = (x[1] * y[0] - x[2] * y[0] - x[0] * y[1] + x[2] * y[1] + x[0] * y[2] - x[1] * y[2]) / ((x[1] - x[0]) * (x[1] - x[2]) * (x[2] - x[0]))
    p = np.array([a2, a1, a0])
    return p


def mask_to_binary(x, l):
    """Converts a mask array of indices to a binary array.

    Args:
        x (np.ndarray): The array of indices.
        l (int): The length of the boolean array

    Returns:
       np.ndarray : The boolean mask.
    """
    binary = np.zeros(l, dtype=bool)
    if len(x) == 0:
        return binary
    for i in range(l):
        if i in x:
            binary[i] = True
    return binary.astype(bool)

def estimate_continuum(x, y, width=7, n_knots=14, cont_val=0.9):
    """Estimates the continuum of a spectrum with cubic splines.

    Args:
        x (np.ndarray): The input spectrum.
        y (np.ndarray): The input spectrum
        width (int, optional): The distance between knots in units of x. Defaults to 7, assuming Angstroms.
        n_knots (int, optional): The number of knots to estimat ethe continuum with. Defaults to 14.
        cont_val (float, optional): The percentile of the continuum. Defaults to 0.9, assuming a max-normalized spectrum.

    Returns:
        np.ndarray: The estimated continuum.
    """
    nx = x.size
    continuum_coarse = np.ones(nx, dtype=np.float64)
    for ix in range(nx):
        use = np.where((x > x[ix]-width/2) & (x < x[ix]+width/2))[0]
        if np.all(~np.isfinite(y[use])):
            continuum_coarse[ix] = np.nan
        else:
            continuum_coarse[ix] = weighted_median(y[use], weights=None, med_val=cont_val)
    
    good = np.where(np.isfinite(y))[0]
    knot_points = x[np.linspace(good[0], good[-1], num=n_knots).astype(int)]
    interp_fun = scipy.interpolate.CubicSpline(knot_points, continuum_coarse[np.linspace(good[0], good[-1], num=14).astype(int)], extrapolate=False, bc_type='not-a-knot')
    continuum = interp_fun(x)
    return continuum


# Horizontal median of a 2d image
def horizontal_median(image, width):
    """Computes the median filter for each row in an image.

    Args:
        image (np.ndarray): The input image.
        width (int): The width of the filter in pixels.

    Returns:
        np.ndarray: The filtered image.
    """
    ny, nx = image.shape
    out_image = np.empty(shape=(ny, nx), dtype=np.float64)
    for i in range(ny):
        out_image[i, :] = median_filter1d(image[i, :], width)
    return out_image

# Vertical median of a 2d image
def vertical_median(image, width):
    """Computes the median filter for each column in an image.

    Args:
        image (np.ndarray): The input image.
        width (int): The width of the filter in pixels.

    Returns:
        np.ndarray: The filtered image.
    """
    ny, nx = image.shape
    out_image = np.empty(shape=(ny, nx), dtype=np.float64)
    for i in range(nx):
        out_image[:, i] = median_filter1d(image[:, i], width)
    return out_image

# Out of bounds of image
@njit
def outob(x, y, nx, ny):
    """Shorthand to determine if a point is within bounds of a 2d image.

    Args:
        x (int): The x point to test.
        y (int): The y point to test.
        nx (int): The number of array columns.
        ny (int): The number of array rows.

    Returns:
        bool: Whether or not the point is within bounds
    """
    return x + 1 > nx or x + 1 < 1 or y + 1 > ny or y + 1 < 1


# Returns a modified gaussian
def gauss_modified(x, amp, mu, sigma, d):
    """Constructs a modified Gaussian (variable exponent)

    Args:
        x (np.ndarray): The independent variable.
        amp (float): The amplitude (height).
        mu (float): The mean (center point).
        sigma (float): The standard deviation.
        d (float): The modified exponent (2 for a standard Gaussian)

    Returns:
        np.ndarray: The modified Gaussian
    """
    return amp * np.exp(-0.5 * (np.abs((x - mu) / sigma))**(2 * d))


# Compatible with Parameters objects
def gauss_modified_solver(pars, x, data):
    """Wrapper to optimize a modified Gaussian.

    Args:
        pars (Parameters): The parameters object.
        x ([type]): The independent variable.
        data ([type]): The data to model.

    Returns:
        float: The RMS between the data and model. This is the value to minimize.
        cons: Returns 1 since there are no additional constraints to this problem.
    """
    model = gauss_modified(x, pars['amp'].value, pars['mu'].value, pars['sigma'].value, pars['d'].value)
    good = np.where(np.isfinite(data))[0]
    rms = np.sqrt(np.nansum((data - model)**2) / good.size)
    return rms, 1

if 'torch' in sys.modules:
    class LinearInterp1d(torch.autograd.Function):
        
        def __call__(self, x, y, xnew, out=None):
            return self.forward(x, y, xnew, out)

        def forward(ctx, x, y, xnew, out=None):

            # making the vectors at least 2D
            is_flat = {}
            require_grad = {}
            v = {}
            device = []
            for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
                assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                            'at most 2-D.'
                if len(vec.shape) == 1:
                    v[name] = vec[None, :]
                else:
                    v[name] = vec
                is_flat[name] = v[name].shape[0] == 1
                require_grad[name] = vec.requires_grad
                device = list(set(device + [str(vec.device)]))
            assert len(device) == 1, 'All parameters must be on the same device.'
            device = device[0]

            # Checking for the dimensions
            assert (v['x'].shape[1] == v['y'].shape[1]
                    and (
                            v['x'].shape[0] == v['y'].shape[0]
                            or v['x'].shape[0] == 1
                            or v['y'].shape[0] == 1
                        )
                    ), ("x and y must have the same number of columns, and either "
                        "the same number of row or one of them having only one "
                        "row.")

            reshaped_xnew = False
            if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
                and (v['xnew'].shape[0] > 1)):
                # if there is only one row for both x and y, there is no need to
                # loop over the rows of xnew because they will all have to face the
                # same interpolation problem. We should just stack them together to
                # call interp1d and put them back in place afterwards.
                original_xnew_shape = v['xnew'].shape
                v['xnew'] = v['xnew'].contiguous().view(1, -1)
                reshaped_xnew = True

            # identify the dimensions of output and check if the one provided is ok
            D = max(v['x'].shape[0], v['xnew'].shape[0])
            shape_ynew = (D, v['xnew'].shape[-1])
            if out is not None:
                if out.numel() != shape_ynew[0]*shape_ynew[1]:
                    # The output provided is of incorrect shape.
                    # Going for a new one
                    out = None
                else:
                    ynew = out.reshape(shape_ynew)
            if out is None:
                ynew = torch.zeros(*shape_ynew, dtype=y.dtype, device=device)

            # moving everything to the desired device in case it was not there
            # already (not handling the case things do not fit entirely, user will
            # do it if required.)
            for name in v:
                v[name] = v[name].to(device)

            # calling searchsorted on the x values.
            #ind = ynew
            #searchsorted(v['x'].contiguous(), v['xnew'].contiguous(), ind)
            ind = np.searchsorted(v['x'].contiguous().numpy().flatten(), v['xnew'].contiguous().numpy().flatten())
            ind = torch.tensor(ind)
            # the `-1` is because searchsorted looks for the index where the values
            # must be inserted to preserve order. And we want the index of the
            # preceeding value.
            ind -= 1
            # we clamp the index, because the number of intervals is x.shape-1,
            # and the left neighbour should hence be at most number of intervals
            # -1, i.e. number of columns in x -2
            ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1).long()

            # helper function to select stuff according to the found indices.
            def sel(name):
                if is_flat[name]:
                    return v[name].contiguous().view(-1)[ind]
                return torch.gather(v[name], 1, ind)

            # activating gradient storing for everything now
            enable_grad = False
            saved_inputs = []
            for name in ['x', 'y', 'xnew']:
                if require_grad[name]:
                    enable_grad = True
                    saved_inputs += [v[name]]
                else:
                    saved_inputs += [None, ]
            # assuming x are sorted in the dimension 1, computing the slopes for
            # the segments
            is_flat['slopes'] = is_flat['x']
            # now we have found the indices of the neighbors, we start building the
            # output. Hence, we start also activating gradient tracking
            with torch.enable_grad() if enable_grad else contextlib.suppress():
                v['slopes'] = (
                        (v['y'][:, 1:]-v['y'][:, :-1])
                        /
                        (v['x'][:, 1:]-v['x'][:, :-1])
                    )

                # now build the linear interpolation
                ynew = sel('y') + sel('slopes')*(
                                        v['xnew'] - sel('x'))

                if reshaped_xnew:
                    ynew = ynew.view(original_xnew_shape)

            ctx.save_for_backward(ynew, *saved_inputs)
            return ynew

        @staticmethod
        def backward(ctx, grad_out):
            inputs = ctx.saved_tensors[1:]
            gradients = torch.autograd.grad(
                            ctx.saved_tensors[0],
                            [i for i in inputs if i is not None],
                            grad_out, retain_graph=True)
            result = [None, ] * 5
            pos = 0
            for index in range(len(inputs)):
                if inputs[index] is not None:
                    result[index] = gradients[pos]
                    pos += 1
            return (*result,)


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