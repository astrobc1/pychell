#### A helper file containing math routines
# Default python modules
# Debugging
from pdb import set_trace as stop

# Science/Math
import scipy.interpolate # spline interpolation
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as units

# LLVM
from numba import jit, njit, prange
import numba
import numba.types as nt
from llc import jit_filter_function


@jit_filter_function
def fmedian(x):
    if np.sum(np.isfinite(x)) == 0:
        return np.nan
    else:
        return np.nanmedian(x)

# Fast median filter 1d over a fixed box width in "pixels"
def median_filter1d(x, width, preserve_nans=True):
    
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
#@njit(nt.float64[:](nt.float64[:], nt.float64, nt.float64, nt.float64))
def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)


# Compatible with Parameters objects
def gauss_solver(pars, x, data):
    model = gauss(x, pars['amp'].value, pars['mu'].value, pars['sigma'].value)
    good = np.where(np.isfinite(data))[0]
    rms = np.sqrt(np.nansum((data - model)**2) / good.size)
    return rms, 1


# Fixes nans with cubis spline interpolation
def fix_nans(x, y):
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
    
    bad = np.where(~np.isfinite(x))
    good = np.where(np.isfinite(x))
    
    if good[0].size == 0:
        return np.full(x.shape, fill_value=np.nan)
    else:
        out = scipy.ndimage.filters.generic_filter(x, fmedian, size=width, cval=np.nan, mode='constant')
    
    if preserve_nans:
        out[bad] = np.nan # If a nan is overwritten with a new value, rewrite with nan
        
    return out

# Computes the RV content per pixel.
@jit
def rv_content_per_pixel(wave, flux, snr=100, gain=1.0, use_blaze=False):

    counts = snr**2
    pe = gain * counts
    A_center = pe
    good = np.where(np.isfinite(wave) & np.isfinite(flux))[0]
    if use_blaze:
        A = A_center * np.abs(np.sinc(0.01 * (wave - np.nanmean(wave))))**1.6 * flux # modulate by a true blaze
    else:
        A = A_center * flux
    rvc_per_pix = np.empty(wave.size, dtype=np.float64)
    A_spline = scipy.interpolate.CubicSpline(wave, A, extrapolate=True, bc_type='not-a-knot')
    for j in range(wave.size-1):
        if j in good:
            slope = A_spline(wave[j], 1)
            rvc_per_pix[j] = cs.c * np.sqrt(A[j]) / (wave[j] * np.abs(slope))
        else:
            rvc_per_pix[j] = np.nan
    return rvc_per_pix

# Returns the hermite polynomial of degree deg over the variable x
def hermfun(x, deg):
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
    return np.nanmedian(np.abs(x - np.nanmedian(x)))

# This calculates the weighted percentile of a data set.
def weighted_median(data, weights=None, med_val=0.5):

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
        w_median = (data[weights == np.nanmax(weights)])
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
    weights = w / np.nansum(w)
    wm = weighted_mean(x, w)
    dev = x - wm
    bias_estimator = 1.0 - np.nansum(weights ** 2) / np.nansum(weights) ** 2
    var = np.nansum(dev ** 2 * weights) / bias_estimator
    return np.sqrt(var)

# This calculates the weighted mean of array x with weights w
@jit
def weighted_mean(x, w):
    return np.nansum(x * w) / np.nansum(w)

# Rolling function f over a window given w of y given the independent variable x
def rolling_fun_true_window(f, x, y, w):

    output = np.empty(x.size, dtype=np.float64)

    for i in range(output.size):
        locs = np.where((x > x[i] - w/2) & (x <= x[i] + w/2))
        if len(locs) == 0:
            output[i] = np.nan
        else:
            output[i] = f(y[locs])

    return output

# Locates the closest value to a given value in an array
# Returns the value and index.
def find_closest(x, val):
    diffs = np.abs(x - val)
    loc = np.nanargmin(diffs)
    return loc, x[loc]

# Cross correlation for trace profile
#(nt.float64[:](nt.float64[:], nt.float64[:], nt.float64[:]))
def cross_correlate(y1, y2, lags):

    # Shifts y2 and compares it to y1
    n1 = y1.size
    n2 = y2.size
  
    nlags = lags.size
    
    corrfun = np.zeros(nlags, dtype=float)
    corrfun[:] = np.nan
    for i in range(nlags):
        #ylag = np.interp(np.arange(y2.size), np.arange(y2.size) - lags[i], y2, left=np.nan, right=np.nan)
        ylag = np.roll(y2, -1*lags[i], axis=0)
        vec_cross = y1 * ylag
        weights = np.ones(n1, dtype=np.float64)
        bad = np.where(~np.isfinite(vec_cross))[0]
        if bad.size > 0:
            weights[bad] = 0
        corrfun[i] = np.nansum(vec_cross * weights) / np.nansum(weights)

    return corrfun - np.nanmin(corrfun)

# Calculates the reduced chi squared
def reduced_chi_square(x, err):

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
    a0 = (-x[2] * y[1] * x[0]**2 + x[1] * y[2] * x[0]**2 + x[2]**2 * y[1] * x[0] - x[1]**2 * y[2] * x[0] - x[1] * x[2]**2 * y[0] + x[1]**2 * x[2] * y[0])/((x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2]))
    a1 = (y[1] * x[0]**2 - y[2] * x[0]**2 - x[1]**2 * y[0] + x[2]**2 * y[0] - x[2]**2 * y[1] + x[1]**2 * y[2]) / ((x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2]))
    a2 = (x[1] * y[0] - x[2] * y[0] - x[0] * y[1] + x[2] * y[1] + x[0] * y[2] - x[1] * y[2]) / ((x[1] - x[0]) * (x[1] - x[2]) * (x[2] - x[0]))
    p = np.array([a2, a1, a0])
    return p

# Converts a mask array to a binary array. Note: Also needs size of full array. Only 1d arrays
def mask_to_binary(arr, l):
    binary = np.zeros(l, dtype=bool)
    if len(arr) == 0:
        return binary
    for i in range(l):
        if i in arr:
            binary[i] = True
    return binary.astype(bool)


def estimate_continuum(x, y, width=7, n_knots=14, cont_val=0.9):
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

# Angular separation on the sky
def angular_sep(ra1, dec1, ra2, dec2):
    coord1 = SkyCoord(ra=ra1, dec=dec1, unit=(units.hourangle, units.deg))
    coord2 = SkyCoord(ra=ra2, dec=dec2, unit=(units.hourangle, units.deg))
    asep = np.abs(coord1.separation(coord2).value)
    return asep

# Horizontal median of a 2d image
def horizontal_median(image, width):
    ny, nx = image.shape
    out_image = np.empty(shape=(ny, nx), dtype=np.float64)
    for i in range(ny):
        out_image[i, :] = median_filter1d(image[i, :], width)
    return out_image

# Vertical median of a 2d image
def vertical_median(image, width):
    ny, nx = image.shape
    out_image = np.empty(shape=(ny, nx), dtype=np.float64)
    for i in range(nx):
        out_image[:, i] = median_filter1d(image[:, i], width)
    return out_image

# Out of bounds of image
def outob(x, y, nx, ny):
    return x + 1 > nx or x + 1 < 1 or y + 1 > ny or y + 1 < 1


# Returns a modified gaussian
def gauss_modified(x, amp, mu, sigma, d):
    return amp * np.exp(-0.5 * (np.abs((x - mu) / sigma))**(2 * d))


# Compatible with Parameters objects
def gauss_modified_solver(pars, x, data):
    model = gauss_modified(x, pars['amp'].value, pars['mu'].value, pars['sigma'].value, pars['d'].value)
    good = np.where(np.isfinite(data))[0]
    rms = np.sqrt(np.nansum((data - model)**2) / good.size)
    return rms, 1