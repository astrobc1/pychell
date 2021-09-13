
# Maths
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.constants as cs
import scipy.signal
from scipy.interpolate import LSQUnivariateSpline
import pychell.maths as pcmath


class LFCWavelengthSolution:

    def __init__(self, f0, df):
        self.f0 = f0
        self.df = df

    def compute_wls(self, wave_estimate, lfc_flux):

        # Identify bad pixels in lfc flux
        lfc_flux_smooth = pcmath.median_filter1d(lfc_flux, width=3)
        rel_errors = (lfc_flux - lfc_flux_smooth) / lfc_flux_smooth
        bad = np.where(np.abs(rel_errors) > 6 * np.nanstd(rel_errors))[0]
        if bad.size > 0:
            lfc_flux[bad] = np.nan

        # Pixel grid
        nx = len(wave_estimate)
        xarr = np.arange(nx).astype(float)

        # Generate theoretical LFC peaks
        lfc_centers_freq_theoretical = np.arange(self.f0 - 10000 * self.df, self.f0 + 10001 * self.df, self.df)
        lfc_centers_wave_theoretical = cs.c / lfc_centers_freq_theoretical
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[::-1] * 1E10
        good = np.where((lfc_centers_wave_theoretical > np.nanmin(wave_estimate) - 2) & (lfc_centers_wave_theoretical < np.nanmax(wave_estimate) + 2))
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[good]

        # Estimate and remove background flux
        background = pcmath.cspline_fit_fancy(wave_estimate, lfc_flux, window=1.25, n_knots=100, percentile=0)
        lfc_flux_no_bg = lfc_flux - background
        lfc_peak_max = np.nanmax(lfc_flux_no_bg)

        # Estimate continuum
        continuum = pcmath.cspline_fit_fancy(wave_estimate, lfc_flux_no_bg, window=1.0, n_knots=200, percentile=0.99)
        lfc_flux_norm = lfc_flux_no_bg / continuum

        # Estimate peaks in pixel space (just indices)
        peaks = scipy.signal.find_peaks(lfc_flux_norm, height=np.zeros(nx) + 0.5, distance=5)[0]
        peaks = np.sort(peaks)

        # Estimate spacing between peaks, assume linear trend across order
        peak_spacing = np.polyval(np.polyfit(peaks[1:], np.diff(peaks), 1), xarr)

        # Only consider peaks with significant flux
        good_peaks = []
        for peak in peaks:
            if lfc_flux_no_bg[peak] >= 0.2 * lfc_peak_max:
                good_peaks.append(peak)
        good_peaks = np.array(good_peaks)

        # Fit each peak with a Gaussian
        lfc_centers_pix = np.full(good_peaks.size, np.nan)

        for i in range(len(good_peaks)):
            use = np.where((xarr >= good_peaks[i] - peak_spacing[good_peaks[i]] / 2) & (xarr < good_peaks[i] + peak_spacing[good_peaks[i]] / 2))[0]
            p0 = np.array([np.nanmax(lfc_flux_no_bg[use]), good_peaks[i], len(use) / 4])
            opt_result = scipy.optimize.minimize(self.fit_peak, p0, args=(xarr[use], lfc_flux_no_bg[use]), method='Nelder-Mead')
            pbest = opt_result.x
            lfc_centers_pix[i] = pbest[1]

        # Determine which LFC spot matches each peak
        lfc_centers_wave = []
        for i in range(len(lfc_centers_pix)):
            diffs = np.abs(wave_estimate[int(np.round(lfc_centers_pix[i]))] - lfc_centers_wave_theoretical)
            k = np.argmin(diffs)
            lfc_centers_wave.append(lfc_centers_wave_theoretical[k])

        lfc_centers_wave = np.array(lfc_centers_wave)

        # Interpolate
        #knots = np.linspace(np.min(lfc_centers_pix) + 0.0001, np.max(lfc_centers_pix) - 0.0001, num=9)
        #wavelength_solution = scipy.interpolate.CubicSpline(lfc_centers_pix, lfc_centers_wave, extrapolate=False, knots=knots)(xarr)
        #wavelength_solution = scipy.interpolate.LSQUnivariateSpline(lfc_centers_pix, lfc_centers_wave, t=knots, ext=3)(xarr)
        #bad = np.where((xarr < knots[0]) | (xarr > knots[-1]))[0]
        #if bad.size > 0:
        #    wavelength_solution[bad] = np.nan

        wavelength_solution = np.polyfit(lfc_centers_pix, lfc_centers_wave, 4)

        return wavelength_solution


    @staticmethod
    def fit_peak(pars, x, data):
        model = pcmath.gauss(x, *pars)
        rms = pcmath.rmsloss(data, model)
        return rms


class LFCLSF:

    def __init__(self, f0, df, n_knots, dl):
        self.f0 = f0
        self.df = df
        self.n_knots = n_knots
        

    def compute_lsf(self, lfc_wave, lfc_flux):

        # Identify bad pixels in lfc flux
        lfc_flux_smooth = pcmath.median_filter1d(lfc_flux, width=3)
        rel_errors = (lfc_flux - lfc_flux_smooth) / lfc_flux_smooth
        bad = np.where(np.abs(rel_errors) > 6 * np.nanstd(rel_errors))[0]
        if bad.size > 0:
            lfc_flux[bad] = np.nan

        # Generate theoretical LFC peaks
        lfc_centers_freq_theoretical = np.arange(self.f0 - 10000 * self.df, self.f0 + 10001 * self.df, self.df)
        lfc_centers_wave_theoretical = cs.c / lfc_centers_freq_theoretical
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[::-1] * 1E10
        good = np.where((lfc_centers_wave_theoretical > np.nanmin(lfc_wave)) & (lfc_centers_wave_theoretical < np.nanmax(lfc_wave)))
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[good]

        # Estimate and remove background flux
        background = pcmath.cspline_fit_fancy(lfc_wave, lfc_flux, window=1.25, n_knots=100, percentile=0)
        lfc_flux_no_bg = lfc_flux - background
        lfc_peak_max = np.nanmax(lfc_flux_no_bg)

        # Estimate continuum
        continuum = pcmath.cspline_fit_fancy(lfc_wave, lfc_flux_no_bg, window=1.0, n_knots=200, percentile=0.99)
        lfc_flux_norm = lfc_flux_no_bg / continuum

        # Peak spacing in wavelength space
        peak_spacing = np.polyval(np.polyfit(lfc_centers_wave_theoretical[1:], np.diff(lfc_centers_wave_theoretical), 1), lfc_wave)

        # Loop over theoretical peaks and shift
        waves_all = []
        flux_all = []
        for i in range(len(lfc_centers_wave_theoretical)):
            use = np.where((lfc_wave >= lfc_centers_wave_theoretical[i] - peak_spacing[lfc_centers_wave_theoretical[i]] / 2) & (lfc_wave < lfc_centers_wave_theoretical[i] + peak_spacing[lfc_centers_wave_theoretical[i]] / 2))[0]
            if len(use) >= 5:
                waves_all += list(lfc_wave[use] - lfc_centers_wave_theoretical[i])
                flux_all += list(lfc_flux[use])
        
        # Prep for Spline fit
        waves_all = np.array(waves_all, dtype=float)
        flux_all = np.array(flux_all, dtype=float)
        ss = np.argsort(waves_all)
        waves_all = waves_all[ss]
        flux_all = flux_all[ss]
        good = np.where(np.isfinite(waves_all) & np.isfinite(flux_all))[0]
        waves_all = waves_all[good]
        flux_all = flux_all[good]

        # Spline fit
        knots = np.linspace(np.nanmin(waves_all) + 0.0001, np.nanmax(waves_all) - 0.0001, num=self.n_knots)
        cspline_fit = LSQUnivariateSpline(waves_all, flux_all, t=knots[1:-1], k=3, ext=0)

        # Final grid
        #dw = knots.max() - knots.min()
        #fiducial_grid = np.arange( + self.dl, self.dl)
        #lsf = cspline_fit(fiducial_grid)

        return fiducial_grid, lsf


    @staticmethod
    def fit_peak(pars, x, data):
        model = pcmath.gauss(x, *pars)
        rms = pcmath.rmsloss(data, model)
        return rms