
# Maths
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.constants as cs
import scipy.signal
from scipy.interpolate import LSQUnivariateSpline
import pychell.maths as pcmath


class LFCWavelengthSolution:

    def __init__(self, f0, df, poly_order=None, n_knots=None, peak_separation=None):
        self.f0 = f0
        self.df = df
        if poly_order is None and n_knots is None:
            self.poly_order = 4
            self.n_knots = None
        elif n_knots is not None:
            self.n_knots = n_knots
            self.poly_order = None
        else:
            self.poly_order = poly_order
            self.n_knots = None
        self.peak_separation = peak_separation

    def compute_wls(self, wave_estimate, lfc_flux):

        # Identify bad pixels in lfc flux
        lfc_flux_cp = np.copy(lfc_flux)
        lfc_flux_smooth = pcmath.median_filter1d(lfc_flux_cp, width=3)
        rel_errors = (lfc_flux_cp - lfc_flux_smooth) / lfc_flux_smooth
        bad = np.where(np.abs(rel_errors) > 9 * np.nanstd(rel_errors))[0]
        if bad.size > 0:
            lfc_flux_cp[bad] = np.nan

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
        background = pcmath.cspline_fit_fancy(wave_estimate, lfc_flux_cp, window=1.25, n_knots=100, percentile=0)
        lfc_flux_no_bg = lfc_flux_cp - background
        lfc_peak_max = pcmath.weighted_median(lfc_flux_no_bg, percentile=0.75)

        # Estimate continuum
        continuum = pcmath.cspline_fit_fancy(wave_estimate, lfc_flux_no_bg, window=1.0, n_knots=200, percentile=0.99)
        lfc_flux_norm = lfc_flux_no_bg / continuum

        # Estimate peaks in pixel space (just indices)
        peaks = scipy.signal.find_peaks(lfc_flux_norm, height=np.full(nx, 0.5), distance=0.8*self.peak_separation)[0]
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
            p0 = np.array([np.nanmax(lfc_flux_no_bg[use]) / 2, good_peaks[i], len(use) / 4])
            bounds = [(1E-5, np.nanmax(lfc_flux_no_bg[use])), (p0[1] - self.peak_separation / 2, p0[1] + self.peak_separation / 2), (0.25 * p0[2], 4*p0[2])]
            opt_result = scipy.optimize.minimize(self.fit_peak, p0, args=(xarr[use], lfc_flux_no_bg[use]), method='Nelder-Mead', bounds=bounds)
            pbest = opt_result.x
            lfc_centers_pix[i] = pbest[1]

        # Determine which LFC spot matches each peak
        lfc_centers_wave = []
        for i in range(len(lfc_centers_pix)):
            try:
                diffs = np.abs(wave_estimate[int(np.round(lfc_centers_pix[i]))] - lfc_centers_wave_theoretical)
            except:
                breakpoint()
            k = np.argmin(diffs)
            lfc_centers_wave.append(lfc_centers_wave_theoretical[k])

        lfc_centers_wave = np.array(lfc_centers_wave)

        # Interpolate
        if self.n_knots is not None:
            knots = np.linspace(np.min(lfc_centers_pix) + 1E-5, np.max(lfc_centers_pix) - 1E-5, num=self.n_knots)
            wavelength_solution = pcmath.cspline_fit(lfc_centers_pix, lfc_centers_wave, knots)(xarr)
            bad = np.where((xarr < knots[0]) | (xarr > knots[-1]) | (wavelength_solution == 0))[0]
            if bad.size > 0:
                wavelength_solution[bad] = np.nan
        else:
            pfit = np.polyfit(lfc_centers_pix, lfc_centers_wave, 4)
            wavelength_solution = np.polyval(pfit, xarr)

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
        self.dl = dl
        

    def compute_lsf(self, lfc_wave, lfc_flux):

        # Identify bad pixels in lfc flux
        lfc_flux_cp = np.copy(lfc_flux)
        lfc_flux_smooth = pcmath.median_filter1d(lfc_flux_cp, width=3)
        rel_errors = (lfc_flux_cp - lfc_flux_smooth) / lfc_flux_smooth
        bad = np.where(np.abs(rel_errors) > 9 * np.nanstd(rel_errors))[0]
        if bad.size > 0:
            lfc_flux_cp[bad] = np.nan

        # Generate theoretical LFC peaks
        lfc_centers_freq_theoretical = np.arange(self.f0 - 10000 * self.df, self.f0 + 10001 * self.df, self.df)
        lfc_centers_wave_theoretical = cs.c / lfc_centers_freq_theoretical
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[::-1] * 1E10
        good = np.where((lfc_centers_wave_theoretical > np.nanmin(lfc_wave)) & (lfc_centers_wave_theoretical < np.nanmax(lfc_wave)))
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[good]

        # Estimate and remove background flux
        background = pcmath.cspline_fit_fancy(lfc_wave, lfc_flux_cp, window=1.25, n_knots=100, percentile=0)
        lfc_flux_no_bg = lfc_flux_cp - background
        lfc_peak_max = np.nanmax(lfc_flux_no_bg)

        # Estimate continuum
        continuum = pcmath.cspline_fit_fancy(lfc_wave, lfc_flux_no_bg, window=1.0, n_knots=200, percentile=0.99)
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
                flux_all += list(lfc_flux_norm[use])
        
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
        cspline_fit = LSQUnivariateSpline(waves_all, flux_all, t=knots[1:-1], k=3, ext=3)

        # Final grid
        nx = int(np.ceil((knots[-1] - knots[0]) / self.dl))
        if nx % 2 == 0:
            nx += 1
        fiducial_grid = np.arange(int(-nx / 2), int(nx / 2) + 1) * self.dl
        lsf = cspline_fit(fiducial_grid)
        bad = np.where(lsf < 0)[0]
        if bad.size > 0:
            lsf[bad] = 0

        # Normalize
        lsf -= np.nanmin(lsf)
        lsf /= np.nansum(lsf)

        return fiducial_grid, lsf


    @staticmethod
    def fit_peak(pars, x, data):
        model = pcmath.gauss(x, *pars)
        rms = pcmath.rmsloss(data, model)
        return rms