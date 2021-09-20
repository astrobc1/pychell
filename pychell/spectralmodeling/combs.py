
# Maths
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.constants as cs
import scipy.signal
from scipy.interpolate import LSQUnivariateSpline
import pychell.maths as pcmath

class LFCAnalyzer:

    def __init__(self, f0, df):
        self.f0 = f0
        self.df = df

    def flag_bad_pixels(self, lfc_flux, width=3, thresh=9):

        # First remove negatives
        bad = np.where(lfc_flux < 0)[0]
        if bad.size > 0:
            lfc_flux[bad] = np.nan

        # Identify bad pixels in lfc flux
        lfc_flux_out = np.copy(lfc_flux)
        lfc_flux_smooth = pcmath.median_filter1d(lfc_flux, width=width)
        rel_errors = (lfc_flux - lfc_flux_smooth) / lfc_flux_smooth
        bad = np.where(np.abs(rel_errors) > thresh * np.nanstd(rel_errors))[0]
        if bad.size > 0:
            lfc_flux_out[bad] = np.nan
        return lfc_flux_out

    def gen_theoretical_peaks(self, lfc_wave):
        lfc_centers_freq_theoretical = np.arange(self.f0 - 10000 * self.df, self.f0 + 10001 * self.df, self.df)
        lfc_centers_wave_theoretical = cs.c / lfc_centers_freq_theoretical
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[::-1] * 1E10
        good = np.where((lfc_centers_wave_theoretical > np.nanmin(lfc_wave) - 2) & (lfc_centers_wave_theoretical < np.nanmax(lfc_wave) + 2))
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[good]
        return lfc_centers_wave_theoretical

    def estimate_background(self, lfc_wave, lfc_flux):
        background = pcmath.cspline_fit_fancy(lfc_wave, lfc_flux, window=1.25, n_knots=100, percentile=0)
        return background

    def estimate_continuum(self, lfc_wave, lfc_flux):
        continuum = pcmath.cspline_fit_fancy(lfc_wave, lfc_flux, window=1.0, n_knots=200, percentile=0.99)
        return continuum


class LFCWavelengthSolution(LFCAnalyzer):

    def __init__(self, f0, df, poly_order=None, n_knots=None, peak_separation=None):
        super().__init__(f0, df)
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

        # Flag bad pixels
        lfc_flux = self.flag_bad_pixels(lfc_flux)

        # Pixel grid
        nx = len(wave_estimate)
        xarr = np.arange(nx).astype(float)

        # Generate theoretical LFC peaks
        lfc_centers_wave_theoretical = self.gen_theoretical_peaks(wave_estimate)

        # Remove background flux
        background = self.estimate_background(wave_estimate, lfc_flux)
        lfc_flux_no_bg = lfc_flux - background
        lfc_peak_max = pcmath.weighted_median(lfc_flux_no_bg, percentile=0.75)
        
        # Estimate continuum
        continuum = self.estimate_continuum(wave_estimate, lfc_flux_no_bg)
        lfc_flux_norm = lfc_flux_no_bg / continuum

        # Estimate peaks in pixel space (just indices)
        peaks = scipy.signal.find_peaks(lfc_flux_norm, height=np.full(nx, 0.5), distance=0.8*self.peak_separation)[0]
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
            yy -= np.nanmin(yy)
            yy /= np.nanmax(yy)
            p0 = np.array([1.0, good_peaks[i], len(use) / 4])
            bounds = [(0.8, 1.5), (p0[1] - self.peak_separation / 2, p0[1] + self.peak_separation / 2), (0.25 * p0[2], 4*p0[2])]
            opt_result = scipy.optimize.minimize(self.fit_peak, p0, args=(xarr[use], lfc_flux_no_bg[use]), method='Nelder-Mead', bounds=bounds)
            pbest = opt_result.x
            lfc_centers_pix[i] = pbest[1]
            rms_norm[i] = opt_result.fun

        # Flag bad fits
        good_rms = pcmath.weighted_median(rms_norm, percentile=0.75)
        good = np.where(rms_norm < good_rms)[0]
        if good.size < 10:
            raise ValueError(f"LFC Peak fitting only found {good.size} < 10 good peaks!")
        lfc_centers_pix = lfc_centers_pix[good]

        # Determine which LFC spot matches each peak
        lfc_centers_wave = []
        for i in range(len(lfc_centers_pix)):
            diffs = np.abs(wave_estimate[int(np.round(lfc_centers_pix[i]))] - lfc_centers_wave_theoretical)
            k = np.argmin(diffs)
            lfc_centers_wave.append(lfc_centers_wave_theoretical[k])
        lfc_centers_wave = np.array(lfc_centers_wave)

        # Fit the peaks with splines or polynomial
        if self.n_knots is not None:
            knots = np.linspace(np.min(lfc_centers_pix) + 1E-5, np.max(lfc_centers_pix) - 1E-5, num=self.n_knots)
            wavelength_solution = pcmath.cspline_fit(lfc_centers_pix, lfc_centers_wave, knots)(xarr)
            bad = np.where((xarr < knots[0]) | (xarr > knots[-1]) | (wavelength_solution == 0))[0]
            if bad.size > 0:
                wavelength_solution[bad] = np.nan
        else:
            pfit = np.polyfit(lfc_centers_pix, lfc_centers_wave, self.poly_order)
            wavelength_solution = np.polyval(pfit, xarr)

        return wavelength_solution


    @staticmethod
    def fit_peak(pars, x, data):
        model = pcmath.gauss(x, *pars)
        rms = pcmath.rmsloss(data, model)
        return rms


class LFCGaussianFitter(LFCAnalyzer):

    def __init__(self, f0, df, n_knots, dl):
        super().__init__(f0, df)
        self.n_knots = n_knots
        self.dl = dl
        

    def compute_lsf_width(self, lfc_wave, lfc_flux):

        # Flag bad pixels
        lfc_flux = self.flag_bad_pixels(lfc_flux)

        # Pixel grid
        nx = len(lfc_wave)
        xarr = np.arange(nx).astype(float)

        # Generate theoretical LFC peaks
        lfc_centers_wave_theoretical = self.gen_theoretical_peaks(lfc_wave)

        # Remove background flux
        background = self.estimate_background(lfc_wave, lfc_flux)
        lfc_flux_no_bg = lfc_flux - background
        lfc_peak_max = pcmath.weighted_median(lfc_flux_no_bg, percentile=0.75)
        
        # Estimate continuum
        continuum = self.estimate_continuum(lfc_wave, lfc_flux_no_bg)
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

        # Fiducial Grid
        nx = int(np.ceil((knots[-1] - knots[0]) / self.dl))
        if nx % 2 == 0:
            nx += 1
        fiducial_grid = np.arange(int(-nx / 2), int(nx / 2) + 1) * self.dl
        lsf_init = cspline_fit(fiducial_grid)
        bad = np.where(lsf_init < 0)[0]
        if bad.size > 0:
            lsf_init[bad] = 0

        # Measure fwhm
        fwhm = pcmath.measure_fwhm(fiducial_grid, lsf_init)
        sigma = pcmath.fwhmtosigma(fwhm)

        # Fit with Gaussian
        p0 = np.array([1.0, 0.001, sigma])
        bounds = [(0.8, 1.2), (-0.1, 0.1), (0.8*sigma, 1.2*sigma)]
        fit_result = scipy.optimize.minimize(self.fit_peak, p0, args=(fiducial_grid, lsf_init), bounds=bounds, method="Nelder-Mead")
        pbest = fit_result.x
        sigma_best = pbest[2]

        return sigma_best


    @staticmethod
    def fit_peak(pars, x, data):
        model = pcmath.gauss(x, *pars)
        rms = pcmath.rmsloss(data, model)
        return rms