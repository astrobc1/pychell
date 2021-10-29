
# Maths
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.constants as cs
import scipy.signal
from scipy.interpolate import LSQUnivariateSpline
import pychell.maths as pcmath

class LFCAnalyzer:

    ####################
    #### INITIALIZE ####
    ####################

    def __init__(self, f0, df, peak_separation=None, wls_poly_order=None, n_wls_knots=None, n_lsf_knots=6, lsf_wave_spacing=0.01):
        self.f0 = f0
        self.df = df
        self.peak_separation = peak_separation
        if wls_poly_order is None and n_wls_knots is None:
            self.wls_poly_order = 6
            self.n_wls_knots = None
        elif n_wls_knots is not None:
            self.n_wls_knots = n_wls_knots
            self.wls_poly_order = None
        else:
            self.wls_poly_order = wls_poly_order
            self.n_wls_knots = None
        self.n_lsf_knots = n_lsf_knots
        self.lsf_wave_spacing = lsf_wave_spacing

    
    #########################
    #### PRIMARY METHODS ####
    #########################

    def compute_lsf_all(self, times_sci, times_cal, wls_cal_scifiber, lfc_cal_scifiber, do_orders=None):

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
                    lsfwidths_cal_scifiber[order_index, i] = self.compute_lsf(wls_cal_scifiber[:, order_index, i], lfc_cal_scifiber[:, order_index, i])
                except:
                    print(f"Warning! Could not compute LSF for order {order_index+1} cal spectrum {i+1}")


        for order_index in range(n_orders):

            if order_index + 1 not in do_orders:
                continue

            lsfwidths_sci_scifiber[order_index, :] = np.interp(times_sci, times_cal, lsfwidths_cal_scifiber[order_index, :], left=lsfwidths_cal_scifiber[order_index, 0], right=lsfwidths_cal_scifiber[order_index, -1])
        
        return lsfwidths_sci_scifiber

    def compute_lsf(self, lfc_wave, lfc_flux):

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
                flux_all += list(lfc_flux_no_bg[use] / np.nansum(lfc_flux_no_bg[use]))
        
        # Prep for Spline fit
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
        opt_result = scipy.optimize.minimize(self.fit_lsf, p0, args=(waves_all, flux_all), method='Nelder-Mead', bounds=bounds)
        pbest = opt_result.x
        sigma = pbest[1]

        return sigma

    def compute_wls_all(self, times_sci, times_cal, wave_estimate_sci, wave_estimate_cal, lfc_sci_calfiber, lfc_cal_calfiber, lfc_cal_scifiber, do_orders=None, method="interp"):

        # Numbers
        nx, n_orders = wave_estimate_sci.shape
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
        n_cal_spec = len(times_cal)
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
                    wls_cal_scifiber[:, order_index, i] = self.compute_wls(wave_estimate_sci[:, order_index], lfc_cal_scifiber[:, order_index, i])
                except:
                    print(f"Warning! Could not compute wls for order {order_index+1} cal spectrum {i+1}")
                
                try:
                    wls_cal_calfiber[:, order_index, i] = self.compute_wls(wave_estimate_cal[:, order_index], lfc_cal_calfiber[:, order_index, i])
                except:
                    print(f"Warning! Could not compute wls for order {order_index+1} cal spectrum {i+1}")

            # Loop over science spectra and compute wls for cal fiber
            if times_sci is not None:
                for i in range(n_sci_spec):
                    print(f"Computing cal fiber wls for order {order_index+1} science spectrum {i+1}")
                    try:
                        wls_sci_calfiber[:, order_index, i] = self.compute_wls(wave_estimate_cal[:, order_index], lfc_sci_calfiber[:, order_index, i])
                    except:
                        print(f"Warning! Could not compute wls for order {order_index+1} science spectrum {i+1}")

            # Loop over science observations, computer wls for science fiber
            if times_sci is not None:
                for i in range(n_sci_spec):
                    print(f"Computing sci fiber wls for order {order_index+1} science spectrum {i+1}")
                    if method == "interp":
                        for x in range(nx):
                            wls_sci_scifiber[x, order_index, i] = np.interp(times_sci[i], times_cal, wls_cal_scifiber[x, order_index, :], left=wls_cal_scifiber[x, order_index, 0], right=wls_cal_scifiber[x, order_index, -1])
                    elif method == "relinterp":
                        k_cal_start = np.where(times_cal < times_sci[i])[-1]
                        k_cal_end = np.where(times_cal > times_sci[i])[0]
                        for x in range(nx):
                            linear_coeffs_cal_scifiber = np.polyfit([times_cal[k_cal_start], times_cal[k_cal_end]], [wls_cal_scifiber[x, k_cal_start], wls_cal_scifiber[x, k_cal_end]])
                            linear_coeffs_cal_calfiber = np.polyfit([times_cal[k_cal_start], times_cal[k_cal_end]], [wls_cal_calfiber[x, k_cal_start], wls_cal_calfiber[x, k_cal_end]])
                            mean_slope = (linear_coeffs_cal_scifiber[0] + linear_coeffs_cal_calfiber[0]) / 2
                            wls_sci_scifiber[x, order_index, i] = np.interp(times_sci[i], times_cal, wls_cal_scifiber[x, order_index, :], left=wls_cal_scifiber[x, order_index, 0], right=wls_cal_scifiber[x, order_index, -1])
                    elif method == "nearest":
                        pass
                    else:
                        raise ValueError("method must be nearest or interp")

        return wls_sci_scifiber, wls_sci_calfiber, wls_cal_scifiber, wls_cal_calfiber

    def compute_wls(self, wave_estimate, lfc_flux):

        # Generate theoretical LFC peaks
        lfc_centers_wave_theoretical = self.gen_theoretical_peaks(wave_estimate)

        # Number of pixels
        nx = len(wave_estimate)
        xarr = np.arange(nx)
        
        # Flag bad pixels
        lfc_flux = self.flag_bad_pixels(lfc_flux)

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
            opt_result = scipy.optimize.minimize(self.fit_peak, p0, args=(xarr[use], lfc_flux_no_bg[use] / np.nanmax(lfc_flux_no_bg[use])), method='Nelder-Mead', bounds=bounds)
            pbest = opt_result.x
            lfc_centers_pix[i] = pbest[1]
            rms_norm[i] = opt_result.fun

        # Flag bad fits
        good_rms = pcmath.weighted_median(rms_norm, percentile=0.75)
        good = np.where(rms_norm < 3*good_rms)[0]
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
        if self.n_wls_knots is not None:
            knots = np.linspace(np.min(lfc_centers_pix) + 1E-5, np.max(lfc_centers_pix) - 1E-5, num=self.n_wls_knots)
            wavelength_solution = pcmath.cspline_fit(lfc_centers_pix, lfc_centers_wave, knots)(xarr)
            bad = np.where((xarr < knots[0]) | (xarr > knots[-1]) | (wavelength_solution == 0))[0]
            if bad.size > 0:
                wavelength_solution[bad] = np.nan
        else:
            pfit = np.polyfit(lfc_centers_pix, lfc_centers_wave, self.wls_poly_order)
            wavelength_solution = np.polyval(pfit, xarr)

        return wavelength_solution

    
    #################
    #### HELPERS ####
    #################

    def gen_theoretical_peaks(self, lfc_wave, n_left=10_000, n_right=10_000):
        lfc_centers_freq_theoretical = np.arange(self.f0 - n_left * self.df, self.f0 + (n_right + 1) * self.df, self.df)
        lfc_centers_wave_theoretical = cs.c / lfc_centers_freq_theoretical
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[::-1] * 1E10
        good = np.where((lfc_centers_wave_theoretical > np.nanmin(lfc_wave) - 2) & (lfc_centers_wave_theoretical < np.nanmax(lfc_wave) + 2))
        lfc_centers_wave_theoretical = lfc_centers_wave_theoretical[good]
        return lfc_centers_wave_theoretical

    def estimate_background(self, lfc_wave, lfc_flux):
        background = pcmath.generalized_median_filter1d(lfc_flux, width=int(2 * self.peak_separation), percentile=0.01)
        return background

    def estimate_continuum(self, lfc_wave, lfc_flux):
        continuum = pcmath.generalized_median_filter1d(lfc_flux, width=int(2 * self.peak_separation), percentile=0.99)
        return continuum

    @staticmethod
    def flag_bad_pixels(lfc_flux, smooth_width=3, sigma_thresh=9):

        # First remove negatives
        bad = np.where(lfc_flux < 0)[0]
        if bad.size > 0:
            lfc_flux[bad] = np.nan

        # Identify bad pixels in lfc flux
        lfc_flux_out = np.copy(lfc_flux)
        lfc_flux_smooth = pcmath.median_filter1d(lfc_flux, width=smooth_width)
        rel_errors = (lfc_flux - lfc_flux_smooth) / lfc_flux_smooth
        bad = np.where(np.abs(rel_errors) > sigma_thresh * np.nanstd(rel_errors))[0]
        if bad.size > 0:
            lfc_flux_out[bad] = np.nan
        return lfc_flux_out

    @staticmethod
    def fit_peak(pars, x, data):
        model = pcmath.gauss(x, *pars)
        rms = pcmath.rmsloss(data, model)
        return rms

    @staticmethod
    def fit_lsf(pars, x, data):
        model = pcmath.gauss(x, pars[0], 0, pars[1])
        rms = pcmath.rmsloss(data, model)
        return rms


