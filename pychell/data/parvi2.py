# Base Python
import os
import copy
import glob

# Maths
import numpy as np

# Astropy
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units
from astropy.io import fits
import scipy.constants as cs

# Pychell deps
from pychell.data.parser import DataParser
import pychell.maths as pcmath
import pychell.data.spectraldata as pcspecdata


#######################
#### NAME AND SITE ####
#######################

spectrograph = "PARVI2"
observatory = {
    "name" : "Palomar",
    "lat": 33.3537819182,
    "long": -116.858929898,
    "alt": 1713.0
}

######################
#### DATA PARSING ####
######################
    
def parse_object(data):
    data.object = data.header["OBJECT"]
    return data.object
    
def parse_utdate(data):
    utdate = "".join(data.header["P200_UTC"].split('T')[0].split("-"))
    data.utdate = utdate
    return data.utdate
    
def parse_sky_coord(data):
    if data.header['P200RA'] is not None and data.header['P200DEC'] is not None:
        data.skycoord = SkyCoord(ra=data.header['P200RA'], dec=data.header['P200DEC'], unit=(units.hourangle, units.deg))
    else:
        data.skycoord = SkyCoord(ra=np.nan, dec=np.nan, unit=(units.hourangle, units.deg))
    return data.skycoord

def parse_itime(data):
    data.itime = data.header["EXPTIME"]
    return data.itime
    
def parse_exposure_start_time(data):
    data.time_obs_start = Time(float(data.header["START"]) / 1E9, format="unix")
    return data.time_obs_start

def parse_fiber_nums(data):
    return int(data.header["FIBER"])
    
def parse_spec1d(data):
    fits_data = fits.open(data.input_file)
    fits_data.verify('fix')
    data.header = fits_data[0].header
    data.wave = 10 * fits_data[4].data[0, data.order_num - 1, :]
    data.flux = fits_data[4].data[3, data.order_num - 1, :]
    data.flux_unc = fits_data[4].data[4, data.order_num - 1, :]
    data.mask = np.ones_like(data.flux)
    
def compute_exposure_midpoint(data):
    jdsi, jdsf = [], []
    # Eventually we will fill fluxes with an arbitrary read value.
    for key in data.header:
        if key.startswith("TIMEI"):
            jdsi.append(Time(float(data.header[key]) / 1E9, format="unix").jd)
        if key.startswith("TIMEF"):
            jdsf.append(Time(float(data.header[key]) / 1E9, format="unix").jd)
    mean_jd = (np.nanmax(jdsf) - np.nanmin(jdsi)) / 2 + np.nanmin(jdsi)
    return mean_jd

###################
#### WAVE INFO ####
###################

def estimate_wls(data, order_num=None, fiber_num=None):
    if order_num is None:
        order_num = data.order_num
    if fiber_num == 1:
        pcoeffs = wls_coeffs_fiber1[order_num - 1]
    else:
        pcoeffs = wls_coeffs_fiber3[order_num - 1]
    wls = np.polyval(pcoeffs, np.arange(2048))
    return wls


################################
#### REDUCTION / EXTRACTION ####
################################

read_noise = 0.0


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

# RV Zero point [m/s] (approx, fiber 3, Sci)
rv_zero_point = -5604.0

# LFC info
f0 = cs.c / (1559.91370 * 1E-9) # freq of pump line [Hz]
df = 10.0000000 * 1E9 # spacing of peaks [Hz]

# Approximate quadratic length solution coeffs as a starting point
wls_coeffs_fiber1 = np.array([np.array([-4.55076143e-06,  6.46954257e-02,  1.13576849e+04]), np.array([-4.58074699e-06,  6.52405082e-02,  1.14478637e+04]), np.array([-4.61025893e-06,  6.57787741e-02,  1.15394862e+04]), np.array([-4.63939424e-06,  6.63119836e-02,  1.16325870e+04]), np.array([-4.66824865e-06,  6.68418161e-02,  1.17272020e+04]), np.array([-4.69691659e-06,  6.73698714e-02,  1.18233684e+04]), np.array([-4.72549113e-06,  6.78976716e-02,  1.19211242e+04]), np.array([-4.75406395e-06,  6.84266626e-02,  1.20205092e+04]), np.array([-4.78272524e-06,  6.89582156e-02,  1.21215642e+04]), np.array([-4.81156368e-06,  6.94936289e-02,  1.22243315e+04]), np.array([-4.84066635e-06,  7.00341299e-02,  1.23288548e+04]), np.array([-4.87011868e-06,  7.05808770e-02,  1.24351795e+04]), np.array([-4.90000437e-06,  7.11349613e-02,  1.25433523e+04]), np.array([-4.93040529e-06,  7.16974093e-02,  1.26534218e+04]), np.array([-4.96140147e-06,  7.22691846e-02,  1.27654382e+04]), np.array([-4.99307094e-06,  7.28511908e-02,  1.28794536e+04]), np.array([-5.02548968e-06,  7.34442733e-02,  1.29955218e+04]), np.array([-5.05873155e-06,  7.40492227e-02,  1.31136989e+04]), np.array([-5.09286815e-06,  7.46667771e-02,  1.32340427e+04]), np.array([-5.12796873e-06,  7.52976250e-02,  1.33566134e+04]), np.array([-5.16410012e-06,  7.59424087e-02,  1.34814734e+04]), np.array([-5.20132656e-06,  7.66017272e-02,  1.36086876e+04]), np.array([-5.23970961e-06,  7.72761399e-02,  1.37383230e+04]), np.array([-5.27930803e-06,  7.79661700e-02,  1.38704496e+04]), np.array([-5.32017762e-06,  7.86723086e-02,  1.40051400e+04]), np.array([-5.36237113e-06,  7.93950185e-02,  1.41424695e+04]), np.array([-5.40593803e-06,  8.01347386e-02,  1.42825167e+04]), np.array([-5.45092442e-06,  8.08918885e-02,  1.44253631e+04]), np.array([-5.49737285e-06,  8.16668733e-02,  1.45710936e+04]), np.array([-5.54532210e-06,  8.24600885e-02,  1.47197965e+04]), np.array([-5.59480705e-06,  8.32719259e-02,  1.48715640e+04]), np.array([-5.64585843e-06,  8.41027786e-02,  1.50264918e+04]), np.array([-5.69850265e-06,  8.49530480e-02,  1.51846799e+04]), np.array([-5.75276153e-06,  8.58231494e-02,  1.53462324e+04]), np.array([-5.80865210e-06,  8.67135196e-02,  1.55112579e+04]), np.array([-5.86618631e-06,  8.76246238e-02,  1.56798699e+04]), np.array([-5.92537078e-06,  8.85569635e-02,  1.58521866e+04]), np.array([-5.98620651e-06,  8.95110850e-02,  1.60283316e+04]), np.array([-6.04868857e-06,  9.04875880e-02,  1.62084340e+04]), np.array([-6.11280573e-06,  9.14871353e-02,  1.63926289e+04]), np.array([-6.17854020e-06,  9.25104625e-02,  1.65810573e+04]), np.array([-6.24586713e-06,  9.35583892e-02,  1.67738669e+04]), np.array([-6.31475434e-06,  9.46318306e-02,  1.69712124e+04]), np.array([-6.38516178e-06,  9.57318091e-02,  1.71732556e+04]), np.array([-6.45704113e-06,  9.68594683e-02,  1.73801661e+04]), np.array([-6.53033529e-06,  9.80160868e-02,  1.75921219e+04])], dtype=np.ndarray)

wls_coeffs_fiber3 = np.array([np.array([-4.81010182e-06,  6.66206799e-02,  1.13620147e+04]), np.array([-4.80483769e-06,  6.68866934e-02,  1.14523351e+04]), np.array([-4.80246198e-06,  6.71772470e-02,  1.15440898e+04]), np.array([-4.80288294e-06,  6.74916068e-02,  1.16373143e+04]), np.array([-4.80601059e-06,  6.78290739e-02,  1.17320451e+04]), np.array([-4.81175682e-06,  6.81889840e-02,  1.18283200e+04]), np.array([-4.82003546e-06,  6.85707081e-02,  1.19261778e+04]), np.array([-4.83076234e-06,  6.89736526e-02,  1.20256587e+04]), np.array([-4.84385541e-06,  6.93972596e-02,  1.21268042e+04]), np.array([-4.85923479e-06,  6.98410069e-02,  1.22296572e+04]), np.array([-4.87682291e-06,  7.03044085e-02,  1.23342619e+04]), np.array([-4.89654453e-06,  7.07870147e-02,  1.24406641e+04]), np.array([-4.91832694e-06,  7.12884125e-02,  1.25489110e+04]), np.array([-4.94209998e-06,  7.18082261e-02,  1.26590518e+04]), np.array([-4.96779618e-06,  7.23461166e-02,  1.27711369e+04]), np.array([-4.99535093e-06,  7.29017832e-02,  1.28852190e+04]), np.array([-5.02470251e-06,  7.34749625e-02,  1.30013522e+04]), np.array([-5.05579230e-06,  7.40654300e-02,  1.31195928e+04]), np.array([-5.08856489e-06,  7.46729997e-02,  1.32399991e+04]), np.array([-5.12296820e-06,  7.52975246e-02,  1.33626317e+04]), np.array([-5.15895370e-06,  7.59388975e-02,  1.34875531e+04]), np.array([-5.19647649e-06,  7.65970512e-02,  1.36148283e+04]), np.array([-5.23549555e-06,  7.72719590e-02,  1.37445250e+04]), np.array([-5.27597386e-06,  7.79636352e-02,  1.38767131e+04]), np.array([-5.31787862e-06,  7.86721356e-02,  1.40114654e+04]), np.array([-5.36118145e-06,  7.93975584e-02,  1.41488575e+04]), np.array([-5.40585863e-06,  8.01400444e-02,  1.42889682e+04]), np.array([-5.45189125e-06,  8.08997776e-02,  1.44318790e+04]), np.array([-5.49926555e-06,  8.16769864e-02,  1.45776750e+04]), np.array([-5.54797310e-06,  8.24719438e-02,  1.47264448e+04]), np.array([-5.59801110e-06,  8.32849684e-02,  1.48782805e+04]), np.array([-5.64938267e-06,  8.41164251e-02,  1.50332779e+04]), np.array([-5.70209714e-06,  8.49667260e-02,  1.51915372e+04]), np.array([-5.75617040e-06,  8.58363314e-02,  1.53531626e+04]), np.array([-5.81162521e-06,  8.67257508e-02,  1.55182627e+04]), np.array([-5.86849160e-06,  8.76355435e-02,  1.56869510e+04]), np.array([-5.92680722e-06,  8.85663203e-02,  1.58593459e+04]), np.array([-5.98661780e-06,  8.95187444e-02,  1.60355711e+04]), np.array([-6.04797757e-06,  9.04935325e-02,  1.62157557e+04]), np.array([-6.11094972e-06,  9.14914563e-02,  1.64000347e+04]), np.array([-6.17560694e-06,  9.25133440e-02,  1.65885495e+04]), np.array([-6.24203190e-06,  9.35600816e-02,  1.67814477e+04]), np.array([-6.31031792e-06,  9.46326148e-02,  1.69788840e+04]), np.array([-6.38056947e-06,  9.57319503e-02,  1.71810205e+04]), np.array([-6.45290292e-06,  9.68591582e-02,  1.73880269e+04]), np.array([-6.52744721e-06,  9.80153735e-02,  1.76000812e+04])], dtype=np.ndarray)


# LSF widths (temporary)
lsf_linear_coeffs = np.array([0.00167515, 0.08559148])
lsf_widths = np.polyval(lsf_linear_coeffs, np.arange(46))
lsf_widths = [[lw*0.7, lw, lw*1.3] for lw in lsf_widths]