# Base Python
import os

from pychell.data.parser import DataParser
import glob
from astropy.io import fits
import pychell.data as pcdata

# Maths
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units
import scipy.constants as cs

# Pychell deps
import pychell.maths as pcmath

#######################
#### NAME AND SITE ####
#######################

spectrograph = "PARVI"
observatory = {
    "name" : "Palomar",
    "lat": 33.3537819182,
    "long": -116.858929898,
    "alt": 1713.0
}

######################
#### DATA PARSING ####
######################

class PARVIParser(DataParser):
    
    def categorize_raw_data(self, reducer):

        # Stores the data as above objects
        data_dict = {}
        
        # PARVI science files
        all_files = glob.glob(self.input_path + '*data*.fits')
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
        
        # Order map
        data_dict['order_maps'] = []
        for master_flat in data_dict['master_flats']:
            order_map_fname = self.gen_order_map_filename(source=master_flat)
            data_dict['order_maps'].append(pcdata.ImageMap(input_file=order_map_fname, source=master_flat,  parser=self, order_map_fun='trace_orders_from_flat_field'))
        for sci_data in data_dict['science']:
            self.pair_order_map(sci_data, data_dict['order_maps'])
        
        self.print_summary(data_dict)

        return data_dict
    
    def pair_order_map(self, data, order_maps):
        for order_map in order_maps:
            if order_map.source == data.master_flat:
                data.order_map = order_map
                return

    def parse_image_num(self, data):
        string_list = data.base_input_file.split('.')
        data.image_num = string_list[4]
        return data.image_num
        
    def parse_target(self, data):
        data.target = data.header["OBJECT"]
        return data.target
        
    def parse_utdate(self, data):
        utdate = "".join(data.header["DATE_OBS"].split('-'))
        data.utdate = utdate
        return data.utdate
        
    def parse_sky_coord(self, data):
        data.skycoord = SkyCoord(ra=data.header['TCS_RA'], dec=data.header['TCS_DEC'], unit=(units.hourangle, units.deg))
        return data.skycoord
    
    def parse_itime(self, data):
        data.itime = data.header["EXPTIME"]
        return data.itime
        
    def parse_exposure_start_time(self, data):
        data.time_obs_start = Time(float(data.header["START"]) / 1E9, format="unix")
        return data.time_obs_start

    def parse_fiber_num(self, data):
        return int(data.header["FIBER"])

    def classify_traces(self, data):
        pass
        # !!HERE!!
        #fibers = data.
        #pass
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)
        fits_data.verify('fix')
        data.header = fits_data[0].header
        data.apriori_wave_grid = 10 * fits_data[4].data[0, data.order_num - 1, :]
        data.flux = fits_data[4].data[3, data.order_num - 1, :]
        data.flux_unc = fits_data[4].data[4, data.order_num - 1, :]
        data.mask = np.ones_like(data.flux)
        
    def compute_exposure_midpoint(self, data):
        jds, fluxes = [], []
        # Eventually we will fill fluxes with an arbitrary read value.
        # Then, the mean_jd will be computed with pcmath.weighted_mean(jds[1:], np.diff(fluxes))
        for key in data.header:
            if key.startswith("TIMEI"):
                jds.append(Time(float(data.header[key]) / 1E9, format="unix").jd)
        jds = np.array(jds)
        mean_jd = np.nanmean(jds)
        return mean_jd



################################
#### REDUCTION / EXTRACTION ####
################################


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

# RV Zero point [m/s] (approx, fiber 3)
rv_zero_point = -5604.0

# LFC - Sci (fiber 1 - fiber 3)
fiber_diffs = np.array([-199.44961138537522, -152.2491592284224, -114.68619418459653, -85.05641027978223, -61.90739659769577, -44.00913351316186, -30.32698141833805, -19.997039483497606, -12.303753058397945, -6.659649465161012, -2.587083164676585, 0.2981273963817932, 2.30128752374409, 3.6617533271036304, 4.564311154865238, 5.149120693380538, 5.5203310798655325, 5.753477292074987, 5.901761821585354, 6.001324197120766, 6.0755982825650205, 6.138854411912023, 6.199020320850822, 6.259871468100101, 6.322677684843512, 6.387389119568016, 6.453440127304845, 6.520245053295908, 6.587454742971423, 6.6550370320943895, 6.723238384657834, 6.792477202179819, 6.8632120667730705, 6.935820238522258, 7.0105130339068005, 7.087305185200315, 7.166044832419973, 7.246499328898394, 7.328479436474287, 7.411970620603972, 7.497224888425416, 7.584749783191777, 7.675112578840279, 7.768457204678464, 7.863608744934647, 7.956615241830514])

# LSF widths (temporary)
lsf_linear_coeffs = np.array([0.00167515, 0.08559148])
lsf_widths = np.polyval(lsf_linear_coeffs, np.arange(46))
lsf_widths = [[lw*0.7, lw, lw*1.3] for lw in lsf_widths]

# Approximate wavelength solution coefficnents for each order, fiber 1 (LFC)
wls_coeffs = [np.array([7.20697948e-06, 3.67460556e-02, 1.12286214e+04]), np.array([5.05581638e-06, 4.24663111e-02, 1.13501810e+04]), np.array([3.22528768e-06, 4.73711982e-02, 1.14667250e+04]), np.array([1.67690770e-06, 5.15643447e-02, 1.15794400e+04]), np.array([3.75534581e-07, 5.51396921e-02, 1.16893427e+04]), np.array([-7.10802964e-07,  5.81820837e-02,  1.17972991e+04]), np.array([-1.61107682e-06,  6.07678378e-02,  1.19040438e+04]), np.array([-2.35142842e-06,  6.29653049e-02,  1.20101966e+04]), np.array([-2.95533525e-06,  6.48354095e-02,  1.21162779e+04]), np.array([-3.44377496e-06,  6.64321764e-02,  1.22227231e+04]), np.array([-3.83538672e-06,  6.78032403e-02,  1.23298942e+04]), np.array([-4.14662982e-06,  6.89903401e-02,  1.24380922e+04]), np.array([-4.39193925e-06,  7.00297962e-02,  1.25475660e+04]), np.array([-4.58387820e-06,  7.09529713e-02,  1.26585223e+04]), np.array([-4.73328709e-06,  7.17867142e-02,  1.27711326e+04]), np.array([-4.84942915e-06,  7.25537869e-02,  1.28855405e+04]), np.array([-4.94013215e-06,  7.32732739e-02,  1.30018676e+04]), np.array([-5.01192617e-06,  7.39609738e-02,  1.31202183e+04]), np.array([-5.07017706e-06,  7.46297729e-02,  1.32406847e+04]), np.array([-5.11921536e-06,  7.52900004e-02,  1.33633496e+04]), np.array([-5.16246045e-06,  7.59497651e-02,  1.34882899e+04]), np.array([-5.20253946e-06,  7.66152723e-02,  1.36155790e+04]), np.array([-5.24140082e-06,  7.72911219e-02,  1.37452888e+04]), np.array([-5.28042184e-06,  7.79805856e-02,  1.38774911e+04]), np.array([-5.32051014e-06,  7.86858642e-02,  1.40122593e+04]), np.array([-5.36219842e-06,  7.94083236e-02,  1.41496685e+04]), np.array([-5.40573212e-06,  8.01487089e-02,  1.42897968e+04]), np.array([-5.45114946e-06,  8.09073365e-02,  1.44327255e+04]), np.array([-5.49835342e-06,  8.16842630e-02,  1.45785395e+04]), np.array([-5.54717499e-06,  8.24794307e-02,  1.47273271e+04]), np.array([-5.59742715e-06,  8.32927882e-02,  1.48791809e+04]), np.array([-5.64894883e-06,  8.41243860e-02,  1.50341969e+04]), np.array([-5.70163833e-06,  8.49744455e-02,  1.51924756e+04]), np.array([-5.75547508e-06,  8.58434015e-02,  1.53541213e+04]), np.array([-5.81052925e-06,  8.67319152e-02,  1.55192427e+04]), np.array([-5.86695809e-06,  8.76408587e-02,  1.56879531e+04]), np.array([-5.92498805e-06,  8.85712681e-02,  1.58603707e+04]), np.array([-5.98488157e-06,  8.95242649e-02,  1.60366187e+04]), np.array([-6.04688747e-06,  9.05009435e-02,  1.62168261e+04]), np.array([-6.11117352e-06,  9.15022230e-02,  1.64011279e+04]), np.array([-6.17773991e-06,  9.25286629e-02,  1.65896656e+04]), np.array([-6.24631195e-06,  9.35802387e-02,  1.67825877e+04]), np.array([-6.31621058e-06,  9.46560766e-02,  1.69800498e+04]), np.array([-6.38619856e-06,  9.57541455e-02,  1.71822150e+04]), np.array([-6.45430063e-06,  9.68709018e-02,  1.73892538e+04]), np.array([-6.51759531e-06,  9.80008858e-02,  1.76013435e+04])]

# LFC info
f0 = cs.c / (1559.91370 * 1E-9) # freq of pump line [Hz]
df = 10.0000000 * 1E9 # spacing of peaks [Hz]