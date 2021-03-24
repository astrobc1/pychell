# Import the barycorrpy module
try:
    from barycorrpy import get_BC_vel
    from barycorrpy.utc_tdb import JDUTC_to_BJDTDB
except:
    pass
from astropy.time import Time
import sklearn.cluster
import glob
import numpy as np
import os
import importlib
import warnings
import copy
import pychell.data as pcdata
from astropy.io import fits
import sys
from astropy.coordinates import SkyCoord
import pychell.maths as pcmath
import astropy.units as units

class DataParser:
    
    def __init__(self, config):
        self.input_path = config["data_input_path"]
        try:
            self.run_output_path = config["run_output_path"]
        except:
            pass
        
        self.spectrograph = config["spectrograph"]
    
    def categorize_raw_data(self, config):
        # Allowed entries in data_dict:
        # flats, darks, bias
        # master_flats, master_darks, master_bias
        # science
        # wavecals
        raise NotImplementedError("Must implement a categorize method for this instrument")
    
    def load_specmod(self, relmod):
        try:
            return importlib.import_module(relmod)
        except:
            raise ValueError("Could not load spectrograph module for " + relmod)
            
    
    def parse_image_header(self, data):
        
        # Parse the header
        fits_data = fits.open(data.input_file)[0]
        
        # Just in case
        try:
            fits_data.verify('fix')
        except:
            pass
        
        data.header = fits_data.header
        
        # Parse the sky coord and time of obs
        self.parse_sky_coord(data)
        self.parse_time(data)
        self.parse_target(data)
        self.parse_itime(data)
        
        return data.header
    
    def parse_image(self, data):
        return fits.open(data.input_file, do_not_scale_image_data=True)[0].data.astype(float)
    
    def parse_date(self, data):
        raise NotImplementedError("Must implement a parse_date method for this instrument")
    
    def parse_image_num(self, data):
        raise NotImplementedError("Must implement a parse_image_num method for this instrument")
    
    def correct_readmath(self, data, data_image):
        # Corrects NDRs - Number of dynamic reads, or Non-destructive reads.
        # This reduces the read noise by sqrt(NDR)
        if 'NDR' in data.header:
            data_image /= float(data.header['NDR'])
        return data_image

    def gen_master_dark_filename(self, group):
        fname = self.run_output_path + os.sep + 'calib' + os.sep + 'master_dark_' + str(group[0].utdate) + str(group[0].itime) + 's.fits'
        return input_file
    
    def gen_master_flat_filename(self, group):
        img_nums = np.array([int(f.image_num) for f in group])
        img_start, img_end = str(np.min(img_nums)), str(np.max(img_nums))
        fname = self.run_output_path + 'calib' + os.sep + 'master_flat_' + str(group[0].utdate) + '_imgs' + img_start + '-' + img_end + '.fits'
        return fname
    
    def gen_order_map_filename(self, source):
        fname = self.run_output_path + 'calib' + os.sep + source.base_input_file_noext + '_order_map.fits'
        return fname
    
    def parse_itime(self, data):
        data.itime = data.header["ITIME"]
        return data.itime
    
    def gen_master_calib_header(self, data):
        data.header = copy.deepcopy(data.individuals[0].header)
        data.skycoord = copy.deepcopy(data.individuals[0].skycoord)
        data.time_obs_start = copy.deepcopy(data.individuals[0].time_obs_start)
    
    def parse_sky_coord(self, data):
        data.skycoord = SkyCoord(ra=data.header['RA'], dec=data.header['DEC'], unit=(units.hourangle, units.deg))
        return data.skycoord
        
    def parse_time(self, data):
        data.time_obs_start = Time(float(data.header['TIME']), scale='utc', format='jd')
        return data.time_obs_start
    
    def compute_barycenter_corrections(self, data, target=None):
        
        # Compute the jd mid point
        jdmid = self.compute_midpoint(data)
        
        # Parse the target
        if target is None:
            target = self.parse_target(data)
            
        # Parse the spectrograph mod
        relmod = 'pychell.data.' + self.spectrograph.lower()
        specmod = self.load_specmod(relmod)
        
        # BJD
        data.bjd = JDUTC_to_BJDTDB(JDUTC=jdmid, starname=target.replace('_', ' '), obsname=specmod.observatory['name'], leap_update=False)[0][0]
        
        # bc vel
        data.bc_vel = get_BC_vel(JDUTC=jdmid, starname=target.replace('_', ' '), obsname=specmod.observatory['name'], leap_update=False)[0][0]
        
        return data.bjd, data.bc_vel
        
    def load_barycenter_corrections(self, forward_models):
        
        # Check if forward models have a barycorr file attribute
        if hasattr(forward_models, 'bary_corr_file') and forward_models.bary_corr_file is not None:
            bjds, bc_vels = np.loadtxt(forward_models.data_input_path + forward_models.bary_corr_file, delimiter=',', unpack=True)
            bjds, bc_vels = np.atleast_1d(bjds), np.atleast_1d(bc_vels)
        else:
            bjds, bc_vels = np.zeros(forward_models.n_spec), np.zeros(forward_models.n_spec)
            for ispec in range(forward_models.n_spec):
                if hasattr(forward_models[ispec].data, 'bjd') and hasattr(forward_models[ispec].data, 'bc_vel'):
                    bjds[ispec] = forward_models[ispec].data.bjd
                    bc_vels[ispec] = forward_models[ispec].data.bc_vel
                else:
                    bjds[ispec], bc_vels[ispec] = self.compute_barycenter_corrections(forward_models[ispec].data, target=forward_models.star_name)
                
        if forward_models.compute_bc_only:
            np.savetxt(forward_models.run_output_path + 'bary_corrs_' + forward_models.star_name + '.txt', np.array([bjds, bc_vels]).T, delimiter=',')
            sys.exit("Compute BC info only is set!")
            
        for ispec in range(forward_models.n_spec):
            forward_models[ispec].data.bjd = bjds[ispec]
            forward_models[ispec].data.bc_vel = bc_vels[ispec]
    
    def compute_midpoint(self, data):
        return self.parse_time(data).jd + self.parse_itime(data) / (2 * 86400)
        
    def get_n_orders(self, data):
        raise NotImplementedError("Must implement get_n_orders")
        
    def get_n_traces(self, data):
        raise NotImplementedError("Must implement get_n_traces")
    
    def group_darks(self, darks):
        groups = []
        itimes = np.array([dark.itime for dark in darks])
        itimes_unq = np.unique(itimes)
        for t in itimes_unq:
            good = np.where(itimes == t)[0]
            indiv_darks = [darks[i] for i in good]
            groups.append(indiv_darks)
        return groups
    
    def group_flats(self, flats):
        
        # Groups
        groups = []
        
        # Number of total flats
        n_flats = len(flats)
        
        # Create a clustering object
        density_cluster = sklearn.cluster.DBSCAN(eps=0.01745, min_samples=2, metric='euclidean', algorithm='auto', p=None, n_jobs=1)
        
        # Points are the ra and dec and time
        dist_matrix = np.empty(shape=(n_flats, n_flats), dtype=float)
        for i in range(n_flats):
            for j in range(n_flats):
                dpsi = np.abs(flats[i].skycoord.separation(flats[j].skycoord).value)
                dt = np.abs(flats[i].time_obs_start.jd - flats[j].time_obs_start.jd)
                dpsi /= np.pi
                dt /= 10  # Places more emphasis on delta psi
                dist_matrix[i, j] = np.sqrt(dpsi**2 + dt**2)
        
        # Fit
        db = density_cluster.fit(dist_matrix)
        
        # Extract the labels
        labels = db.labels_
        good_labels = np.where(labels >= 0)[0]
        if good_labels.size == 0:
            raise ValueError('The flat pairing algorithm failed!')
        good_labels_init = labels[good_labels]
        labels_unique = np.unique(good_labels_init)
        
        # The number of master flats
        n_mflats = len(labels_unique)

        for l in range(n_mflats):
            this_label = np.where(good_labels_init == labels_unique[l])[0]
            indiv_flats = [flats[lb] for lb in this_label]
            groups.append(indiv_flats)
            
        return groups
    
    def pair_master_bias(self, data, master_bias):
        data.master_bias = master_bias
        
    def pair_order_map(self, data):
        pass
    
    def pair_master_dark(self, data, master_darks):
        n_masker_darks = len(master_darks)
        itimes = np.array([master_darks[i].itime for i in range(n_masker_darks)], dtype=float)
        good = np.where(data.itime == itimes)[0]
        if good.size != 1:
            raise ValueError(str(good.size) + " master dark(s) found for\n" + str(self))
        else:
            data.master_dark = master_darks[good[0]]
    
    def pair_master_flat(self, data, master_flats):
        ang_seps = np.array([np.abs(data.skycoord.separation(master_flat.skycoord)).value for master_flat in master_flats], dtype=float)
        time_seps = np.array([np.abs(data.time_obs_start.value - master_flat.time_obs_start.value) for master_flat in master_flats], dtype=float)
        ds = np.sqrt(ang_seps**2 + time_seps**2)
        minds_loc = np.argmin(ds)
        data.master_flat = master_flats[minds_loc]
        
    def save_reduced_orders(self, data, reduced_orders):
        fname = self.gen_reduced_spectra_filename(data)
        hdu = fits.PrimaryHDU(reduced_orders, header=data.header)
        hdu.writeto(fname, overwrite=True)

    def gen_reduced_spectra_filename(self, data):
        fname = self.run_output_path + 'spectra' + os.sep + data.base_input_file_noext + '_' + data.target + '_reduced.fits'
        return fname
        
    def print_summary(self, data_dict):
    
        n_sci_tot = len(data_dict['science'])
        targets_all = np.array([data_dict['science'][i].target for i in range(n_sci_tot)], dtype='<U50')
        targets_unique = np.unique(targets_all)
        for i in range(len(targets_unique)):
            
            target = targets_unique[i]
            
            locs_this_target = np.where(targets_all == target)[0]
            
            sci_this_target = [data_dict['science'][j] for j in locs_this_target]
            
            print('Target: ' + target)
            print('    N Exposures: ' + str(locs_this_target.size))
            if hasattr(sci_this_target[0], 'master_bias'):
                print('    Master Bias File(s): ')
                print('    ' + data_dict['science'].master_bias.base_input_file)
                
            if hasattr(sci_this_target[0], 'master_dark'):
                darks_this_target_all = np.array([sci.master_dark for sci in sci_this_target], dtype=pcdata.RawImage)
                darks_unique = np.unique(darks_this_target_all)
                print('  Master Dark File(s): ')
                for d in darks_unique:
                    print('    ' + d.base_input_file)
                
            if hasattr(sci_this_target[0], 'master_flat'):
                flats_this_target_all = np.array([sci.master_flat for sci in sci_this_target], dtype=pcdata.RawImage)
                flats_unique = np.unique(flats_this_target_all)
                print('  Master Flat File(s): ')
                for f in flats_unique:
                    print('    ' + f.base_input_file)
                    
            print('')
  
    def load_filelist(self, forward_models):
        input_files = [forward_models.data_input_path + f for f in np.atleast_1d(np.genfromtxt(forward_models.data_input_path + forward_models.flist_file, dtype='<U100', comments='#').tolist())]
        return input_files
            
class iSHELLParser(DataParser):
    
    def categorize_raw_data(self, config):

        # Stores the data as above objects
        data_dict = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files1 = glob.glob(self.input_path + '*data*.fits')
        sci_files2 = glob.glob(self.input_path + '*spc*.fits')
        sci_files = sci_files1 + sci_files2
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        n_sci_files = len(sci_files)
        
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
            
        # Darks assumed to contain dark in filename
        if config['dark_subtraction']:
            dark_files = glob.glob(self.input_path + '*dark*.fits')
            n_dark_files = len(dark_files)
            data_dict['darks'] = [pcdata.RawImage(input_file=dark_files[f], parser=self) for f in range(n_dark_files)]
            dark_groups = self.group_darks(data_dict['darks'])
            data_dict['master_darks'] = []
            for dark_group in dark_groups:
                master_dark_fname = self.gen_master_dark_filename(dark_group)
                data_dict['master_darks'].append(pcdata.MasterCalibImage(dark_group, input_file=master_dark_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_dark(sci, data_dict['master_darks'])
                
            for flat in data_dict['flats']:
                self.pair_master_dark(flat, data_dict['master_darks'])
        
        # iSHELL flats must contain flat in the filename
        if config['flat_division']:
            flat_files = glob.glob(self.input_path + '*flat*.fits')
            n_flat_files = len(flat_files)
            data_dict['flats'] = [pcdata.RawImage(input_file=flat_files[f], parser=self) for f in range(n_flat_files)]
            flat_groups = self.group_flats(data_dict['flats'])
            data_dict['master_flats'] = []
            for flat_group in flat_groups:
                master_flat_fname = self.gen_master_flat_filename(flat_group)
                data_dict['master_flats'].append(pcdata.MasterCalibImage(flat_group, input_file=master_flat_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_flat(sci, data_dict['master_flats'])
            
        # iSHELL ThAr images must contain arc (not implemented yet)
        if config['wavelength_calibration']:
            thar_files = glob.glob(self.input_path + '*arc*.fits')
            data_dict['wavecals'] = [pcdata.RawImage(input_file=thar_files[f], parser=self) for f in range(len(thar_files))]
            data_dict['master_wavecals'] = self.group_wavecals(data_dict['wavecals'])
            for sci in data_dict['science']:
                self.pair_master_wavecal(sci, data_dict['master_wavecals'])
                
                
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
        
    def parse_time(self, data):
        data.time_obs_start = Time(float(data.header['TCS_UTC']) + 2400000.5, scale='utc', format='jd')
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 1
    
    def get_n_orders(self, data):
        mode = data.header["XDTILT"].lower()
        if mode == "kgas":
            return 29
        else:
            return None
        
    def parse_spec1d(self, data):
        
        # Load the flux, flux unc, and bad pix arrays
        fits_data = fits.open(data.input_file)[0]
        fits_data.verify('fix')
        data.header = fits_data.header
        oi = data.order_num - 1
        data.flux, data.flux_unc, data.mask = fits_data.data[oi, 0, :, 0].astype(np.float64), fits_data.data[oi, 0, :, 1].astype(np.float64), fits_data.data[oi, 0, :, 2].astype(np.float64)

        # Flip the data so wavelength is increasing for iSHELL data
        data.flux = data.flux[::-1]
        data.mask = data.mask[::-1]
        data.flux_unc = data.flux_unc[::-1]
     
class SimulatedParser(DataParser):
        
    def parse_target(self, data):
        return "GJ_699"
        
    def parse_time(self, data):
        data.time_obs_start = Time(float(data.header['JD']), scale='utc', format='jd')
        return data.time_obs_start
    
    def get_n_orders(self, data):
        return 1
    
    def parse_itime(self, data):
        return 0
    
    def compute_midpoint(self, data):
        return data.time_obs_start.jd
        
    def parse_spec1d(self, data):
        
        # Load the flux, flux unc, and bad pix arrays
        fits_data = fits.open(data.input_file)[0]
        fits_data.verify('fix')
        data.header = fits_data.header
        data.default_wave_grid, data.flux, data.mask = fits_data.data[:, 0].astype(np.float64), fits_data.data[:, 1].astype(np.float64), fits_data.data[:, 2].astype(np.float64)
        data.flux_unc = np.ones_like(data.flux)
    
class MinervaNorthParser(DataParser):
    
    def categorize_raw_data(self, config):

        # Stores the data as above objects
        data_dict = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files1 = glob.glob(self.input_path + '*data*.fits')
        sci_files2 = glob.glob(self.input_path + '*spc*.fits')
        sci_files = sci_files1 + sci_files2
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        n_sci_files = len(sci_files)
        
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
            
        # Darks assumed to contain dark in filename
        if config['dark_subtraction']:
            dark_files = glob.glob(self.input_path + '*dark*.fits')
            n_dark_files = len(dark_files)
            data_dict['darks'] = [pcdata.RawImage(input_file=dark_files[f], parser=self) for f in range(n_dark_files)]
            dark_groups = self.group_darks(data_dict['darks'])
            data_dict['master_darks'] = []
            for dark_group in dark_groups:
                master_dark_fname = self.gen_master_dark_filename(dark_group)
                data_dict['master_darks'].append(pcdata.MasterCalibImage(dark_group, input_file=master_dark_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_dark(sci, data_dict['master_darks'])
                
            for flat in data_dict['flats']:
                self.pair_master_dark(flat, data_dict['master_darks'])
        
        # iSHELL flats must contain flat in the filename
        if config['flat_division']:
            flat_files = glob.glob(self.input_path + '*flat*.fits')
            n_flat_files = len(flat_files)
            data_dict['flats'] = [pcdata.RawImage(input_file=flat_files[f], parser=self) for f in range(n_flat_files)]
            flat_groups = self.group_flats(data_dict['flats'])
            data_dict['master_flats'] = []
            for flat_group in flat_groups:
                master_flat_fname = self.gen_master_flat_filename(flat_group)
                data_dict['master_flats'].append(pcdata.MasterCalibImage(flat_group, input_file=master_flat_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_flat(sci, data_dict['master_flats'])
            
        # iSHELL ThAr images must contain arc (not implemented yet)
        if config['wavelength_calibration']:
            thar_files = glob.glob(self.input_path + '*arc*.fits')
            data_dict['wavecals'] = [pcdata.RawImage(input_file=thar_files[f], parser=self) for f in range(len(thar_files))]
            data_dict['master_wavecals'] = self.group_wavecals(data_dict['wavecals'])
            for sci in data_dict['science']:
                self.pair_master_wavecal(sci, data_dict['master_wavecals'])
                
                
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
        
    def parse_time(self, data):
        data.time_obs_start = Time(float(data.header['JD']), scale='utc', format='jd')
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 1
    
    def get_n_orders(self, data):
        return 29
    
    def parse_itime(self, data):
        data.itime = data.header["EXPTIME"]
        return data.itime
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)[0]
        fits_data.verify('fix')
        data.header = fits_data.header
        oi = data.order_num - 1
        data.default_wave_grid, data.flux, data.flux_unc, data.mask = fits_data.data[oi, :, 0].astype(np.float64), fits_data.data[oi, :, 1].astype(np.float64), fits_data.data[oi, :, 2].astype(np.float64), fits_data.data[oi, :, 3].astype(np.float64)
             
class MinervaAustralisParser(DataParser):
    
    def categorize_raw_data(self, config):

        # Stores the data as above objects
        data_dict = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files1 = glob.glob(self.input_path + '*data*.fits')
        sci_files2 = glob.glob(self.input_path + '*spc*.fits')
        sci_files = sci_files1 + sci_files2
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        n_sci_files = len(sci_files)
        
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
            
        # Darks assumed to contain dark in filename
        if config['dark_subtraction']:
            dark_files = glob.glob(self.input_path + '*dark*.fits')
            n_dark_files = len(dark_files)
            data_dict['darks'] = [pcdata.RawImage(input_file=dark_files[f], parser=self) for f in range(n_dark_files)]
            dark_groups = self.group_darks(data_dict['darks'])
            data_dict['master_darks'] = []
            for dark_group in dark_groups:
                master_dark_fname = self.gen_master_dark_filename(dark_group)
                data_dict['master_darks'].append(pcdata.MasterCalibImage(dark_group, input_file=master_dark_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_dark(sci, data_dict['master_darks'])
                
            for flat in data_dict['flats']:
                self.pair_master_dark(flat, data_dict['master_darks'])
        
        # iSHELL flats must contain flat in the filename
        if config['flat_division']:
            flat_files = glob.glob(self.input_path + '*flat*.fits')
            n_flat_files = len(flat_files)
            data_dict['flats'] = [pcdata.RawImage(input_file=flat_files[f], parser=self) for f in range(n_flat_files)]
            flat_groups = self.group_flats(data_dict['flats'])
            data_dict['master_flats'] = []
            for flat_group in flat_groups:
                master_flat_fname = self.gen_master_flat_filename(flat_group)
                data_dict['master_flats'].append(pcdata.MasterCalibImage(flat_group, input_file=master_flat_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_flat(sci, data_dict['master_flats'])
            
        # iSHELL ThAr images must contain arc (not implemented yet)
        if config['wavelength_calibration']:
            thar_files = glob.glob(self.input_path + '*arc*.fits')
            data_dict['wavecals'] = [pcdata.RawImage(input_file=thar_files[f], parser=self) for f in range(len(thar_files))]
            data_dict['master_wavecals'] = self.group_wavecals(data_dict['wavecals'])
            for sci in data_dict['science']:
                self.pair_master_wavecal(sci, data_dict['master_wavecals'])
                
                
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
        
    def parse_time(self, data):
        data.time_obs_start = Time(float(data.header['JD']), scale='utc', format='jd')
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 1
    
    def get_n_orders(self, data):
        return 29
    
    def parse_itime(self, data):
        data.itime = data.header["ITIME"]
        return data.itime
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)[0]
        fits_data.verify('fix')
        data.header = fits_data.header
        oi = data.order_num - 1
        data.default_wave_grid, data.flux, data.flux_unc = fits_data.data[oi, :, 0].astype(np.float64), fits_data.data[oi, :, 1].astype(np.float64), fits_data.data[oi, :, 2].astype(np.float64)
        data.mask = np.ones(data.flux.size)
        
    def compute_barycenter_corrections(self, data, target=None):
        
        # Compute the jd mid point
        jdmid = self.compute_midpoint(data)
        
        # Parse the target
        if target is None:
            target = self.parse_target(data)
        
        # BJD
        data.bjd = JDUTC_to_BJDTDB(JDUTC=jdmid, starname=target.replace('_', ' '), lat=-27.7977, longi=151.8554, alt=682, leap_update=False)[0][0]
        
        # bc vel
        data.bc_vel = get_BC_vel(JDUTC=jdmid, starname=target.replace('_', ' '), lat=-27.7977, longi=151.8554, alt=682, leap_update=False)[0][0]
        
        return data.bjd, data.bc_vel
    
class PARVIParser(DataParser):
    
    def categorize_raw_data(self, config):

        # Stores the data as above objects
        data_dict = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files1 = glob.glob(self.input_path + '*data*.fits')
        sci_files2 = glob.glob(self.input_path + '*spc*.fits')
        sci_files = sci_files1 + sci_files2
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        n_sci_files = len(sci_files)
        
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
            
        # Darks assumed to contain dark in filename
        if config['dark_subtraction']:
            dark_files = glob.glob(self.input_path + '*dark*.fits')
            n_dark_files = len(dark_files)
            data_dict['darks'] = [pcdata.RawImage(input_file=dark_files[f], parser=self) for f in range(n_dark_files)]
            dark_groups = self.group_darks(data_dict['darks'])
            data_dict['master_darks'] = []
            for dark_group in dark_groups:
                master_dark_fname = self.gen_master_dark_filename(dark_group)
                data_dict['master_darks'].append(pcdata.MasterCalibImage(dark_group, input_file=master_dark_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_dark(sci, data_dict['master_darks'])
                
            for flat in data_dict['flats']:
                self.pair_master_dark(flat, data_dict['master_darks'])
        
        # iSHELL flats must contain flat in the filename
        if config['flat_division']:
            flat_files = glob.glob(self.input_path + '*flat*.fits')
            n_flat_files = len(flat_files)
            data_dict['flats'] = [pcdata.RawImage(input_file=flat_files[f], parser=self) for f in range(n_flat_files)]
            flat_groups = self.group_flats(data_dict['flats'])
            data_dict['master_flats'] = []
            for flat_group in flat_groups:
                master_flat_fname = self.gen_master_flat_filename(flat_group)
                data_dict['master_flats'].append(pcdata.MasterCalibImage(flat_group, input_file=master_flat_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_flat(sci, data_dict['master_flats'])
            
        # iSHELL ThAr images must contain arc (not implemented yet)
        if config['wavelength_calibration']:
            thar_files = glob.glob(self.input_path + '*arc*.fits')
            data_dict['wavecals'] = [pcdata.RawImage(input_file=thar_files[f], parser=self) for f in range(len(thar_files))]
            data_dict['master_wavecals'] = self.group_wavecals(data_dict['wavecals'])
            for sci in data_dict['science']:
                self.pair_master_wavecal(sci, data_dict['master_wavecals'])
                
                
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
        
    def parse_time(self, data):
        data.time_obs_start = Time(float(data.header["START"]) / 1E9, format="unix")
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 2
    
    def get_n_orders(self, data):
        mode = data.header["XDTILT"].lower()
        if mode == "kgas":
            return 29
        else:
            return None
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)
        fits_data.verify('fix')
        data.header = fits_data[0].header
        data.default_wave_grid = 10 * fits_data[1].data[0, data.order_num - 1, :]
        data.flux = fits_data[1].data[7, data.order_num - 1, :]
        data.flux_unc = fits_data[1].data[8, data.order_num - 1, :]
        data.mask = np.ones_like(data.flux)
        
    def compute_midpoint(self, data):
        jds, fluxes = [], []
        # Eventually we will fill fluxes with an arbitrary read value.
        # Then, the mean_jd will be computed with pcmath.weighted_mean(jds[1:], np.diff(fluxes))
        for key in data.header:
            if key.startswith("TIMEI"):
                jds.append(Time(float(data.header[key]) / 1E9, format="unix").jd)
        jds = np.array(jds)
        mean_jd = np.nanmean(jds)
        return mean_jd
                
class CHIRONParser(DataParser):
    
    def categorize_raw_data(self, config):

        # Stores the data as above objects
        data_dict = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files1 = glob.glob(self.input_path + '*data*.fits')
        sci_files2 = glob.glob(self.input_path + '*spc*.fits')
        sci_files = sci_files1 + sci_files2
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        n_sci_files = len(sci_files)
        
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
            
        # Darks assumed to contain dark in filename
        if config['dark_subtraction']:
            dark_files = glob.glob(self.input_path + '*dark*.fits')
            n_dark_files = len(dark_files)
            data_dict['darks'] = [pcdata.RawImage(input_file=dark_files[f], parser=self) for f in range(n_dark_files)]
            dark_groups = self.group_darks(data_dict['darks'])
            data_dict['master_darks'] = []
            for dark_group in dark_groups:
                master_dark_fname = self.gen_master_dark_filename(dark_group)
                data_dict['master_darks'].append(pcdata.MasterCalibImage(dark_group, input_file=master_dark_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_dark(sci, data_dict['master_darks'])
                
            for flat in data_dict['flats']:
                self.pair_master_dark(flat, data_dict['master_darks'])
        
        # iSHELL flats must contain flat in the filename
        if config['flat_division']:
            flat_files = glob.glob(self.input_path + '*flat*.fits')
            n_flat_files = len(flat_files)
            data_dict['flats'] = [pcdata.RawImage(input_file=flat_files[f], parser=self) for f in range(n_flat_files)]
            flat_groups = self.group_flats(data_dict['flats'])
            data_dict['master_flats'] = []
            for flat_group in flat_groups:
                master_flat_fname = self.gen_master_flat_filename(flat_group)
                data_dict['master_flats'].append(pcdata.MasterCalibImage(flat_group, input_file=master_flat_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_flat(sci, data_dict['master_flats'])
            
        # iSHELL ThAr images must contain arc (not implemented yet)
        if config['wavelength_calibration']:
            thar_files = glob.glob(self.input_path + '*arc*.fits')
            data_dict['wavecals'] = [pcdata.RawImage(input_file=thar_files[f], parser=self) for f in range(len(thar_files))]
            data_dict['master_wavecals'] = self.group_wavecals(data_dict['wavecals'])
            for sci in data_dict['science']:
                self.pair_master_wavecal(sci, data_dict['master_wavecals'])
                
                
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
        utdate = "".join(data.header["DATE-OBS"].split('-')[0:3])
        data.utdate = utdate
        return data.utdate
        
    def parse_sky_coord(self, data):
        data.skycoord = SkyCoord(ra=data.header['RA'], dec=data.header['DEC'], unit=(units.hourangle, units.deg))
        return data.skycoord
        
    def parse_time(self, data):
        data.time_obs_start = Time(data.header['DATE-OBS'])
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 1
    
    def get_n_orders(self, data):
        return 62
    
    def parse_itime(self, data):
        data.itime = data.header["EXPTIME"]
        return data.itime
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)[0]
        fits_data.verify('fix')
        data.header = fits_data.header
        oi = data.order_num - 1
        data.default_wave_grid, data.flux = fits_data.data[oi, :, 0].astype(np.float64), fits_data.data[oi, :, 1].astype(np.float64)
        data.flux_unc = np.zeros_like(data.flux) + 1E-3
        data.mask = np.ones_like(data.flux)
        
        
        
class IRDParser(DataParser):
    
    def categorize_raw_data(self, config):

        # Stores the data as above objects
        data_dict = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files1 = glob.glob(self.input_path + '*data*.fits')
        sci_files2 = glob.glob(self.input_path + '*spc*.fits')
        sci_files = sci_files1 + sci_files2
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        n_sci_files = len(sci_files)
        
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
            
        # Darks assumed to contain dark in filename
        if config['dark_subtraction']:
            dark_files = glob.glob(self.input_path + '*dark*.fits')
            n_dark_files = len(dark_files)
            data_dict['darks'] = [pcdata.RawImage(input_file=dark_files[f], parser=self) for f in range(n_dark_files)]
            dark_groups = self.group_darks(data_dict['darks'])
            data_dict['master_darks'] = []
            for dark_group in dark_groups:
                master_dark_fname = self.gen_master_dark_filename(dark_group)
                data_dict['master_darks'].append(pcdata.MasterCalibImage(dark_group, input_file=master_dark_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_dark(sci, data_dict['master_darks'])
                
            for flat in data_dict['flats']:
                self.pair_master_dark(flat, data_dict['master_darks'])
        
        # iSHELL flats must contain flat in the filename
        if config['flat_division']:
            flat_files = glob.glob(self.input_path + '*flat*.fits')
            n_flat_files = len(flat_files)
            data_dict['flats'] = [pcdata.RawImage(input_file=flat_files[f], parser=self) for f in range(n_flat_files)]
            flat_groups = self.group_flats(data_dict['flats'])
            data_dict['master_flats'] = []
            for flat_group in flat_groups:
                master_flat_fname = self.gen_master_flat_filename(flat_group)
                data_dict['master_flats'].append(pcdata.MasterCalibImage(flat_group, input_file=master_flat_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_flat(sci, data_dict['master_flats'])
            
        # iSHELL ThAr images must contain arc (not implemented yet)
        if config['wavelength_calibration']:
            thar_files = glob.glob(self.input_path + '*arc*.fits')
            data_dict['wavecals'] = [pcdata.RawImage(input_file=thar_files[f], parser=self) for f in range(len(thar_files))]
            data_dict['master_wavecals'] = self.group_wavecals(data_dict['wavecals'])
            for sci in data_dict['science']:
                self.pair_master_wavecal(sci, data_dict['master_wavecals'])
                
                
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
        
    def parse_time(self, data):
        data.time_obs_start = Time(float(data.header["START"]) / 1E9, format="unix")
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 2
    
    def get_n_orders(self, data):
        mode = data.header["XDTILT"].lower()
        if mode == "kgas":
            return 29
        else:
            return None
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)
        fits_data.verify('fix')
        data.header = fits_data[0].header
        data.default_wave_grid = 10 * fits_data[1].data[0, data.order_num - 1, :]
        data.flux = fits_data[1].data[7, data.order_num - 1, :]
        data.flux_unc = fits_data[1].data[8, data.order_num - 1, :]
        data.mask = np.ones_like(data.flux)
        
    def compute_midpoint(self, data):
        jds, fluxes = [], []
        # Eventually we will fill fluxes with an arbitrary read value.
        # Then, the mean_jd will be computed with pcmath.weighted_mean(jds[1:], np.diff(fluxes))
        for key in data.header:
            if key.startswith("TIMEI"):
                jds.append(Time(float(data.header[key]) / 1E9, format="unix").jd)
        jds = np.array(jds)
        mean_jd = np.nanmean(jds)
        return mean_jd