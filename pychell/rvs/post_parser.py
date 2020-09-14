import numpy as np
import pickle
import glob
import datetime
import os
from pdb import set_trace as stop

class PostParser:
    
    def __init__(self, output_path_root, do_orders=None, bad_rvs_dict=None, star_name='', xcorr=False, **kwargs):
        
        self.output_path_root = output_path_root

        if do_orders is None:
            self.do_orders = self.get_orders()
        else:
            self.do_orders = do_orders
            
        self.do_orders = np.atleast_1d(self.do_orders)
        self.n_orders = len(self.do_orders)
        
        self.bad_rvs_dict = {} if bad_rvs_dict is None else bad_rvs_dict
        self.star_name = star_name
        
        self.xcorr = xcorr
        
        self.star_name = star_name
        
        # Auto populate
        for key in kwargs:
            setattr(self, key, kwargs[key])
            
    def resolve_iter_indices(self, iter_indices):
        if iter_indices == 'best':
            _, iter_indices = self.get_best_iters()
        elif iter_indices is None:
            iter_indices = np.zeros(self.n_orders).astype(int) + self.n_iters_rvs - 1
        elif type(iter_indices) is int:
            iter_indices = np.zeros(self.n_orders).astype(int) + iter_indices
        return iter_indices
    
    def resolve_rvprec_templates(self, templates=None):
        
        if type(templates) is list:
            return templates
        elif type(templates) is str:
            return [templates]
        elif templates is None:
            _templates = []
            if 'gas_cell' in self.forward_models[0].templates_dict:
                _templates += ['gas_cell']
            if 'star' in self.forward_models[0].templates_dict:
                _templates += ['star']
            return _templates
        
            
    def parse_forward_models(self):
        if not hasattr(self, 'forward_models'):
            self.forward_models = []
            for o in range(self.n_orders):
                print('Loading in forward model for ' + os.path.basename(self.output_path_root[0:-1]) + ' , Order ' + str(self.do_orders[o]))
                try:
                    f = glob.glob(self.output_path_root + 'Order' + str(self.do_orders[o]) + os.sep + '*forward_models*.pkl')[0]
                    with open(f, 'rb') as ff:
                        self.forward_models.append(pickle.load(ff))
                except:
                    raise ValueError("Could not load forward model for order " + str(self.do_orders[o]))
            
            self.n_spec = self.forward_models[0].n_spec
            self.n_obs_nights = self.forward_models[0].n_obs_nights
            self.n_nights = self.forward_models[0].n_nights
            self.index_offset = int(not self.forward_models[0][0].models_dict['star'].from_synthetic)
            self.n_iters_rvs = self.forward_models[0].n_template_fits
            self.n_iters_opt = self.n_iters_rvs + self.index_offset
            self.tag = self.forward_models[0].tag + '_' + datetime.date.today().strftime("%d%m%Y")
            self.spectrograph = self.forward_models[0].spectrograph

    def parse_rms(self):
        
        if hasattr(self, 'rms'):
            return self.rms
        
        # Parse the fwms
        self.parse_forward_models()
            
        rms = np.empty(shape=(self.n_orders, self.n_spec, self.n_iters_opt), dtype=float)
        for o in range(self.n_orders):
            for ispec in range(self.n_spec):
                rms[o, ispec, :] = [self.forward_models[o][ispec].opt_results[k][1] for k in range(self.n_iters_opt)]
        self.rms = rms
        return self.rms

    def parse_stellar_templates(self):
        if hasattr(self, 'stellar_templates'):
            return self.stellar_templates
        
        self.stellar_templates = []
        for o in range(self.n_orders):
            f = glob.glob(self.output_path_root + 'Order' + str(self.do_orders[o]) + os.sep + 'Templates' + os.sep + '*stellar_templates*.npz')[0]
            self.stellar_templates.append(np.load(f)['stellar_templates'])
        return self.stellar_templates
            

    def parse_parameters(self):
    
        pars_numpy_vals = []
        pars_numpy_minvs = []
        pars_numpy_maxvs = []
        pars_numpy_varies = []
        pars_numpy_unc = []
        for o in range(n_orders):
            n_pars = len(forward_models[o].opt_results[0][0])
            pars_numpy_vals.append(np.zeros(shape=(n_spec, n_iters_opt, n_pars), dtype=bool))
            pars_numpy_minvs.append(np.zeros(shape=(n_spec, n_iters_opt, n_pars), dtype=bool))
            pars_numpy_maxvs.append(np.zeros(shape=(n_spec, n_iters_opt, n_pars), dtype=bool))
            pars_numpy_varies.append(np.zeros(shape=(n_spec, n_iters_opt, n_pars), dtype=bool))
            pars_numpy_unc.append(np.zeros(shape=(n_spec, n_iters_opt, n_pars), dtype=bool))
            for ispec in range(n_spec):
                for j in range(n_iters_opt):
                    pars_numpy_vals[o][ispec, j, :] = forward_models.opt_results[j][0].unpack()['value']
                    pars_numpy_minvs[o][ispec, j, :] = forward_models.opt_results[j][0].unpack()['minv']
                    pars_numpy_maxvs[o][ispec, j, :] = forward_models.opt_results[j][0].unpack()['maxv']
                    pars_numpy_varies[o][ispec, j, :] = forward_models.opt_results[j][0].unpack()['vary']
                    pars_numpy_unc[o][ispec, j, :] = forward_models.opt_results[j][0].unpack()['unc']
                    
        return pars_numpy_vals, pars_numpy_minvs, pars_numpy_maxvs, pars_numpy_varies, pars_numpy_unc

    def parse_rvs(self):
    
        # If exists, return
        if hasattr(self, 'rvs_dict'):
            return self.rvs_dict
        
        # Define new dictionary to only get rvs, not ccf info
        rvs_dict = {}
    
        # Load in a single forward model object to determine if x corr is set
        fname = glob.glob(self.output_path_root + 'Order' + str(self.do_orders[0]) + os.sep + 'RVs' + os.sep + '*.npz')[0]
        rvs0 = np.load(fname)
        rvs_dict['do_xcorr'] = True if 'rvs_xcorr' in rvs0 else False
        self.n_spec = rvs0['rvs'].shape[0]
        self.n_iters_rvs = rvs0['rvs'].shape[1]
        self.n_nights = len(rvs0['n_obs_nights'])
        rvs_dict['n_obs_nights'] = rvs0['n_obs_nights']
        self.do_xcorr = rvs_dict['do_xcorr']
        
        # Create arrays
        rvs_dict['rvs'] = np.full(shape=(self.n_orders, self.n_spec, self.n_iters_rvs), fill_value=np.nan)
        rvs_dict['rvs_nightly'] = np.full(shape=(self.n_orders, self.n_nights, self.n_iters_rvs), fill_value=np.nan)
        rvs_dict['unc_nightly'] = np.full(shape=(self.n_orders, self.n_nights, self.n_iters_rvs), fill_value=np.nan)
        rvs_dict['BJDS'] = rvs0['BJDS']
        rvs_dict['BJDS_nightly'] = rvs0['BJDS_nightly']
        
        if rvs_dict['do_xcorr']:
            rvs_dict['rvsx'] = np.full(shape=(self.n_orders, self.n_spec, self.n_iters_rvs), fill_value=np.nan)
            rvs_dict['rvsx_nightly'] = np.full(shape=(self.n_orders, self.n_nights, self.n_iters_rvs), fill_value=np.nan)
            rvs_dict['uncx_nightly'] = np.full(shape=(self.n_orders, self.n_nights, self.n_iters_rvs), fill_value=np.nan)
            rvs_dict['BIS'] = np.full(shape=(self.n_orders, self.n_spec, self.n_iters_rvs), fill_value=np.nan)

        # Load in rvs for each order
        for o in range(self.n_orders):
            print('Loading in RVs for ' + os.path.basename(self.output_path_root[0:-1]) + ' , Order ' + str(self.do_orders[o]))
            fname = glob.glob(self.output_path_root + 'Order' + str(self.do_orders[o]) + os.sep + 'RVs' + os.sep + '*.npz')[0]
            rvfile = np.load(fname)
            rvs_dict['rvs'][o, :, :] = rvfile['rvs']
            rvs_dict['rvs_nightly'][o, :, :] = rvfile['rvs_nightly']
            rvs_dict['unc_nightly'][o, :, :] = rvfile['unc_nightly']
            if rvs_dict['do_xcorr']:
                rvs_dict['rvsx'][o, :, :] = rvfile['rvs_xcorr']
                rvs_dict['rvsx_nightly'][o, :, :] = rvfile['rvs_xcorr_nightly']
                rvs_dict['uncx_nightly'][o, :, :] = rvfile['unc_xcorr_nightly']
                rvs_dict['BIS'][o, :, :] = rvfile['bis']
                
        self.rvs_dict = rvs_dict

        return self.rvs_dict
            
    def get_best_iters(self):
        
        best_iters = np.zeros(self.n_orders, dtype=int)
        best_stddevs = np.zeros(self.n_orders, dtype=int)
        
        for o in range(self.n_orders):
            stddevs = np.full(self.n_iters_rvs, fill_value=np.nan)
            for k in range(self.n_iters_rvs):
                if self.xcorr:
                    stddevs[k] = np.nanstd(self.rvs_dict['rvsx_nightly'][o, :, k])
                else:
                    stddevs[k] = np.nanstd(self.rvs_dict['rvs_nightly'][o, :, k])
            best_iters[o] = np.nanargmin(stddevs)
            best_stddevs[o] = stddevs[best_iters[o]]

        return best_stddevs, best_iters


    def get_orders(self):
        orders_found = glob.glob(self.output_path_root + "Order*" + os.sep)
        do_orders = np.array([int(o.split(os.sep)[-2][5:]) for o in orders_found])
        do_orders = np.sort(do_orders)
        return do_orders
    
    