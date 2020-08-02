

import numpy as np
import pickle
import glob
import os
from pdb import set_trace as stop

def get_orders(output_path_root):
    orders_found = glob.glob(output_path_root + "Order*" + os.sep)
    do_orders = np.array([int(o[5:]) for o in orders_found])
    return do_orders

def parse_forward_models(output_path_root, do_orders=None):
    
    if do_orders is None:
        do_order = get_orders(output_path_root)
    
    forward_models = np.empty(shape=(n_ord, n_spec), dtype=object)
    n_orders = len(do_orders)
    for o in range(n_orders):
        for ispec in range(n_spec):
            forward_models[o, ispec] = parse_forward_model(output_path_root, do_orders[o], ispec + 1)
        
    return forward_models

def parse_rvs(output_path_root, do_orders=None):
    
    if do_orders is None:
        orders_found = glob.glob(output_path_root + "Order*" + os.sep)
        do_orders = np.array([int(o[5:]) for o in orders_found])
        
    n_orders = len(do_orders)
        
    # Load in the global parameters dictionary
    with open(output_path_root + 'global_parameters_dictionary.pkl', 'rb') as f:
        gpars = pickle.load(f)
    
    rvs_dict = {}
    
    # Load in a single forward model object to determine remaining things
    fname = glob.glob(output_path_root + 'Order' + str(do_orders[0]) + os.sep + 'RVs' + os.sep + '*.npz')[0]
    rvs0 = np.load(fname)
    n_spec, n_iters = rvs0['rvs'].shape
    n_obs_nights = rvs0['n_obs_nights']
    n_nights = len(n_obs_nights)
    rvs_dict['do_xcorr'] = True if 'rvs_xcorr' in rvs0 else False
        
        
    rvs_dict['rvs'] = np.zeros(shape=(n_orders, n_spec, n_iters)) + np.nan
    rvs_dict['rvs_nightly'] = np.zeros(shape=(n_orders, n_nights, n_iters)) + np.nan
    rvs_dict['unc_nightly'] = np.zeros(shape=(n_orders, n_nights, n_iters)) + np.nan
    rvs_dict['n_obs_nights'] = n_obs_nights
    rvs_dict['BJDS'] = rvs0['BJDS']
    rvs_dict['BJDS_nightly'] = rvs0['BJDS_nightly']
    
    if rvs_dict['do_xcorr']:
        rvs_dict['rvsx'] = np.zeros(shape=(n_orders, n_spec, n_iters)) + np.nan
        rvs_dict['rvsx_nightly'] = np.zeros(shape=(n_orders, n_nights, n_iters)) + np.nan
        rvs_dict['uncx_nightly'] = np.zeros(shape=(n_orders, n_nights, n_iters)) + np.nan

    # Load in rvs for each order
    for o in range(n_orders):
        print('Loading in RVs for Order ' + str(do_orders[o]))
        fname = glob.glob(output_path_root + 'Order' + str(do_orders[o]) + os.sep + 'RVs' + os.sep + '*.npz')[0]
        rvfile = np.load(fname)
        rvs_dict['rvs'][o, :, :] = rvfile['rvs']
        rvs_dict['rvs_nightly'][o, :, :] = rvfile['rvs_nightly']
        rvs_dict['unc_nightly'][o, :, :] = rvfile['unc_nightly']
        if rvs_dict['do_xcorr']:
            rvs_dict['rvsx'][o, :, :] = rvfile['rvs_xcorr']
            rvs_dict['rvsx_nightly'][o, :, :] = rvfile['rvs_xcorr_nightly']
            rvs_dict['uncx_nightly'][o, :, :] = rvfile['unc_xcorr_nightly']

    return rvs_dict


def parse_forward_model(output_path_root, order_num, spec_num):
    print('Parsing Forward Model For Order ' + str(order_num) + ', Observation ' + str(spec_num))
    fname = glob.glob(output_path_root + 'Order' + str(order_num) + os.sep + 'Fits' + os.sep + '*_forward_model_*_spec' + str(spec_num) + '.pkl')[0]
    with open(fname, 'rb') as f:
            fwm = pickle.load(f)
    return fwm

def parse_templates(output_path_root, do_orders=None):
    
    if do_orders is None:
        do_order = get_orders(output_path_root)
        
    n_orders = len(do_orders)
    templates = [{}]*n_orders
    for o in range(n_orders):
        fwm = parse_forward_model(output_path_root, order_num=do_orders[o], spec_num=1)
        for t in fwm.templates_dict:
            templates[o][t] = fwm.templates_dict[t]
            
    return templates

def parse_stellar_templates(output_path_root, do_orders=None, iter_indexes=None):
    
    if do_orders is None:
        do_order = get_orders(output_path_root)
        
    n_orders = len(do_orders)
    stellar_templates = []
    for o in range(n_orders):
        stellar_templates.append(parse_stellar_template(output_path_root, do_orders[o], iter_index=iter_indexes[o]))
            
    return stellar_templates

def parse_stellar_template(output_path_root, order_num, iter_index):
    f = glob.glob(output_path_root + 'Order' + str(order_num) + os.sep + 'Stellar_Templates' + os.sep + '*.npz')[0]
    template_temp = np.load(f)['stellar_templates']
    template = np.array([template_temp[:, 0], template_temp[:, iter_index + 1]]).T
    return template

def parse_rms(output_path_root, do_orders=None):
    
    if do_orders is None:
        do_order = get_orders(output_path_root)
    
    n_orders = len(do_orders)
    fwm0 = parse_forward_model(output_path_root, do_orders[0], 1)
    n_spec = len(glob.glob(output_path_root + 'Order' + str(do_orders[0]) +  os.sep + 'Fits' + os.sep + '*.pkl'))
    n_iters = fwm0.n_template_fits + (not fwm0.models_dict['star'].from_synthetic)
    rms = np.empty(shape=(n_orders, n_spec, n_iters), dtype=float)
    for o in range(n_orders):
        for ispec in range(n_spec):
            fwm = parse_forward_model(output_path_root, do_orders[o], ispec + 1)
            rms[o, ispec, :] = [fwm.opt[k][0] for k in range(n_iters)]
        
    return rms

def parameter_unpack(pars, iter_indexes):
    
    n_orders, n_spec, n_iters = pars.shape
    n_pars = len(pars[0, 0, 0].keys())
    pars_unpacked = np.empty(shape=(n_orders, n_spec, n_pars), dtype=float)
    varies_unpacked = np.empty(shape=(n_orders, n_spec, n_pars), dtype=bool)
    for o in range(n_orders):
        for ispec in range(n_spec):
            pars_unpacked[o, ispec, :] = pars[o, ispec, iter_indexes[o]].unpack()['values']
            varies_unpacked[o, ispec, :] = pars[o, ispec, iter_indexes[o]].unpack()['values']
                
    return pars_unpacked, varies_unpacked

def parse_parameters(output_path_root, do_orders=None):
    
    if do_orders is None:
        do_order = get_orders(output_path_root)
        
    n_orders = len(do_orders)
    fwm0 = parse_forward_model(output_path_root, do_orders[0], 1)
    n_spec = len(glob.glob(output_path_root + 'Order' + str(do_orders[0]) +  os.sep + 'Fits' + os.sep + '*.pkl'))
    n_iters = fwm0.n_template_fits + (not fwm0.models_dict['star'].from_synthetic)
    pars = np.empty(shape=(n_orders, n_spec, n_iters), dtype=object)
    for o in range(n_orders):
        for ispec in range(n_spec):
            fwm = parse_forward_model(output_path_root, do_orders[o], ispec + 1)
            pars[o, ispec, :] = [fwm.best_fit_pars[k] for k in range(n_iters)]
            
    return pars