

extraction_settings = {
    
    # Order map algorithm (options: 'from_flats, 'empirical')
    'order_map': 'empirical',
    
    # Pixels to mask on the top, bottom, left, and right edges
    'mask_left_edge': 20,
    'mask_right_edge': 20,
    'mask_top_edge': 20,
    'mask_bottom_edge': 20,
    
    # The height of an order is defined as where the flat is located.
    # This masks additional pixels on each side of the initial trace profile before moving forward.
    # The profile is further flagged after thes sky background is estimated.
    'mask_trace_edges':  3,
    
    # The degree of the polynomial to fit the individual order locations
    'trace_pos_polyorder' : 2,
    
    # Whether or not to perform a sky subtraction
    # The number of rows used to estimate the sky background (lowest n_sky_rows in the trace profile are used).
    'sky_subtraction': True,
    'n_sky_rows': 8,
    
    # The trace profile is constructed using oversampled data.
    # This is the oversample factor.
    'oversample': 16
    
}

forward_model_settings = {
    
    # The number of pixels to crop on each side of the spectrum
    'crop_pix': [10, 10],
    
    # If the user only wishes to compute the BJDS and barycorrs for later.
    'compute_bc_only': False,
    
    # Number of iterations to update the stellar template
    'n_template_fits': 10, # a zeroth iteration (flat template) does not count towards this number.
    
    # Cross correlation / bisector span stuff for each iteration. Will take longer.
    # A cross correlation will still be run to estimate the correct overall RV before fitting
    # if starting from a synthetic template
    'do_xcorr': False,
    
    # Model Resolution (n_model_pixels = model_resolution * n_data_pixels)
    # This is only important because of the instrument line profile (LSF)
    # 8 seems sufficient.
    'model_resolution': 8,
    
    # Which nights to use for the stellar template
    # Optional are: empty list for all nights, 'best' for highest snr night (from rms of fits), or a specific list of nights
    'nights_for_template': [],
    
    # The cross correlation grid and bisector span sampling
    'xcorr_vels': [12 * 1000, 10.0], #  # cc_vels = np.arange(best_vel - xcorr_vels[0], best_vel + xcorr_vels[0], xcorr_vels[1])
    'n_bs' : 1000,
    
    # The target function. Must live in pychell_target_functions.py
    'target_function': 'rms_model',
    
    # Flags the N worst pixels in fitting
    'flag_n_worst_pixels': 20
    
}