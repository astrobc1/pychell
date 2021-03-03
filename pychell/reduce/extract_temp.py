def extract_single_trace2(data, data_image, trace_map_image, trace_dict, config, refine_trace_pos=True):
    """Extract a single trace.

    Args:
        data (SpecDataImage): The data to extract.
        data_image (np.ndarray): The corresponding image.
        trace_map_image (np.ndarray): The image trace map image containing labels of each individual trace.
        trace_dict (dict): The dictionary containing location information for this trace
        config (dict): The reduction settings dictionary.
        refine_trace_pos (bool, optional): Whether or not to refine the trace position. Defaults to True.

    Returns:
        np.ndarray: The optimally reduced spectra with shape=(nx, 3)
        np.ndarray: The boxcar reduced spectra with shape=(nx,)
        CubicSpline: The trace profile defined by a CubicSpline object.
        y_positions_refined: The refined trace positions.
    """
    # Stopwatch
    stopwatch = pcutils.StopWatch()
    
    # Image dimensions
    ny, nx = data_image.shape
    
    # Helpful arrays
    xarr, yarr = np.arange(nx), np.arange(ny)
    
    # Extract the oversample factor
    M = config['oversample']
    
    #################################
    ##### Trace Profile & Y Pos #####
    #################################

    # Estimate y trace positions from the given order mapping
    y_positions_estimate = np.polyval(trace_dict['pcoeffs'], xarr)

    # Extract the height of the trace
    height = int(np.ceil(trace_dict['height']))

    # Create trace_image where only the relevant trace is seen, still ny x nx
    trace_image = np.copy(data_image)
    good_data = np.where(np.isfinite(trace_image))
    badpix_mask = np.zeros_like(trace_image)
    badpix_mask[good_data] = 1
    good_trace = np.where(trace_map_image == trace_dict['label'])
    bad_trace = np.where(trace_map_image != trace_dict['label'])
    if bad_trace[0].size > 0:
        trace_image[bad_trace] = np.nan
        badpix_mask[bad_trace] = 0
        
    # Flag obvious bad pixels
    trace_image_smooth = pcmath.median_filter2d(trace_image, width=5)
    med_val = pcmath.weighted_median(trace_image_smooth, percentile=0.99)
    bad = np.where((trace_image < 0) | (trace_image > 2 * med_val))
    if bad[0].size > 0:
        trace_image[bad] = np.nan
        badpix_mask[bad] = 0
        
    # The image in units of PE
    trace_image = convert_image_to_pe(trace_image, config['detector_props'])
    
    print('    Estimating Trace Profile ...', flush=True)
    
    # Estimate the trace profile from the current y positions
    trace_profile_cspline_estimate = estimate_trace_profile(trace_image, y_positions_estimate, height, M=M, mask_edges=[config['mask_trace_edges'], config['mask_trace_edges']])
        
    print('    Refining Trace Profile ...', flush=True)
    
    # Refine trace position with cross correlation
    y_positions_refined = np.copy(y_positions_estimate)
    trace_profile_cspline = copy.deepcopy(trace_profile_cspline_estimate)
    y_positions_refined = refine_trace_position(data, trace_image, y_positions_refined, trace_profile_cspline, badpix_mask, height, config, trace_pos_polyorder=config['trace_pos_polyorder'], M=M)
    
    # Now with a possibly better y positions array, re-estimate the trace profile.
    trace_profile_cspline = estimate_trace_profile(trace_image, y_positions_refined, height, M=M, mask_edges=[config['mask_trace_edges'], config['mask_trace_edges']])
    
    ###########################
    ##### Sky Subtraction #####
    ###########################
    
    # Estimate sky and remove from profile
    if config['sky_subtraction']:
        print('    Estimating Background Sky ...', flush=True)
        sky = estimate_sky(trace_image, y_positions_refined, trace_profile_cspline, height, n_sky_rows=config['n_sky_rows'], M=M)
        tp = trace_profile_cspline(trace_profile_cspline.x)
        #_, trace_max = estimate_trace_max(trace_profile_cspline)
        tp -= pcmath.weighted_median(tp, percentile=0.05)
        bad = np.where(tp < 0)[0]
        if bad.size > 0:
            tp[bad] = np.nan
        good = np.where(np.isfinite(tp))[0]
        trace_profile_cspline = scipy.interpolate.CubicSpline(trace_profile_cspline.x[good], tp[good])
    else:
        sky = None
        
    # Determine the fractions of the pixels used
    pixel_fractions = generate_pixel_fractions(trace_image, trace_profile_cspline, y_positions_refined, badpix_mask, height, min_profile_flux=config['min_profile_flux'])
    
    # Perform fit
    pars = OptimParams.Parameters()
    
    # Set points for trace position
    coeffs = np.polyfit(xarr, y_positions_refined, config["trace_pos_polyorder"])
    test_pixels_x = np.linspace(config["mask_left_edge"], config["mask_right_edge"], num=config["trace_pos_polyorder"] + 1).astype(int)
    for i in range(config["trace_pos_polyorder"] + 1):
        val = np.polyval(coeffs, test_pixels_x[i])
        vlb = val - height / 4
        vub = val + height / 4
        pars["trace_pos_" + str(i + 1)] = OptimParams.Parameter(name="trace_pos_" + str(i + 1), value=val, minv=vlb, maxv=vub, vary=True)
        
    breakpoint()
    
    # 
    
def fit_trace_wrapper(pars, trace_image, config, pixel_fractions, test_pixels_x, xarr):
    
    # Generate the trace polynomial coeffs
    trace_pos_points = np.array([pars["trace_pos_" + str(i + 1)].value for i in range(config["trace_pos_polyorder"])], dtype=float)
    trace_pos_coeffs = pcmath.poly_coeffs(test_pixels_x, trace_pos_points)
    
    # With the trace profile positions, determine the 
    pcmath.cspline_interp(c, y, xnew)