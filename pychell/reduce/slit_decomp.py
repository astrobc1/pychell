########################

def slit_decomposition_wrapper(data, trace_image, y_positions, trace_profile_cspline, badpix_mask, pixel_fractions, height, config, detector_props, sky=None, M=16, n_iters=100, n_chunks=5):
    
    goody, goodx = np.where(badpix_mask == 1)
    x_start, x_end = goodx[0], goodx[-1]
    y_start, y_end = goody[0], goody[-1]
    x_chunks = np.linspace(x_start, x_end, num=n_chunks+1).astype(int)
    
    for ichunk in range(n_chunks):
        
        chunk_x_start, chunk_x_end = x_chunks[ichunk], x_chunks[ichunk + 1]
        goody_chunk, _ = np.where(badpix_mask[:, chunk_x_start:chunk_x_end] == 1)
        chunk_y_start, chunk_y_end = goody_chunk[0], goody_chunk[-1]
        
        trace_image_chunk = trace_image[chunk_y_start:chunk_y_end, chunk_x_start:chunk_x_end]
        badpix_mask_chunk = badpix_mask[chunk_y_start:chunk_y_end, chunk_x_start:chunk_x_end]
        y_positions_chunk = y_positions[chunk_x_start:chunk_x_end] - chunk_y_start
        pixel_fractions_chunk = pixel_fractions[chunk_y_start:chunk_y_end, chunk_x_start:chunk_x_end]
        sky_chunk = sky[chunk_x_start:chunk_x_end]
        slit_decomposition_extraction(data, trace_image_chunk, y_positions_chunk, trace_profile_cspline, badpix_mask_chunk, pixel_fractions_chunk, height, config, detector_props, sky=sky_chunk, M=M, n_iters=100)
        

def slit_decomposition_extraction(data, trace_image, y_positions, trace_profile_cspline, badpix_mask, pixel_fractions, height, config, detector_props, sky=None, M=16, n_iters=100):
    
    print('Decomposing the slit function !')
    
    # Image dimensions
    ny, nx = trace_image.shape
    
    # Good x and y positions
    #goody, goodx = np.where(badpix_mask == 1)
    #x_start, x_end = np.min(goodx), np.max(goodx)
    #y_start, y_end = np.min(goody), np.max(goody)
    
    trace_profile_fiducial_grid, trace_profile = trace_profile_cspline.x, trace_profile_cspline(trace_profile_cspline.x)
    # The left and right profile positions
    left_trace_profile_inds = np.where(trace_profile_fiducial_grid < -1)[0]
    right_trace_profile_inds = np.where(trace_profile_fiducial_grid > 1)[0]
    left_trace_profile_ypos= trace_profile_fiducial_grid[left_trace_profile_inds]
    right_trace_profile_ypos = trace_profile_fiducial_grid[right_trace_profile_inds]
    left_trace_profile = trace_profile[left_trace_profile_inds]
    right_trace_profile = trace_profile[right_trace_profile_inds]
    
    # Find where it intersections at some minimum flux value
    left_ycut, _ = pcmath.intersection(left_trace_profile_ypos, left_trace_profile, config['min_profile_flux'], precision=1000)
    right_ycut, _ = pcmath.intersection(right_trace_profile_ypos, right_trace_profile, config['min_profile_flux'], precision=1000)
    
    # Find the aperture size at inf resolution
    # Find the aperture at finite resolution
    y_hr_temp = np.arange(0, ny, 1/M)
    window_size = right_ycut - left_ycut
    _, ny_decimal = pcmath.find_closest(y_hr_temp, window_size)
    #N = int(np.ceil(ny_decimal * M))
    N = M * ny + 1
    #ny = int(np.ceil(N / M))
    

    # Generate the current spectrum from standard optimal extraction
    current_spectrum, _, _ = pmassey_wrapper(data, trace_image, y_positions, trace_profile_cspline, badpix_mask, pixel_fractions, height, config, sky=sky)
    
    #S = np.zeros((ny, nx), dtype=float)
    #S = trace_image - np.outer(np.ones(ny), sky)
    #for x in range(nx):
        #ycol_lr_dec = np.arange(y_positions[x] - ny/2, y_positions[x] + ny/2)
    #    use = np.arange(y_positions[x] - ny/2, y_positions[x] + ny/2).astype(int)
    #    S[:, x] = trace_image[use, x] - sky[x]
    
    
    # Sub pixel fractions
    w = np.zeros(shape=(N, ny, nx), dtype=float)
    lagrange = 100.01
    yarr = np.arange(ny)
    arrhrbig = np.arange(-2*ny, 2*ny, 1/M)
    for x in range(nx):
        ypos = y_positions[x]
        ygrid_zerocenter_arbitrary = yarrbig - ypos
        #j0 = 
        # = int(ny/2) - ypos
        j0 = np.nanargmin(np.abs(ypos - arrhrbig))
        fr = np.min(np.abs(ypos*M - arrhrbig))
        for j in range(N):
            if j == j0:
                w[j, :, x] = fr
            elif j >= j0 + 1 and j <= j0 + M - 1:
                w[j, :, x] = 1 / M
            elif j == j0 + M:
                w[j, :, x] = 1 - fr
            else:
                continue
                
    # Helpful matrix
    B_upper = np.full(N - 1, fill_value=-1)
    B_lower = np.full(N - 1, fill_value=-1)
    B_main = np.full(N, fill_value=2)
    B = scipy.sparse.diags((B_lower, B_main, B_upper), (-1, 0, 1)).toarray()
    B[0, 0] = 1
    B[-1, -1] = 1
    f = np.copy(current_spectrum)
    wjkx = slit_decomp_sum(w)
    S = trace_image - np.outer(np.ones(trace_image[:, 0].size), sky)
    badf = np.where(~np.isfinite(f))[0]
    if badf.size > 0:
        f[badf] = 0
        S[:, badf] = 0
    bad_data = np.where(~np.isfinite(S)) #
    if bad_data[0].size > 0:
        S[bad_data] = 0

    n_iters = 100
    for iteration in range(n_iters):
        print(iteration + 1)
        f, g = slit_decomp_iter(f, N, M, B, S, w, lagrange, wjkx)
        
        #trace_profile_cspline_new = scipy.interpolate.CubicSpline(np.arange(0, ny, 1/M), g, extrapolate=True)
        #badpix_mask_new = flag_bad_pixels(S, f, y_positions, trace_profile_cspline_new, pixel_fractions, badpix_mask, height, sky=None, nsig=6)

    
@jit
def slit_decomp_iter(f, N, M, B, S, w, lagrange, wjkx):
    R = np.einsum('yx,x,kyx->k', S, f, w)
    #A = np.einsum('x,jkx->jk', f**2, wjkx)
    A = np.einsum("x,jyx,kyx->jk", f**2, w, w)
    g = np.linalg.solve(A + lagrange * B, R)
    g /= np.trapz(g)
    inner_sum = np.einsum('j,jyx->yx', g, w)
    C = np.nansum(S * inner_sum, axis=0)
    D = np.nansum(inner_sum**2, axis=0)
    #corrections = 
    fnew = (C / D) # / corrections
    
    return fnew, g

# Compute the quantity SUM_y w_j,y,x * w_k,y,x
#@njit(parallel=True)
def slit_decomp_sum(w):
    N, ny, nx = w.shape
    M = int((N - 1) / ny)
    wjkx = np.zeros(shape=(N, N, nx), dtype=float)
    for x in range(nx):
        for j in range(N):
            wjkx[j, :, x] = (M + 1) * w[j, 0, x] * w[:, 0, x]
            
    return wjkx