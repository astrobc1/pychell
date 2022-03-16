
import numpy as np

# pychell
import pychell.maths as pcmath
import pychell.data as pcdata


def gen_master_bias(master_bias):
    """Generate a master bias image.

    Args:
        master_bias (MasterCal): The master bias object with the attribute group containing the individual bias images.

    Returns:
        np.ndarray: The master bias image.
    """
    bias_cube = group[0].generate_cube(master_bias.group)
    master_bias_image = np.nanmedian(bias_cube, axis=0)
    return master_bias_image


def gen_master_dark(master_dark, do_bias):
    """Generate a master dark image.

    Args:
        master_dark (MasterCal): The master dark object with the attribute group containing the individual darks.
        do_bias (bool): Whether or not to perform a master bias correction.

    Returns:
        np.ndarray: The master flat field image.
    """
    # Generate a data cube
    n_darks = len(master_dark.group)
    darks_cube = pcdata.Echellogram.generate_cube(master_dark.group)

    # Median crunch
    master_dark_image = np.nanmedian(darks_cube, axis=0)

    # Bias subtraction
    if do_bias:
        master_bias_image = master_dark.master_bias.parse_image()
        master_dark_image -= master_bias_image
    
    # Flag obvious bad pixels
    bad = np.where(master_dark_image < 0)
    if bad[0].size > 0:
        master_dark_image[bad] = np.nan
        
    # Return
    return master_dark_image


def gen_master_flat(master_flat, do_bias, do_dark):
    """Generate a master flat image.

    Args:
        master_flat (MasterCal): The master flat object with the attribute group containing the individual flats.
        do_bias (bool): Whether or not to perform a master bias correction.
        do_dark (bool): Whether or not to perform a master dark correction.

    Returns:
        np.ndarray: The master dark image.
    """
    
    # Generate a data cube
    flats_cube = pcdata.Echellogram.generate_cube(master_flat.group)

    # Median crunch
    master_flat_image = np.nanmedian(flats_cube, axis=0)

    # Dark and Bias subtraction
    if do_bias:
        master_bias_image = master_flat.master_bias.parse_image()
        master_flat_image -= master_bias_image
    if do_dark:
        master_dark_image = master_flat.master_dark.parse_image()
        master_flat_image -= master_dark_image

    # Normalize
    flat_percentile = 0.5
    master_flat_image /= pcmath.weighted_median(master_flat_image, percentile=flat_percentile)
    
    # Flag obvious bad pixels
    bad = np.where((master_flat_image <= flat_percentile*0.01) | (master_flat_image > flat_percentile * 100))
    if bad[0].size > 0:
        master_flat_image[bad] = np.nan
        
    # Return
    return master_flat_image


# def gen_master_fiber_flat(master_fiber_flat, do_bias, do_dark, do_flat):
#     """Generate a master fiber flat image.

#     Args:
#         master_flat (MasterCal): The master flat object with the attribute group containing the individual flats.
#         do_bias (bool): Whether or not to perform a master bias correction.
#         do_dark (bool): Whether or not to perform a master dark correction.

#     Returns:
#         np.ndarray: The master dark image.
#     """
#     # Generate a data cube
#     n_fiber_flats = len(master_fiber_flat.group)
#     fiber_flats_cube = pcdata.Echellogram.generate_cube(master_fiber_flat.group)

#     # Median crunch
#     master_fiber_flat_image = np.nanmedian(fiber_flats_cube, axis=0)

#     # Dark and Bias subtraction
#     if do_bias:
#         master_bias_image = master_fiber_flat.master_bias.parse_image()
#         master_fiber_flat_image -= master_bias_image
#     if do_dark:
#         master_dark_image = master_fiber_flat.master_dark.parse_image()
#         master_fiber_flat_image -= master_dark_image
#     if do_flat:
#         master_flat_image = master_fiber_flat.master_flat.parse_image()
#         master_fiber_flat_image /= master_flat_image
    
#     # Flag obvious bad pixels
#     bad = np.where(master_fiber_flat_image <= 0)
#     if bad[0].size > 0:
#         master_fiber_flat_image[bad] = np.nan
        
#     # Return
#     return master_fiber_flat_image

def gen_coadded_master_image(master_data):
    # Generate a data cube
    image_cube = master_data.generate_cube(master_data.group)

    # Median crunch
    master_image = np.nanmedian(image_cube, axis=0)
    
    # Flag obvious bad pixels
    bad = np.where(master_image <= 0)
    if bad[0].size > 0:
        master_image[bad] = np.nan
        
    # Return
    return master_image


def pre_calibrate(data, data_image, do_bias, do_dark, do_flat):
    """Perform bias, dark, and flat field corrections after calibration images are generated.

    Args:
        data (Echellogram): The data to calibrate.
        data_image ([type]): The image corresponding to data to calibrate.
        do_bias (bool): Whether or not to perform a master bias correction.
        do_dark (bool): Whether or not to perform a master dark correction.
        do_flat (bool): Whether or not to perform a master flat correction.
    """
    
    # Bias correction
    if do_bias:
        master_bias_image = data.master_bias.parse_image()
        data_image -= master_bias_image
        
    # Dark correction
    if do_dark:
        master_dark_image = data.master_dark.parse_image()
        data_image -= master_dark_image
        
    # Flat division
    if do_flat:
        master_flat_image = data.master_flat.parse_image()
        data_image /= master_flat_image