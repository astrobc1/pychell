# Base Python
import os
import importlib
import sys
import copy
import glob

# Astropy fits object
from astropy.io import fits

# Maths
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units
import sklearn.cluster

# Barycorrpy
from barycorrpy import get_BC_vel
from barycorrpy.utc_tdb import JDUTC_to_BJDTDB

# Pychell deps
import pychell.data.spectraldata as pcspecdata
import pychell.maths as pcmath


##############
#### SITE ####
##############

observatory = {
    'name': NotImplemented,
    'lat': NotImplemented,
    'lon': NotImplemented,
    'alt': NotImplemented
}

utc_offset = NotImplemented
rv_zero_point = 0

######################
#### DATA PARSING ####
######################

def categorize_raw_data(data_input_path, output_path):
    raise NotImplementedError("Must implement")


def pair_order_maps(data, order_maps):
    raise NotImplementedError("Must implement")


def parse_image_num(data):
    raise NotImplementedError("Must implement")

def parse_itime(data):
    raise NotImplementedError("Must implement")

def parse_object(data):
    raise NotImplementedError("Must implement")

def parse_utdate(data):
    raise NotImplementedError("Must implement")

def parse_sky_coord(data):
    raise NotImplementedError("Must implement")
    
def parse_exposure_start_time(data):
    raise NotImplementedError("Must implement")

def parse_image(data):
    raise NotImplementedError("Must implement")

def parse_image_header(data):
    raise NotImplementedError("Must implement")

def correct_readmath(data, data_image):
    raise NotImplementedError("Must implement")

def gen_master_calib_filename(master_cal):
    raise NotImplementedError("Must implement")

def gen_master_calib_header(master_cal):
    raise NotImplementedError("Must implement")

def pair_master_bias(data, master_bias):
    raise NotImplementedError("Must implement")

def pair_master_dark(data, master_darks):
    raise NotImplementedError("Must implement")

def pair_master_flat(data, master_flats):
    raise NotImplementedError("Must implement")

def group_darks(darks):
    raise NotImplementedError("Must implement")

def group_flats(flats):
    raise NotImplementedError("Must implement")


def parse_spec1d(data):
    raise NotImplementedError("Must implement")

def parse_fiber_nums(data):
    raise NotImplementedError("Must implement")


ECHELLE_ORDERS = NotImplemented


################################
#### BARYCENTER CORRECTIONS ####
################################

def compute_barycenter_corrections(data, star_name):
    raise NotImplementedError("Must implement")

def compute_exposure_midpoint(data):
    raise NotImplementedError("Must implement")


#########################
#### BASIC WAVE INFO ####
#########################

def estimate_wls(data):
    raise NotImplementedError("Must implement")


################################
#### REDUCTION / EXTRACTION ####
################################

# List of detectors.
read_noise = NotImplemented
dark_current = NotImplemented
gain = NotImplemented
