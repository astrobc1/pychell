import os

from pychell.reduce.reducers import NightlyReducer
from pychell.reduce.extract import OptimalSlitExtractor
from pychell.reduce.order_map import DensityClusterTracer
from pychell.reduce.calib import PreCalibrator, FringingPreCalibrator

# Basic info
spectrograph = "iSHELL"
data_input_path = os.getcwd() + os.sep + "Vega" + os.sep
output_path = os.getcwd() + os.sep

# Create the class
reducer = NightlyReducer(spectrograph=spectrograph,
                         data_input_path=data_input_path, output_path=output_path,
                         pre_calib=FringingPreCalibrator(do_bias=False, do_dark=False, do_flat=True, flat_percentile=0.9, remove_blaze_from_flat=False, remove_fringing_from_flat=False),
                         tracer=DensityClusterTracer(trace_pos_poly_order=2,
                                                     mask_left=200, mask_right=200, mask_top=20, mask_bottom=20),
                         extractor=OptimalSlitExtractor(trace_pos_poly_order=4,
                                                        mask_left=200, mask_right=200, mask_top=20, mask_bottom=20,
                                                        oversample=4),
                         n_cores=1)

# Reduce the night
reducer.reduce_night()