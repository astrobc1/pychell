# Base python
import os

# pychell deps
from pychell.reduce.reducers import StandardReducer
from pychell.reduce.extract import OptimalSlitExtractor
from pychell.reduce.tracers import DensityClusterTracer
from pychell.reduce.calib import PreCalibrator, FringingPreCalibrator

# Basic info
spectrograph = "iSHELL"
data_input_path = "/Users/cale/Desktop/examples/Vega_reduction/Vega/"
output_path = os.getcwd() + os.sep

# Pre calibration
pre_calib = FringingPreCalibrator(do_bias=False, do_dark=False, do_flat=True, flat_percentile=0.9, remove_blaze_from_flat=False, remove_fringing_from_flat=False)

# Order tracing
tracer = DensityClusterTracer(n_orders=29, trace_pos_poly_order=2, order_spacing=30, mask_left=200, mask_right=200, mask_top=50, mask_bottom=20)

# Extractor
extractor = OptimalSlitExtractor(trace_pos_poly_order=4, mask_left=200, mask_right=200, mask_top=50, mask_bottom=20, oversample=4)

# Create the class
reducer = StandardReducer(spectrograph=spectrograph,
                          data_input_path=data_input_path, output_path=output_path,
                          pre_calib=pre_calib,
                          tracer=tracer,
                          extractor=extractor,
                          n_cores=1)

# Reduce the night
reducer.reduce()