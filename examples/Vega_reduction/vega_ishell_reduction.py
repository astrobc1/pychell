# Base python
import os

# pychell deps
from pychell.reduce.recipes import ReduceRecipe
from pychell.reduce.extract import OptimalExtractor
from pychell.reduce.tracers import DensityClusterTracer

# Basic info
spectrograph = "iSHELL"
data_input_path = os.getcwd() + os.sep + "Vega" + os.sep
output_path = os.getcwd() + os.sep + "ReduceOutputs" + os.sep

# Order tracing
tracer = DensityClusterTracer(n_orders=29, poly_order=2, heights=30, order_spacing=20)

# Extraction
extractor = OptimalExtractor(trace_pos_poly_order=4, oversample=4,
                            n_trace_refine_iterations=3, n_extract_iterations=3,
                            badpix_threshold=5, trace_pos_refine_window=15)

# Create the primary Recipe class
recipe = ReduceRecipe(spectrograph=spectrograph,
                       data_input_path=data_input_path, output_path=output_path,
                       do_bias=False, do_flat=True, do_dark=False, flat_percentile=0.75,
                       mask_left=200, mask_right=200, mask_top=20, mask_bottom=20,
                       tracer=tracer,
                       extractor=extractor,
                       n_cores=1)

# Reduce the night
recipe.reduce()