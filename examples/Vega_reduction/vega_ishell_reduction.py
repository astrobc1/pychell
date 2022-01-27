# Base python
import os

# pychell deps
from pychell.reduce.recipes import ReduceRecipe
from pychell.reduce.optimal import OptimalExtractor
from pychell.reduce.trace import PeakTracer

# Basic info
spectrograph = "iSHELL"
data_input_path = os.getcwd() + os.sep + "Vega" + os.sep
output_path = os.getcwd() + os.sep + "ReduceOutputs" + os.sep

# Order tracing
tracer = PeakTracer(n_orders=25, poly_order=2, order_heights=28, order_spacing=30, xleft=500, xright=2048-210, n_slices=10)

# Extractor
extractor = OptimalExtractor(trace_pos_poly_order=4, oversample=16, extract_orders=[1, 2, 5, 10, 15, 20], badpix_threshold=4, remove_background=True, chunk_width=2048, chunk_overlap=0.5, n_iterations=50, trace_pos_refine_window=13, background_smooth_poly_order=None, background_smooth_width=None, min_profile_flux=0.01)

# Create the recipe
recipe = ReduceRecipe(spectrograph="iSHELL", data_input_path=data_input_path, output_path=output_path,
                      do_bias=False, do_dark=False, do_flat=True, flat_percentile=0.75,
                      xrange=[199, 1847],
                      poly_mask_bottom=[[200, 172], [1000, 262], [1800, 290]],
                      poly_mask_top=[[200, 1834], [1000, 1916], [1800, 1932]],
                      tracer=tracer,
                      extractor=extractor,
                      n_cores=1)

# Init the data
recipe.init_data()

# Reduce the night
recipe.reduce()