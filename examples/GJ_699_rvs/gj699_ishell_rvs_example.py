# Base Python
import os
import copy

# Numpy
import numpy as np

# Pychell deps
from pychell.spectralmodeling.spectralrvprob import IterativeSpectralRVProb
import pychell.spectralmodeling.spectralmodels as pcsm
from pychell.spectralmodeling.spectral_objectives import WeightedSpectralUncRMS
from pychell.spectralmodeling.template_augmenters import CubicSplineLSQ, WeightedMedian, WeightedMean
import pychell.data.ishell as ishell

# Optimize deps
from optimize.optimizers import IterativeNelderMead

# Define basic info
spectrograph = "iSHELL"
data_input_path = os.getcwd() + os.sep + "data" + os.sep
filelist = "filelist_example.txt"
output_path = os.getcwd() + os.sep
target_dict = {"name": "GJ_699"}
tag = "gj699_example"
do_orders = [6, 8, 12, 15] # Do some arbitrary orders high in RV content
templates_path = "/Users/gj_876/Research/pychell_templates/" # Must define this!

# Loop over orders
for order_num in do_orders:
    
    # Wls
    wavelength_solution = pcsm.SplineWavelengthSolution(n_splines=6)
    
    # Continuum
    continuum = pcsm.SplineContinuum(n_splines=7)
    
    # LSF
    lsf = pcsm.HermiteLSF(hermdeg=2, width=ishell.lsf_width)
    
    # Star
    star = pcsm.AugmentedStar(input_file=templates_path + "gj699_btsettl_kband.txt")
    
    # Gas cell
    gas_cell = pcsm.DynamicGasCell(input_file=templates_path + ishell.gas_cell_file, depth=ishell.gas_cell_depth)
    
    # Tellurics
    tellurics = pcsm.TelluricsTAPAS(input_path=templates_path, location_tag="maunakea")
    
    # Final spectral model
    spectral_model = pcsm.IterativeSpectralForwardModel(wavelength_solution=wavelength_solution, continuum=continuum, lsf=lsf,
                                                        star=star,
                                                        gas_cell=gas_cell,
                                                        tellurics=tellurics,
                                                        order_num=order_num,
                                                        n_iterations=5,
                                                        model_resolution=8)
    
    # Create the "Problem" object.
    specrvprob = IterativeSpectralRVProb(spectrograph=spectrograph,
                                         spectral_model=spectral_model,
                                         data_input_path=data_input_path, filelist=filelist, output_path=output_path,
                                         tag=tag,
                                         target_dict=target_dict,
                                         augmenter=CubicSplineLSQ(max_thresh=1.005, downweight_tellurics=True),
                                         obj=WeightedSpectralUncRMS(),
                                         optimizer=IterativeNelderMead(),
                                         n_cores=2,
                                         verbose=True)
    
    # Run RVs for this order
    specrvprob.compute_rvs_for_target()