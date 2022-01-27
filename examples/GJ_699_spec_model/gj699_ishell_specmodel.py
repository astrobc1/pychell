# Base Python
import os
import copy

# Numpy
import numpy as np

# Pychell deps
from pychell.spectralmodeling.problems import IterativeSpectralRVProb
import pychell.spectralmodeling.models as pcsm
from pychell.spectralmodeling.objectives import WeightedSpectralUncRMS
from pychell.spectralmodeling.templateaugmenters import WeightedMean
import pychell.data.ishell as ishell

# Optimize deps
from optimize import IterativeNelderMead

# Define basic info
spectrograph = "iSHELL"
data_input_path = os.getcwd() + os.sep + "data" + os.sep
filelist = "filelist_example.txt"
output_path = os.getcwd() + os.sep
star_name = "GJ_699"
tag = "gj699_example"
do_orders = [6, 8, 12, 15] # Do some arbitrary orders (relative to the image)
templates_path = "/Users/cale/Research/pychell_templates/" # Must download templates

# Loop over orders
for order_num in do_orders:
    
    # Wavelength Solution
    wls = pcsm.SplineWls(n_splines=6)
    
    # Continuum
    continuum = pcsm.SplineContinuum(n_splines=10)
    
    # LSF
    lsf = pcsm.HermiteLSF(hermdeg=2, width=ishell.lsf_width)
    
    # Star
    star = pcsm.AugmentedStar(input_file=templates_path + "gj699_btsettl_kband.txt", star_name=star_name)

    # Gas cell
    gas_cell = pcsm.DynamicGasCell(input_file=templates_path + ishell.gas_cell_file, depth=ishell.gas_cell_depth)
    
    # Tellurics
    tellurics = pcsm.TelluricsTAPAS(input_path=templates_path, location_tag="maunakea")
    
    # Final spectral model
    spectral_model = pcsm.IterativeSpectralForwardModel(wls=wls, continuum=continuum, lsf=lsf,
                                                        star=star,
                                                        gas_cell=gas_cell,
                                                        tellurics=tellurics,
                                                        order_num=order_num,
                                                        n_iterations=5,
                                                        model_resolution=1)
    
    # Create the "Problem" object.
    specrvprob = IterativeSpectralRVProb(spectrograph=spectrograph,
                                         spectral_model=spectral_model,
                                         data_input_path=data_input_path, filelist=filelist, output_path=output_path,
                                         tag=tag,
                                         augmenter=WeightedMean(),
                                         obj=WeightedSpectralUncRMS(),
                                         optimizer=IterativeNelderMead(),
                                         n_cores=8,
                                         verbose=True)
    
    # Run RVs for this order
    specrvprob.compute_rvs_for_target()