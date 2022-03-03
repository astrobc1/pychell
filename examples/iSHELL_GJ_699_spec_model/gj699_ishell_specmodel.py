# Pychell deps
import pychell.spectralmodeling as pcsm
from pychell.data import ishell

# Optimize deps
from optimize import IterativeNelderMead

# Basic config for all chunks
spectrograph = "iSHELL"
data_input_path = "data/"
filelist = "filelist.txt"
star_name = "GJ_699" # Must be recognized by simbad
tag = "example" # Final output directory is {spectrograph}_{tag}
do_orders = [221, 222, 223]

#### MUST SET THIS ####
templates_path = "/Users/cale/Research/pychell_templates/"

#### This can be as is ####
output_path = "./"

# Loop over chunks (orders for ishell)
for order in do_orders:
    
    # Model
    wave_estimate = ishell.estimate_order_wls(order)
    pixmin, pixmax = 200, 1800
    sregion = pcsm.SpecRegion1d(pixmin=pixmin, pixmax=pixmax, wavemin=wave_estimate[pixmin], wavemax=wave_estimate[pixmax], order=order)
    model = pcsm.SpectralForwardModel(wls=pcsm.SplineWls(n_splines=6, bounds=[-0.05, 0.05]),
                                      continuum=pcsm.SplineContinuum(n_splines=14),
                                      lsf=pcsm.HermiteLSF(deg=6, sigma=ishell.lsf_sigma, coeff=[-0.05, 0.01, 0.05]),
                                      star=pcsm.AugmentedStar(input_file="/Users/cale/Research/SpectralTemplates/gj699_btsettl_kband.txt", star_name=star_name),
                                      gascell=pcsm.GasCell(input_file=templates_path + ishell.gascell_file, depth=ishell.gascell_depth),
                                      tellurics=pcsm.TelluricsTAPAS(input_file=templates_path + "TAPAS_tellurics_maunakea.npz"),
                                      sregion=sregion, oversample=8)
    
    # Create the "Problem" object.
    specrvprob = pcsm.SpectralRVProblem(spectrograph=spectrograph, model=model,
                                        data_input_path=data_input_path, filelist=filelist)
    
    # Run RVs for this order
    pcsm.compute_rvs_iteratively(specrvprob,
                                 obj=pcsm.RMSSpectralObjective(),
                                 optimizer=IterativeNelderMead(),
                                 augmenter=pcsm.WeightedMeanAugmenter(weight_tellurics=False, weight_fits=True),
                                 output_path=output_path, tag=tag, n_iterations=5, n_cores=8, do_ccf=False, verbose=True)