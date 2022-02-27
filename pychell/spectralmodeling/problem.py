# Base Python
import os
import glob

import dill

# Maths
import numpy as np

# pychell
import pychell.maths as pcmath
import pychell.utils as pcutils
import pychell.spectralmodeling as pcsm
import pychell.data as pcdata
import pychell.spectralmodeling.barycenter

class SpectralRVProblem:
    """The primary container for a spectral forward model problem where the goal is to provide precise RVs.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, spectrograph,
                 data_input_path, filelist,
                 model,
                 spec_mod_func=None):
        """Initiate the top level iterative spectral rv problem object.

        Args:
            spectrograph (str): The name of the spectrograph.
            data_input_path (str): The full path to the data folder.
            filelist (str): A text file listing the observations (filenames) within data_input_path to use.
            model (SpectralForwardModel): The spectral model obejct. For now only IterativeSpectralForwardModel is supported.
        """

        self.model = model
        self.spec_mod_func = spec_mod_func

        # Initialize the data
        self.init_data(spectrograph, data_input_path, filelist)


    def init_data(self, spectrograph, data_input_path, filelist):
        
        # List of input files
        input_files = [data_input_path + f for f in np.atleast_1d(np.genfromtxt(data_input_path + filelist, dtype='<U100', comments='#').tolist())]
        
        # Load in each observation for this order
        data = [pcdata.SpecData1d(fname, spectrograph, self.model.sregion, self.spec_mod_func) for ispec, fname in enumerate(input_files)]
        
        # Sort the data
        spec_module = pcutils.get_spec_module(spectrograph, self.spec_mod_func)
        jds = np.array([spec_module.parse_exposure_start_time(d) for d in data], dtype=float)
        ss = np.argsort(jds)
        self.data = [data[ss[i]] for i in range(len(jds))]

        # Barycenter corrections
        for ispec in range(len(self)):
            bjd, bc_vel = spec_module.get_barycenter_corrections(self.data[ispec], star_name=self.model.star.star_name)
            self.data[ispec].header["bjd"] = bjd
            self.data[ispec].header["bc_vel"] = bc_vel
    
    ###############
    #### MISC. ####
    ###############

    @property
    def spectrograph(self):
        return self.data[0].spectrograph

    @property
    def observatory(self):
        return self.spec_module.observatory

    @property
    def spec_module(self):
        return pcutils.get_spec_module(self.spectrograph, self.spec_mod_func)
    
    def __len__(self):
        return len(self.data)


    ##############
    #### SAVE ####
    ##############
    
    def save_to_pickle(self, output_path):
        fname = f"{output_path}spectralrvprob_{self.model.sregion.label.lower()}.pkl"
        with open(fname, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load_from_pickle(path, label):
        print(f"Loading in Spectral RV Problem for {label}")
        if path[-1] != os.sep:
            path += os.sep
        fname = glob.glob(f"{path}{label}{os.sep}*spectralrvprob*.pkl")[0]
        with open(fname, 'rb') as f:
            specrvprob = dill.load(f)
        return specrvprob