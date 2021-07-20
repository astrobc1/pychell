
# Base python
import warnings

# Maths
import numpy as np

## Pytorch
try:
    import pytorch as torch
except:
    warnings.warn("Could not import pytorch (cannot use any pytorch based optimizers)")
    
# Optimize
from optimize.frameworks import OptProblem

class GlobalFitter(OptProblem):
    """Global fitter for a single order.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, spectrograph,
                 data_input_path, filelist,
                 spectral_model,
                 order_num, tag, output_path, target_dict,
                 bc_corrs=None, crop_pix=[200, 200],
                 n_iterations=10, model_resolution=8,
                 optimizer=None,
                 n_cores=1, verbose=True):
        
        # The number of cores
        self.n_cores = n_cores
        
        # The image order (base 1)
        self.order_num = order_num
        
        # The left and right pixels to crop
        self.crop_pix = crop_pix
        
        # Number of iterations
        self.n_iterations = n_iterations
        
        # Verbose
        self.verbose = verbose
        
        # Input path
        self.data_input_path = data_input_path
        self.filelist = filelist
        
        # The base output path
        self.output_path = output_path

        # The spectrogaph
        self.spectrograph = spectrograph
        self._init_spectrograph()
        
        # Full tag is spectrograph_ + tag
        self.tag = f"{self.spectrograph.lower()}_{tag}"
        
        # The actual output path
        self.output_path += self.tag + os.sep
        self.create_output_paths()
        
        # Initialize the data
        self._init_data()
        
        # Optimize results
        self.opt_results = np.empty(shape=(self.n_spec, self.n_iterations), dtype=dict)
        self.stellar_templates = np.empty(self.n_iterations, dtype=np.ndarray)
        
        # The spectral model
        self.spectral_model = spectral_model
        self._init_spectral_model()
        self.p0cp = copy.deepcopy(self.p0)
        
        # The target dictionary
        self.target_dict = target_dict
        
        # The template augmenter
        self.augmenter = augmenter
        
        # The objective function
        self.obj = obj
        
        # The optimizer
        self.optimizer = optimizer
        
        # Init RVs
        self._init_rvs()
        
        # Print summary
        self._print_init_summary()
        
    
    #def optimize_templates(self, specrvprob, iter_index):
    #    pass
    
    #def optimize_parameters(self, specrvprob, iter_index):
        #pass

        # The number of lsf points
        #n_lsf_pts = forward_models[0].models_dict['lsf'].nx_lsf
        # breakpoint()
    
    #     # The high resolution master grid (also stellar template grid)
    #     wave_hr_master = torch.from_numpy(forward_models.templates_dict['star'][:, 0])
        
    #     # Grids to optimize
    #     if 'star' in templates_to_optimize:
    #         star_flux = torch.nn.Parameter(torch.from_numpy(forward_models.templates_dict['star'][:, 1].astype(np.float64)))
            
    #         # The current best fit stellar velocities
    #         star_vels = torch.from_numpy(np.array([forward_models[ispec].best_fit_pars[-1][forward_models[ispec].models_dict['star'].par_names[0]].value for ispec in range(forward_models.n_spec)]).astype(np.float64))
    #     else:
    #         star_flux = None
    #         star_vels = None
            
    #     if 'lab' in templates_to_optimize:
    #         residual_lab_flux = torch.nn.Parameter(torch.zeros(wave_hr_master.size()[0], dtype=torch.float64) + 1E-4)
    #     else:
    #         residual_lab_flux = None
        
    #     # The partial built forward model flux
    #     base_flux_models = torch.empty((forward_models.templates_dict['star'][:, 0].size, forward_models.n_spec), dtype=torch.float64)
        
    #     # The best fit LSF for each spec (optional)
    #     if 'lsf' in forward_models[0].models_dict and forward_models[0].models_dict['lsf'].enabled:
    #         lsfs = torch.empty((n_lsf_pts, forward_models.n_spec), dtype=torch.float64)
    #     else:
    #         lsfs = None
        
    #     # The data flux
    #     data_flux = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)
        
    #     # Bad pixel arrays for the data
    #     badpix = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)
        
    #     # The wavelength solutions
    #     waves_lr = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)
        
    #     # Weights, may just be binary mask
    #     weights = torch.empty((forward_models[0].data.flux.size, forward_models.n_spec), dtype=torch.float64)

    #     # Loop over spectra and extract to the above arrays
    #     for ispec in range(forward_models.n_spec):

    #         # Best fit pars
    #         pars = forward_models[ispec].opt_results[-1][0]

    #         if 'star' in templates_to_optimize:
    #             x, y = forward_models[ispec].build_hr_nostar(pars, iter_index)
    #         else:
    #             x, y = forward_models[ispec].build_hr(pars, iter_index)
                
    #         waves_lr[:, ispec], base_flux_models[:, ispec] = torch.from_numpy(x), torch.from_numpy(y)

    #         # Fetch lsf and flip for torch. As of now torch does not support the negative step
    #         if 'lsf' in forward_models[0].models_dict and forward_models[0].models_dict['lsf'].enabled:
    #             lsfs[:, ispec] = torch.from_numpy(forward_models[ispec].models_dict['lsf'].build(pars))
    #             lsfs[:, ispec] = torch.from_numpy(np.flip(lsfs[:, ispec].numpy(), axis=0).copy())

    #         # The data and weights, change bad vals to nan
    #         data_flux[:, ispec] = torch.from_numpy(np.copy(forward_models[ispec].data.flux))
    #         weights[:, ispec] = torch.from_numpy(np.copy(forward_models[ispec].data.mask))
    #         bad = torch.where(~torch.isfinite(data_flux[:, ispec]) | ~torch.isfinite(weights[:, ispec]) | (weights[:, ispec] <= 0))[0]
    #         if len(bad) > 0:
    #             data_flux[bad, ispec] = np.nan
    #             weights[bad, ispec] = 0

    #     # CPU or GPU
    #     if torch.cuda.is_available():
    #         torch.device('cuda')
    #     else:
    #         torch.device('cpu')

    #     # Create the Torch model object
    #     model = TemplateOptimizer(base_flux_models, waves_lr, weights, data_flux, wave_hr_master, star_flux=star_flux, star_vels=star_vels, residual_lab_flux=residual_lab_flux, lsfs=lsfs)

    #     # Create the Adam optimizer
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    #     print('Optimizing Template(s) ...', flush=True)

    #     for epoch in range(500):

    #         # Generate the model
    #         optimizer.zero_grad()
    #         loss = model.forward()

    #         # Back propagation (gradient calculation)
    #         loss.backward()
    #         optimizer.step()

    #         if (epoch + 1) % 10 == 0:
    #             print('epoch {}, loss {}'.format(epoch + 1, loss.item()), flush=True)

    #     if 'star' in templates_to_optimize:
    #         new_star_flux = model.star_flux.detach().numpy()
    #         locs = np.where(new_star_flux > 1)[0]
    #         if locs.size > 0:
    #             new_star_flux[locs] = 1
    #         forward_models.templates_dict['star'][:, 1] = new_star_flux

    #     if 'lab' in templates_to_optimize:
    #         residual_lab_flux_fit = model.residual_lab_flux.detach().numpy()
    #         forward_models.templates_dict['residual_lab'] = np.array([np.copy(forward_models.templates_dict['star'][:, 0]), residual_lab_flux_fit]).T
        
        
    # Class to optimize the forward model

        

    # def forward(self):
    
    #     # Stores all low res models
    #     models_lr = torch.empty((self.nx_data, self.n_spec), dtype=torch.float64) + np.nan
        
    #     # Loop over observations
    #     for ispec in range(self.n_spec):
            
    #         # Doppler shift the stellar wavelength grid used for this observation.
    #         if self.star_flux is not None and self.residual_lab_flux is not None:
    #             wave_hr_star_shifted = self.wave_hr_master * torch.exp(self.star_vels[ispec] / cs.c)
    #             star = self.Interp1d()(wave_hr_star_shifted, self.star_flux, self.wave_hr_master).flatten()
    #             model = self.base_flux_models[:, ispec] * star + self.residual_lab_flux
    #         elif self.star_flux is not None and self.residual_lab_flux is None:
    #             wave_hr_star_shifted = self.wave_hr_master * torch.exp(self.star_vels[ispec] / cs.c)
    #             star = self.Interp1d()(wave_hr_star_shifted, self.star_flux, self.wave_hr_master).flatten()
    #             model = self.base_flux_models[:, ispec] * star
    #         else:
    #             model = self.base_flux_models[:, ispec] + self.residual_lab_flux
                
    #         # Convolution. NOTE: PyTorch convolution is a pain in the ass
    #         # Second NOTE: Anything in PyTorch is a pain in the ass
    #         # Third NOTE: Wtf kind of language does the ML community even speak?
    #         if self.lsfs is not None:
    #             model_p = torch.ones((1, 1, self.nx_model + 2 * self.n_pad_model), dtype=torch.float64)
    #             model_p[0, 0, self.n_pad_model:-self.n_pad_model] = model
    #             conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
    #             conv.weight.data = self.lsfs[:, :, :, ispec]
    #             model = conv(model_p).flatten()

    #         # Interpolate onto data grid
    #         good = torch.where(torch.isfinite(self.weights[:, ispec]) & (self.weights[:, ispec] > 0))[0]
    #         models_lr[good, ispec] = self.Interp1d()(self.wave_hr_master, model, self.waves_lr[good, ispec])

    #     # Weighted RMS
    #     good = torch.where(torch.isfinite(self.weights) & (self.weights > 0) & torch.isfinite(models_lr))
    #     wdiffs2 = (models_lr[good] - self.data_flux[good])**2 * self.weights[good]
    #     loss = torch.sqrt(torch.sum(wdiffs2) / (torch.sum(self.weights[good])))

    #     return loss
    
    
    # @staticmethod
    # def h_poly_helper(tt):
    #     A = torch.tensor([
    #         [1, 0, -3, 2],
    #         [0, 1, -2, 1],
    #         [0, 0, 3, -2],
    #         [0, 0, -1, 1]
    #         ], dtype=tt[-1].dtype)
    #     return [sum(A[i, j]*tt[j] for j in range(4)) for i in range(4)]
        
    # @classmethod
    # def h_poly(cls, t):
    #     tt = [ None for _ in range(4) ]
    #     tt[0] = 1
    #     for i in range(1, 4):
    #         tt[i] = tt[i-1]*t
    #     return cls.h_poly_helper(tt)

    # @classmethod
    # def H_poly(cls, t):
    #     tt = [ None for _ in range(4) ]
    #     tt[0] = t
    #     for i in range(1, 4):
    #         tt[i] = tt[i-1]*t*i/(i+1)
    #     return cls.h_poly_helper(tt)

    # @classmethod
    # def interpcs(cls, x, y, xs):
    #     m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    #     m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
    #     I = np.searchsorted(x[1:], xs)
    #     dx = (x[I+1]-x[I])
    #     hh = cls.h_poly((xs-x[I])/dx)
    #     return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

    # class Interp1d(torch.autograd.Function):
    #     def __call__(self, x, y, xnew, out=None):
    #         return self.forward(x, y, xnew, out)

    #     def forward(ctx, x, y, xnew, out=None):

    #         # making the vectors at least 2D
    #         is_flat = {}
    #         require_grad = {}
    #         v = {}
    #         device = []
    #         for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
    #             assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
    #                                         'at most 2-D.'
    #             if len(vec.shape) == 1:
    #                 v[name] = vec[None, :]
    #             else:
    #                 v[name] = vec
    #             is_flat[name] = v[name].shape[0] == 1
    #             require_grad[name] = vec.requires_grad
    #             device = list(set(device + [str(vec.device)]))
    #         assert len(device) == 1, 'All parameters must be on the same device.'
    #         device = device[0]

    #         # Checking for the dimensions
    #         assert (v['x'].shape[1] == v['y'].shape[1]
    #                 and (
    #                     v['x'].shape[0] == v['y'].shape[0]
    #                     or v['x'].shape[0] == 1
    #                     or v['y'].shape[0] == 1
    #                     )
    #                 ), ("x and y must have the same number of columns, and either "
    #                     "the same number of row or one of them having only one "
    #                     "row.")

    #         reshaped_xnew = False
    #         if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
    #         and (v['xnew'].shape[0] > 1)):
    #             # if there is only one row for both x and y, there is no need to
    #             # loop over the rows of xnew because they will all have to face the
    #             # same interpolation problem. We should just stack them together to
    #             # call interp1d and put them back in place afterwards.
    #             original_xnew_shape = v['xnew'].shape
    #             v['xnew'] = v['xnew'].contiguous().view(1, -1)
    #             reshaped_xnew = True

    #         # identify the dimensions of output and check if the one provided is ok
    #         D = max(v['x'].shape[0], v['xnew'].shape[0])
    #         shape_ynew = (D, v['xnew'].shape[-1])
    #         if out is not None:
    #             if out.numel() != shape_ynew[0]*shape_ynew[1]:
    #                 # The output provided is of incorrect shape.
    #                 # Going for a new one
    #                 out = None
    #             else:
    #                 ynew = out.reshape(shape_ynew)
    #         if out is None:
    #             ynew = torch.zeros(*shape_ynew, dtype=y.dtype, device=device)

    #         # moving everything to the desired device in case it was not there
    #         # already (not handling the case things do not fit entirely, user will
    #         # do it if required.)
    #         for name in v:
    #             v[name] = v[name].to(device)

    #         # calling searchsorted on the x values.
    #         #ind = ynew
    #         #searchsorted(v['x'].contiguous(), v['xnew'].contiguous(), ind)
    #         ind = np.searchsorted(v['x'].contiguous().numpy().flatten(), v['xnew'].contiguous().numpy().flatten())
    #         ind = torch.tensor(ind)
    #         # the `-1` is because searchsorted looks for the index where the values
    #         # must be inserted to preserve order. And we want the index of the
    #         # preceeding value.
    #         ind -= 1
    #         # we clamp the index, because the number of intervals is x.shape-1,
    #         # and the left neighbour should hence be at most number of intervals
    #         # -1, i.e. number of columns in x -2
    #         ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1).long()

    #         # helper function to select stuff according to the found indices.
    #         def sel(name):
    #             if is_flat[name]:
    #                 return v[name].contiguous().view(-1)[ind]
    #             return torch.gather(v[name], 1, ind)

    #         # activating gradient storing for everything now
    #         enable_grad = False
    #         saved_inputs = []
    #         for name in ['x', 'y', 'xnew']:
    #             if require_grad[name]:
    #                 enable_grad = True
    #                 saved_inputs += [v[name]]
    #             else:
    #                 saved_inputs += [None, ]
    #         # assuming x are sorted in the dimension 1, computing the slopes for
    #         # the segments
    #         is_flat['slopes'] = is_flat['x']
    #         # now we have found the indices of the neighbors, we start building the
    #         # output. Hence, we start also activating gradient tracking
    #         with torch.enable_grad() if enable_grad else contextlib.suppress():
    #             v['slopes'] = (
    #                     (v['y'][:, 1:]-v['y'][:, :-1])
    #                     /
    #                     (v['x'][:, 1:]-v['x'][:, :-1])
    #                 )

    #             # now build the linear interpolation
    #             ynew = sel('y') + sel('slopes')*(
    #                                     v['xnew'] - sel('x'))

    #             if reshaped_xnew:
    #                 ynew = ynew.view(original_xnew_shape)

    #         ctx.save_for_backward(ynew, *saved_inputs)
    #         return ynew

    #     @staticmethod
    #     def backward(ctx, grad_out):
    #         inputs = ctx.saved_tensors[1:]
    #         gradients = torch.autograd.grad(
    #                         ctx.saved_tensors[0],
    #                         [i for i in inputs if i is not None],
    #                         grad_out, retain_graph=True)
    #         result = [None, ] * 5
    #         pos = 0
    #         for index in range(len(inputs)):
    #             if inputs[index] is not None:
    #                 result[index] = gradients[pos]
    #                 pos += 1
    #         return (*result,)
    