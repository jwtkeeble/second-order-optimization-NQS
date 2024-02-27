import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.func import vmap, grad, jacrev

from typing import List, Optional, Callable, Tuple, Dict
from utils import count_parameters, eyes_like
from copy import deepcopy

# Minres addition
from scipy.sparse.linalg import LinearOperator, minres

class SecondOrderOpt(Optimizer):
    def __init__(self,
                 model: nn.Module,

                 damping: float=1e4,
                 adapt_damping: bool=True,
                 damping_adapt_interval: int = 5,
                 damping_adapt_decay: float = (19/20),
                 min_damping: float=1e-4,
                 max_damping: float=1e20,

                 l2_reg: float=1e-5,

                 gamma_init: float=None,
                 adapt_gamma: bool = True,
                 gamma_adapt_interval: int = 5,
                 gamma_adapt_decay: float = (19/20)**(0.5),
                 min_gamma: float=1e-4,
                 max_gamma: float=1e20,

                 max_epsilon: float=0.95,

                 precondition_method: str='KFAC',
                 quadratic_model: str='QuasiHessian',
                 number_of_minres_it: int=50,

                 use_effective_gradients: bool=False,

                 use_kl_clipping: bool=False,
                 norm_constraint: float=1e-2,
                 kl_max: float=1,
                 
                 chunk_size: Optional[int]=None) -> None:
        """The constructor of the Second-Order Optimisation Class 

        :param model: The neural-network quantum state used to represent the wavefunction.
        :type model: nn.Module

        :param damping: The damping strength applied to quadratic model, which infers the size of the trust-region.
        :type float:

        :param adapt_damping: Flag to determine if the damping strength (of the trust region) should be adapted in accordance to the Levenberg-Marquardt rule.
        :type bool:

        :param damping_adapt_interval: The interval for re-evaluating the damping strength of the trust region.
        :type int:

        :param damping_adapt_decay: The decay adaption constant used to change the damping strength in accordance to the Levenberg-Marquardt.
        :type float:

        :param min_damping: The minimum damping strength of the Levenberg-Marquardt rule.
        :type float:

        :param max_damping: The maximum damping strength of the Levenberg-Marquardt rule.
        :type float:

        :param l2_reg: The l2-norm regularisation constant. 
        :type float:

        :param gamma_init: Initial damping strength applied of the factored Tikhonov regularisation (defaults to sqrt(\lambda + \eta)).
        :type Optional[float]: 

        :param adapt_gamma: Flag to determine if the damping strength (of the factored Tikhonov regularisation) should be adapted in accordance to the Greedy Adjustment rule of Marten's 2015.
        :type bool:

        :param gamma_adapt_interval: The interval for re-evaluating the damping strength of the factored Tikhonov regularisation.
        :type int:

        :param gamma_adapt_decay: The decay adaption constant used to change the damping strength (KFAC preconditioing)
        :type float:

        :param min_gamma: The minimum damping strength of the factored Tikhonov regularisation.
        :type float:

        :param max_gamma: The maximum damping strength of the factored Tikhonov regularisation.
        :type float:

        :param max_epsilon: The maximum value for the exponential moving average constant used to average out the KFAC preconditioning matrices.
        :type float:

        :param precondition_method: The preconditioning method to precondition the energy gradient.
        :type str:

        :param quadratic_model: The quadratic model used to re-scale the preconditioned gradients (This quadratic function should locally follow the curvature of the loss function).
        :type str:

        :param number_of_minres_it: The maximum number of minres iteration used to solve a linear system.
        :type int:
        """

        if not (isinstance(model, nn.Module)):
            raise TypeError(f"Invalid model type: {type(model)}")        

        self.model = model
        self.per_sample_model = deepcopy(model)  # per-sample = no hook version
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.chunk_size = chunk_size

        # TODO: refactorize error function out
        if (damping < 0):
            raise ValueError(f"Invalid damping value: {damping}")
        if not (type(adapt_damping) is bool):
            raise TypeError(f"Invalid adapt_damping flag: {adapt_damping}")
        if (damping_adapt_interval < 0 or type(damping_adapt_interval) is not int):
            raise ValueError(f"Invalid damping_adapt_interval value: {damping_adapt_interval}")
        if (damping_adapt_decay < 0):
            raise ValueError(f"Invalid damping_adapt_decay value: {damping_adapt_decay}")
        if (min_damping < 0):
            raise ValueError(f"Invalid min_damping value: {min_damping}")
        if (max_damping < 0):
            raise ValueError(f"Invalid max_damping value: {max_damping}")
        if (l2_reg < 0):
            raise ValueError(f"Invalid l2_reg value: {l2_reg}")
        if not (type(adapt_gamma) is bool):
            raise ValueError(f"Invalid adapt_gamma value: {adapt_gamma}")
        if (gamma_adapt_interval < 0 or type(gamma_adapt_interval) is not int):
            raise ValueError(f"Invalid gamma_adapt_interval value: {gamma_adapt_interval}")
        if (gamma_adapt_decay < 0):
            raise ValueError(f"Invalid gamma_adapt_decay value: {gamma_adapt_decay}")
        if (min_gamma < 0):
            raise ValueError(f"Invalid min_gamma value: {min_gamma}")
        if (max_gamma < 0):
            raise ValueError(f"Invalid max_gamma value: {max_gamma}")
        if (max_epsilon < 0 or max_epsilon > 1):
            raise ValueError(f"Invalid max_epsilon value: {max_epsilon}")
        if (type(precondition_method) is not str):
            raise TypeError(f"Invalid precondition_method: {type(precondition_method)}")
        if (type(quadratic_model) is not str):
            raise TypeError(f"Invalid quadratic_model: {type(quadratic_model)} ")
        if (type(use_effective_gradients) is not bool):
            raise TypeError(f"Invalid use_effective_gradients: {type(use_effective_gradients)} ")
        if (type(use_kl_clipping) is not bool):
            raise TypeError(f"Invalid use_kl_clipping type: {type(use_kl_clipping)}")
        if (type(norm_constraint) is not float):
            raise TypeError(f"Invalid norm_constraint type: {type(norm_constraint)}")
        if (type(kl_max) is not float):
            raise TypeError(f"Invalid kl_max type: {type(kl_max)}")
        
        self._accepted_modules = [nn.Linear] #['Linear']        # List of `nn.Module` objects that we can precondition in general.
        
        self._accepted_quadratic_models = ['QuasiHessian','Fisher','VMC'] #Accepted Quadratic-Models: 

        # Functionalize a copy here of self.model
        def calc_logabs(params, x):
            _, logabs = torch.func.functional_call(self.per_sample_model, params, x)
            return logabs

        def dlogabs_dx(params, x):
            grad_logabs = jacrev(calc_logabs, argnums=1, has_aux=False)(params, x)
            return grad_logabs, grad_logabs

        def laplacian_psi(params, x):
            grad2_logabs, grad_logabs = jacrev(dlogabs_dx, argnums=1, has_aux=True)(params, x)
            return grad2_logabs.diagonal(0,-2,-1).sum() + grad_logabs.pow(2).sum()
        
        self.kinetic_from_log_fn = lambda params, x: -0.5 * laplacian_psi(params, x)
        self.calc_logabs = calc_logabs
        if (gamma_init is None):
            gamma_init = (damping + l2_reg)**(0.5)
        # Constructing base class Optimizer (defaults to all groups within param groups)
        defaults = dict(damping=damping,
                        adapt_damping=adapt_damping,
                        damping_adapt_interval=damping_adapt_interval,
                        damping_adapt_decay=damping_adapt_decay,
                        min_damping=min_damping,
                        max_damping=max_damping,
                        l2_reg=l2_reg,
                        gamma=gamma_init,
                        adapt_gamma=adapt_gamma,
                        gamma_adapt_interval=gamma_adapt_interval,
                        gamma_adapt_decay=gamma_adapt_decay,
                        min_gamma=min_gamma,
                        max_gamma=max_gamma,
                        max_epsilon=max_epsilon,
                        epsilon=0,
                        rho=0,
                        step=0,
                        precondition_method=precondition_method,
                        quadratic_model=quadratic_model,
                        number_of_minres_it=number_of_minres_it,
                        use_effective_gradients=use_effective_gradients,
                        use_kl_clipping=use_kl_clipping,
                        norm_constraint=norm_constraint,
                        kl_max=kl_max)
        super(SecondOrderOpt, self).__init__(self._prepare_model(), defaults)

    def _prepare_model(self) -> List[dict]:  # Set hooks and return list of param_groups (each as a dictionary)
        print("\nPreparing network for KFAC...")
        _param_groups = []
        total_params = count_parameters(self.model)
        for module in self.model.modules():
            module_name = module.__class__.__name__
            if(type(module) in self._accepted_modules):
                module.register_forward_pre_hook(self.forward_pre_hook)              # forward pre-hooks 
                module.register_full_backward_pre_hook(self.full_backward_pre_hook)  # backward pre-hooks
                params = list(module.parameters())
                if(isinstance(module, nn.Linear)):
                    pi = torch.as_tensor(1, dtype=self.dtype, device=self.device) #TODO: replace device=self.device & dtype=self.dtype with a factory_kwargs dict
                else:
                    raise NameError(f"Unknown module in 'accepted' module list, but no method defined.")
            else:
                continue #if not acceptable module, silently skip
            nparam_layer = count_parameters(module)
            nparam_layer_percent = (nparam_layer/total_params)*100
            print(f"> Adding module: {module_name:20s} | nparams: {nparam_layer:6d} ({nparam_layer_percent:4.2f}%)")
            _param_groups.append({'params':params, 'module':module, 'pi':pi}) #per-layer approximation here!
        print(f"Total number of parameters: {total_params:27d}")
        return _param_groups

    #=========================================================================================================================#
    #                                        FORWARD-PRE AND FULL-BACKWARD HOOKS                                              #
    #=========================================================================================================================#

    def forward_pre_hook(self, module: nn.Module, input: List[Tensor]) -> None:
        a = input[0]
        if(module.bias is not None):
            bias_shape = [*a.shape]
            bias_shape[-1] = 1
            bias_input_vector = torch.ones(*bias_shape, device=a.device, dtype=a.dtype)
            a = torch.cat([a, bias_input_vector], dim=-1)
        self.state[module]['a'] = a

    def full_backward_pre_hook(self, module: nn.Module, grad_output: List[Tensor]) -> None:
        e = grad_output[0] * grad_output[0].size(0) #NOTE: Needed to get correct hooks (otherwise it's divided by batch-size!)
        self.state[module]['e'] = e

    #=========================================================================================================================#
    #                                        MERGE/SPLIT WEIGHT-BIAS FOR LAYERS                                               #
    #=========================================================================================================================#

    def _merge_for_group(self, module: nn.Module, input: List[Tensor]) -> Tensor:
        if(isinstance(module, nn.Linear)):
            if(module.bias is not None):
                return torch.cat([input[0], input[1].unsqueeze(-1)], dim=-1).contiguous()
            else:
                return input[0].contiguous()
        else:
            raise TypeError(f"This module type isn't supported yet: {module}.")

    def _split_for_group(self, module: nn.Module, g: Tensor) -> List[Optional[Tensor]]:
        if(isinstance(module, nn.Linear)):
            if(module.bias is not None):
                gb = g[:,-1].contiguous().view(*module.bias.shape).clone()
                g = g[:,:-1]
            else:
                gb = None
                g = g.contiguous().view(*module.weight.shape).clone()
            return g, gb
        else:
            raise TypeError(f"This module type isn't supported yet: {module}")

    def _get_raw_gradients_for_group(self, group: dict) -> List[Optional[Tensor]]:
       module = group['module']
       grad_weights = group['params'][0].grad.clone()
       if(module.bias is not None):
           grad_biases = group['params'][1].grad.clone()
       else:
           grad_biases = None 
       return grad_weights, grad_biases

    def get_all_gradients(self) -> Dict[str, List[Tensor]]:
        gradients = {}
        for group in self.param_groups:
            module = group['module']
            gradients[module] = self._get_raw_gradients_for_group(group=group)
        return gradients

    def set_all_gradients(self, gradients: Dict[str, List[Tensor]], scale: Optional[float]=None) -> None:
        for group in self.param_groups:
            module = group['module']
            grad = gradients[module]
            group['params'][0].grad.copy_(grad[0])
            if(module.bias is not None):
                group['params'][1].grad.copy_(grad[1])
            if(scale is not None):
                group['params'][0].grad.mul_(scale)
                if(module.bias is not None):
                    group['params'][1].grad.mul_(scale)

    def _get_all_momentum_buffers(self):  
        momentum_buffers = {}
        for group in self.param_groups:
            module = group['module']
            momentum_buffer_list = []
            for p in group['params']:
                param_state = self.state[p]
                if("momentum_buffer" not in param_state):
                    param_state['momentum_buffer'] = torch.zeros_like(p)
                    delta0 = param_state['momentum_buffer']  # here delta_0 is list[Tensor] (w or b)
                else:
                    delta0 = param_state['momentum_buffer']
                momentum_buffer_list.append(delta0)
            momentum_buffers[module] = momentum_buffer_list
        return momentum_buffers

    #=========================================================================================================================#
    #                      RESHAPE PER-SAMPLE GRADIENTS FROM VMAP TO DICT (WITH GROUP AS KEY) AND CACHE                       #
    #=========================================================================================================================#

    def _update_per_sample_gradients_cache(self, params: Dict[str, Tensor], x: Tensor) -> None:
        if(self.param_groups[0]['quadratic_model'] == 'VMC' and self.param_groups[0]['precondition_method'] in ['KFAC']):
            logpsi_grad = vmap(grad(self.calc_logabs, argnums=(0), has_aux=False), in_dims=(None, 0), chunk_size=self.chunk_size)(params, x)
            self.per_sample_logpsi_grads = self._reshape_vmap_params_to_dict(param_dict=logpsi_grad)                   # auto-converts dict (param) to dict (layers)
            logpsi_grad_params_x = vmap(jacrev(jacrev(self.calc_logabs, argnums=(1)), argnums=(0)), in_dims=(None, 0), chunk_size=self.chunk_size)(params, x)
            self.per_sample_logpsi_grad_params_x = self._reshape_vmap_params_to_dict(param_dict=logpsi_grad_params_x)  # auto-converts dict (param) to dict (layers) 
            self.per_sample_elocal_grads = None
        else:
            logpsi_grad = vmap(grad(self.calc_logabs, argnums=(0), has_aux=False), in_dims=(None, 0), chunk_size=self.chunk_size)(params, x)
            self.per_sample_logpsi_grads = self._reshape_vmap_params_to_dict(param_dict=logpsi_grad)  # auto-converts dict (param) to dict (layers)
            if(self.param_groups[0]['quadratic_model'] in self._accepted_quadratic_models):
                if(self.param_groups[0]['quadratic_model'] == 'QuasiHessian'): #only call if using QuasiHessian
                    elocal_grad = vmap(grad(self.kinetic_from_log_fn, argnums=(0), has_aux=False), in_dims=(None, 0), chunk_size=self.chunk_size)(params, x)
                    self.per_sample_elocal_grads = self._reshape_vmap_params_to_dict(param_dict=elocal_grad)  # auto-converts dict (param) to dict (layers)
                else:
                    self.per_sample_elocal_grads = None 
                logpsi_grad_params_x = vmap(jacrev(jacrev(self.calc_logabs, argnums=(1)), argnums=(0)), in_dims=(None, 0), chunk_size=self.chunk_size)(params, x)
                self.per_sample_logpsi_grad_params_x = self._reshape_vmap_params_to_dict(param_dict=logpsi_grad_params_x)  # auto-converts dict (param) to dict (layers)

    def _reshape_vmap_params_to_dict(self, param_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Takes the dictionary of parameters (from torch.func) and re-wraps it to a per-layer dictionary.
        This function works for this network, but may have issues with a 'general'  NQS, a general method is WIP.
        """
        output_list = list(param_dict.values())
        params_from_vmap = {}
        start_idx = 0
        idx = 0
        for group in self.param_groups:
            module = group['module']
            idx += 1
            if(module.bias is not None):
                idx += 1
            params_from_vmap[module] = output_list[start_idx:idx]
            start_idx = idx
        return params_from_vmap

    #=========================================================================================================================#
    #                               EXPONENTIAL MOVING AVERAGE OF THE KRONECKER FACTORS                                       #
    #=========================================================================================================================#
    
    def _update_epsilon(self) -> None: #TODO change name max_epsilon
        self.param_groups[0]['epsilon'] = min(1. - 1. / (self.param_groups[0]['step'] + 1), self.param_groups[0]['max_epsilon'])

    #=========================================================================================================================#
    #                              KRONECKER FACTORS AND FACTORED TIKHONOV REGULARISATION                                     #
    #=========================================================================================================================#

    def _update_all_kronecker_factors(self, elocal: Tensor, energy_mean: Tensor, mode: str='KFAC', inkp_gamma=None) -> None:  # TODO: rename KF in KFAC
        if(mode=='KFAC'):
            for group in self.param_groups:
                module = group['module']
                a = self.state[module]['a']
                e = self.state[module]['e']
                a = a - torch.mean(a, dim=0, keepdim=True)
                e = e - torch.mean(e, dim=0, keepdim=True)
                left_factor = torch.mean(a.transpose(-2,-1) @ a, dim=0)  #NOTE: PyTorch Linear is defined as x.A^T + b, TF/JAX's Linear (aka Dense) is defined as x.A + b (so KFAC terms differ by transpose)
                right_factor = torch.mean(e.transpose(-2,-1) @ e, dim=0)
                if(self.param_groups[0]['step'] == 0):
                    self.state[module]['left_factor'] = left_factor
                    self.state[module]['right_factor'] = right_factor
                else:
                    self.state[module]['left_factor'] = self.param_groups[0]['epsilon'] * self.state[module]['left_factor'] + (1 - self.param_groups[0]['epsilon']) * left_factor
                    self.state[module]['right_factor'] = self.param_groups[0]['epsilon'] * self.state[module]['right_factor'] + (1 - self.param_groups[0]['epsilon']) * right_factor
        elif (mode=='Fisher'):
            pass
        elif (mode=='VMC'):
            pass
        else:
            raise NameError(f"Unknown mode: {mode} chosen")

    def _regularise_all_kronecker_factors(self, gamma: float) -> None:
        if self.param_groups[0]['precondition_method'] in ["KFAC"]:
            for group in self.param_groups:
                module = group['module']
                pi_num = torch.linalg.matrix_norm(self.state[module]['left_factor'], ord='nuc') * self.state[module]['right_factor'].shape[-1]
                pi_den = torch.linalg.matrix_norm(self.state[module]['right_factor'], ord='nuc') * self.state[module]['left_factor'].shape[-1]
                group['pi'] = (pi_num/pi_den)**(0.5)
                if(isinstance(group['module'],nn.Linear)):
                    diag_left_factor = eyes_like(self.state[module]['left_factor']) * (gamma * group['pi'])
                    diag_right_factor = eyes_like(self.state[module]['right_factor']) * (gamma / group['pi'])
                else:
                    raise NameError(f"Unexpected module name found: {module.__class__.__name__} in _regularise_all_kronecker_factors method")
                self.state[module]['left_factor_reg'] = self.state[module]['left_factor'] + diag_left_factor 
                self.state[module]['right_factor_reg'] = self.state[module]['right_factor'] + diag_right_factor
        else:
            pass
    
    def correct_Delta(self, module: nn.Linear, target: Tensor, initial_guess: Tensor, number_of_iterations: int):
        if self.param_groups[0]['quadratic_model'] == 'VMC' or (self.param_groups[0]['precondition_method'] == 'VMC' and self.param_groups[0]['quadratic_model'] == 'QuasiHessian'):
            # VMC metric block-diagonal calc
            grad_logpsi_params1_xi = self._merge_for_group(module, self.per_sample_logpsi_grad_params_x[module]) #shape [batch_size, num_particles, Hout, Hin]
            batch, num_particles, Hout, Hin = grad_logpsi_params1_xi.shape
            grad_logpsi_params1_xi = grad_logpsi_params1_xi.transpose(-2,-1).reshape(batch, num_particles, -1) #shape: [batch_size, num_particles, Hout*Hin]
            block_diagonal = torch.einsum("bin,bim->nm", grad_logpsi_params1_xi, grad_logpsi_params1_xi) / batch
        elif self.param_groups[0]['quadratic_model'] in ['Fisher', 'QuasiHessian']:
            # Fisher matrix block-diagonal calc
            logpsi_grad = self._merge_for_group(module=module, input=self.per_sample_logpsi_grads[module])
            logpsi_grad = logpsi_grad - logpsi_grad.mean(dim=0)
            batch, Hout, Hin = logpsi_grad.shape
            block_diagonal = 4. * torch.einsum("bi,bj->ij", logpsi_grad.transpose(-2,-1).reshape(batch, -1), logpsi_grad.transpose(-2,-1).reshape(batch, -1)) / batch
        else:
            raise NameError(f"Unknown quadratic model ({self.param_groups[0]['quadratic_model']}) precondition method ({self.param_groups[0]['precondition_method']}) configuration selected: ")
        # ==============================
        # Quick preconditioner using diagonal part
        zeta_reg = 0.75 #SHOULD BE 0.75 according to Hessian-Free recommendation of Martens!! Used to be 0.95 here
        diag_block_diagonal = block_diagonal.diagonal(offset=0, dim1=-2, dim2=-1)
        kappa_reg = 1e-2
        identity_min = torch.as_tensor(-min(torch.min(diag_block_diagonal).item(), 0.) * torch.ones(diag_block_diagonal.shape), device=self.device, dtype=self.dtype)
        identity_reg = torch.ones(diag_block_diagonal.shape, device=diag_block_diagonal.device, dtype=diag_block_diagonal.dtype)
        preconditioner = (diag_block_diagonal + identity_min).contiguous()
        preconditioner.add_(kappa_reg * identity_reg) # Summed in two steps to avoid underflow to 0 (leading to inf when inverted)
        preconditioner.pow_(-zeta_reg)
        # ==============================
        dtype = str(torch.get_default_dtype()).split('.')[-1]
        n = Hout * Hin 
        y_target = target.transpose(-2,-1).reshape(-1).detach().cpu().numpy()
        x_init = initial_guess.transpose(-2,-1).reshape(-1).detach().cpu().numpy()
        #We pass our tensors to SciPy (note default Scipy dtype is double) before casting back to PyTorch
        sparse_linear_operator = LinearOperator(shape=(n, n),
                                                dtype= dtype,
                                                matvec=(lambda x: (block_diagonal @ torch.as_tensor(x, device=self.device, dtype=self.dtype)).cpu().detach().numpy()),
                                                rmatvec=(lambda x: (block_diagonal.transpose(-2,-1) @ torch.as_tensor(x, device=self.device, dtype=self.dtype)).cpu().detach().numpy()))
        sparse_preconditioner = LinearOperator(shape=(n, n),
                                                dtype= dtype,
                                                matvec=(lambda x: (preconditioner * torch.as_tensor(x, device=self.device, dtype=self.dtype)).cpu().detach().numpy()),
                                                rmatvec=(lambda x: (preconditioner * torch.as_tensor(x, device=self.device, dtype=self.dtype)).cpu().detach().numpy()))
        corrected_Delta = minres(A=sparse_linear_operator, b=y_target, x0=x_init, shift=-self.param_groups[0]['damping'], M=sparse_preconditioner, maxiter=number_of_iterations, tol=1e-8)[0]
        corrected_Delta_tensor = torch.as_tensor(corrected_Delta, device=self.device, dtype=self.dtype).reshape(Hin, Hout).transpose(-2,-1)
        return corrected_Delta_tensor
        
    def _get_all_unscaled_natural_gradients(self) -> Dict[str, Tensor]:
        updates = {}
        for group in self.param_groups:
            module = group['module']
            updates[module] = self._get_all_unscaled_natural_gradient_for_group(group)
        return updates

    def _get_all_unscaled_natural_gradient_for_group(self, group) -> List[Tensor]:  # was split before!
        module = group['module']
        grad = self._merge_for_group(module=module, input=self._get_raw_gradients_for_group(group))
        if self.param_groups[0]['precondition_method'] in ['KFAC']:
            left = self.state[module]['left_factor_reg']  # left regularised KF
            right = self.state[module]['right_factor_reg']  # right regularised KF
            # https://discuss.pytorch.org/t/17774/27
            Delta = torch.linalg.solve(left, torch.linalg.solve(right, grad).transpose(-2,-1)).transpose(-2,-1)  # (L^-1 \otimes R^-1).vec(V) = vec(R^-1.V.L^-T)
        if group['number_of_minres_it'] > 0 and self.param_groups[0]['precondition_method'] in ['KFAC'] and self.param_groups[0]['quadratic_model'] in ['VMC', 'Fisher', 'QuasiHessian']:
            # Modify with Minres algo using Delta as initial guess
            number_of_minres_it = group['number_of_minres_it']
            init_guess = Delta
            corrected_Delta = self.correct_Delta(module=module, target=grad, initial_guess=init_guess, number_of_iterations=number_of_minres_it)
            return self._split_for_group(module=module, g=corrected_Delta)
        elif group['number_of_minres_it'] > 0 and self.param_groups[0]['precondition_method'] in ['VMC', 'Fisher'] and self.param_groups[0]['quadratic_model'] in ['VMC', 'Fisher', 'QuasiHessian']:
            # Modify with Minres algo using previous_delta or grad as initial guess
            number_of_minres_it = group['number_of_minres_it']
            scale_init_guess = 0.95
            delta_last = self._merge_for_group(module=module, input=self._get_all_momentum_buffers()[module])
            init_guess = scale_init_guess * delta_last
            # init_guess = grad
            corrected_Delta = self.correct_Delta(module=module, target=grad, initial_guess=init_guess, number_of_iterations=number_of_minres_it)
            return self._split_for_group(module=module, g=corrected_Delta)
        else:
            return self._split_for_group(module=module, g=Delta)

    #=========================================================================================================================#
    #                            RESCALING TO QUADRATIC MODEL WITH OPTIMAL ALPHA/MU PAIR                                      #
    #=========================================================================================================================#
    def _rescale_natural_gradients(self, raw_gradients: Dict[str, List[Tensor]], Delta: Dict[str, List[Tensor]], loss: Tensor) -> Tuple[Dict[str, List[Tensor]], Tensor, Tensor, Tensor]:
        alpha, mu, Mdelta = self._get_optimal_alpha_mu(raw_gradients=raw_gradients,
                                                       unscaled_natural_gradients=Delta,
                                                       loss=loss)
        delta0 = self._get_all_momentum_buffers() 
        rescaled_natural_gradients = {}
        for module in Delta:
            rescaled_natural_gradients[module] = [ alpha * Delta + mu * delta0 for Delta, delta0 in zip(Delta[module], delta0[module]) ]
        return rescaled_natural_gradients, alpha, mu, Mdelta
 
    def _get_optimal_alpha_mu(self, raw_gradients: Dict[str, List[Tensor]], unscaled_natural_gradients: Dict[str, List[Tensor]], loss: Tensor) -> Tuple[Tensor, Tensor, Tensor]: 
        delta0 = self._get_all_momentum_buffers()
        #QUADRATIC COMPONENTS
        DeltaT_M_Delta = self._get_vector_matrix_vector_product(logpsi_grads=self.per_sample_logpsi_grads, elocal_grads=self.per_sample_elocal_grads,
                                                                vector1=unscaled_natural_gradients, vector2=unscaled_natural_gradients,
                                                                elocal=self._elocal_cache,
                                                                quadratic_model=self.param_groups[0]['quadratic_model'])
        DeltaT_M_delta0 = self._get_vector_matrix_vector_product(logpsi_grads=self.per_sample_logpsi_grads, elocal_grads=self.per_sample_elocal_grads,
                                                                 vector1=unscaled_natural_gradients, vector2=delta0,
                                                                 elocal=self._elocal_cache,
                                                                 quadratic_model=self.param_groups[0]['quadratic_model'])
        delta0T_M_Delta = DeltaT_M_delta0
        delta0T_M_delta0 = self._get_vector_matrix_vector_product(logpsi_grads=self.per_sample_logpsi_grads, elocal_grads=self.per_sample_elocal_grads,
                                                                 vector1=delta0, vector2=delta0,
                                                                 elocal=self._elocal_cache,
                                                                 quadratic_model=self.param_groups[0]['quadratic_model'])
        #LINEAR COMPONENTS
        gradT_Delta = self._get_scalar_product(vector1=raw_gradients,
                                               vector2=unscaled_natural_gradients)
        gradT_delta0 = self._get_scalar_product(vector1=raw_gradients,
                                                vector2=delta0)
        #NORM COMPONENTS
        DeltaT_Delta = self._get_scalar_product(vector1=unscaled_natural_gradients,
                                                vector2=unscaled_natural_gradients)
        DeltaT_delta0 = self._get_scalar_product(vector1=unscaled_natural_gradients,
                                                 vector2=delta0)
        delta0T_Delta = DeltaT_delta0
        delta0T_delta0 = self._get_scalar_product(vector1=delta0,
                                                  vector2=delta0)

        lambda_plus_eta = (self.param_groups[0]['damping'] + self.param_groups[0]['l2_reg'])
        m00 = DeltaT_M_Delta + lambda_plus_eta * DeltaT_Delta
        m01 = DeltaT_M_delta0 + lambda_plus_eta * DeltaT_delta0
        m10 = delta0T_M_Delta + lambda_plus_eta * delta0T_Delta
        m11 = delta0T_M_delta0 + lambda_plus_eta * delta0T_delta0
        v0 = gradT_Delta
        v1 = gradT_delta0
        
        M = torch.tensor([[m00,m01],[m10,m11]], device=self.device, dtype=self.dtype)  #TODO maybe avoid using M for too many different things
        V = torch.tensor([[v0],[v1]], device=self.device, dtype=self.dtype)
        """
        alpha_opt = torch.tensor([-1. * v0 / m00],  device=self.device, dtype=self.dtype)
        mu_opt = torch.tensor([0],  device=self.device, dtype=self.dtype)
        """
        if(self.param_groups[0]['step'] == 0): #TODO check if need to re-wrap, Martens had analytical formula here?
            alpha_opt = -v0/m00
            mu_opt = torch.zeros_like(v0)
            #alpha_opt = torch.tensor([-1. * v0 / m00],  device=self.device, dtype=self.dtype)
            #mu_opt = torch.tensor([0],  device=self.device, dtype=self.dtype) 
        else:
            alpha_opt, mu_opt = -1 * torch.linalg.solve(M, V)
            #det = m00*m11 - m01*m10                        #analytical formula here (leads to diff. convergence)
            #alpha_opt = -1*(m11*v0 - m01*v1)/det
            #mu_opt = -1*(-m10*v0 + m00*v1)/det
        # Compute REGULARISED Quadratic Model score for free!
        Mdelta = 0.5*((alpha_opt**2 * DeltaT_M_Delta) + (alpha_opt * mu_opt * DeltaT_M_delta0) + (mu_opt * alpha_opt * delta0T_M_Delta) + (mu_opt**2 * delta0T_M_delta0)) + \
                  alpha_opt * gradT_Delta + mu_opt * gradT_delta0 + loss + \
                  0.5*lambda_plus_eta * (alpha_opt**2 * DeltaT_Delta + alpha_opt * mu_opt * DeltaT_delta0 + mu_opt * alpha_opt * delta0T_Delta + mu_opt**2 * delta0T_delta0)
        return alpha_opt, mu_opt, Mdelta


    def _compute_unregularised_qmodel(self, raw_gradients: Dict[str, List[Tensor]], delta: Dict[str, List[Tensor]], loss: Tensor) -> Tensor: 
        #UNREGULARISED QUADRATIC MODEL CALC
        #QUADRATIC COMPONENTS
        deltaT_M_delta = self._get_vector_matrix_vector_product(logpsi_grads=self.per_sample_logpsi_grads, elocal_grads=self.per_sample_elocal_grads,
                                                                vector1=delta, vector2=delta,
                                                                elocal=self._elocal_cache,
                                                                quadratic_model=self.param_groups[0]['quadratic_model'])
        #LINEAR COMPONENTS
        gradT_delta = self._get_scalar_product(vector1=raw_gradients,
                                               vector2=delta)
        Mdelta = 0.5 * deltaT_M_delta + gradT_delta + loss
        return Mdelta  

    #=========================================================================================================================#
    #                     VECTOR-MATRIX-VECTOR AND DOT-PRODUCT METHODS EXPLOITING ALGEBRAIC ASSOCIATIVITY                     #
    #=========================================================================================================================#

    def _get_vector_matrix_vector_product(self, logpsi_grads: Dict[str, List[Tensor]], elocal_grads: Optional[Dict[str, List[Tensor]]],
                                                vector1: Dict[str, List[Tensor]], vector2: Dict[str, List[Tensor]], elocal: Tensor,
                                                quadratic_model: str):
        if(quadratic_model == 'QuasiHessian'):
            batch = logpsi_grads[list(logpsi_grads.keys())[0]][0].shape[0]
            _v1_grad_logpsi_mc_sum = torch.zeros(batch, device=self.device, dtype=self.dtype) #from term B
            _grad_logpsi_mc_v2_sum = torch.zeros(batch, device=self.device, dtype=self.dtype)
            _v1_grad_elocal_mc_sum = torch.zeros(batch, device=self.device, dtype=self.dtype)
            _grad_elocal_mc_v2_sum = torch.zeros(batch, device=self.device, dtype=self.dtype)

            for logpsi_key, elocal_key, vec1_key, vec2_key in zip(logpsi_grads, elocal_grads, vector1, vector2):
                logpsi_grad = self._merge_for_group(module=logpsi_key, input=logpsi_grads[logpsi_key])
                elocal_grad = self._merge_for_group(module=elocal_key, input=elocal_grads[elocal_key])
                v1 = self._merge_for_group(module=vec1_key, input=vector1[vec1_key])
                v2 = self._merge_for_group(module=vec2_key, input=vector2[vec2_key])
                #Flatten
                v1 = v1.reshape(-1) 
                v2 = v2.reshape(-1)
                logpsi_grad_mc = (logpsi_grad - logpsi_grad.mean(dim=0, keepdim=True)).reshape(batch, -1)
                elocal_grad_mc = (elocal_grad - elocal_grad.mean(dim=0, keepdim=True)).reshape(batch, -1)
                _v1_grad_logpsi_mc_sum += torch.sum(v1 * logpsi_grad_mc, dim=-1)
                _grad_logpsi_mc_v2_sum += torch.sum(logpsi_grad_mc * v2, dim=-1)
                _v1_grad_elocal_mc_sum += torch.sum(v1 * elocal_grad_mc, dim=-1)
                _grad_elocal_mc_v2_sum += torch.sum(elocal_grad_mc * v2, dim=-1)
            
            quasi_hessian_B = 4. * torch.mean((elocal - elocal.mean(dim=0)) * _v1_grad_logpsi_mc_sum * _grad_logpsi_mc_v2_sum)
            quasi_hessian_D = 2. * torch.mean(_v1_grad_elocal_mc_sum * _grad_logpsi_mc_v2_sum)
            quasi_hessian_D += 2. * torch.mean(_v1_grad_logpsi_mc_sum * _grad_elocal_mc_v2_sum)
            quasi_hessian_D /= 2.  # Symmetric estimator
            quasi_hessian = quasi_hessian_B + quasi_hessian_D # UF estimator?
            return quasi_hessian
        
        elif(quadratic_model == 'Fisher'):
            batch = logpsi_grads[list(logpsi_grads.keys())[0]][0].shape[0]
            _v1c_sum = torch.zeros(batch, dtype=self.dtype, device=self.device)
            _cv2_sum = torch.zeros(batch, dtype=self.dtype, device=self.device)

            for logpsi_key, vec1_key, vec2_key in zip(logpsi_grads, vector1, vector2):
                c = self._merge_for_group(logpsi_key, logpsi_grads[logpsi_key]) #Hx3
                v1 = self._merge_for_group(vec1_key, vector1[vec1_key])  #Hx3
                v2 = self._merge_for_group(vec2_key, vector2[vec2_key])  #Hx3
                #flatten vectors excluding the batch dim...
                _v1c_sum += torch.sum(v1.reshape(-1) * c.reshape(batch, -1), dim=-1)    
                _cv2_sum += torch.sum(v2.reshape(-1) * c.reshape(batch, -1), dim=-1)

            _fisher_norm = torch.mean(_v1c_sum * _cv2_sum, dim=0)
            return _fisher_norm
        
        elif(quadratic_model == 'VMC'):
            batch = self.per_sample_logpsi_grad_params_x[list(self.per_sample_logpsi_grad_params_x.keys())[0]][0].shape[0]
            num_particles = self.per_sample_logpsi_grad_params_x[list(self.per_sample_logpsi_grad_params_x.keys())[0]][0].shape[1]

            _v1_grads_logpsi_params_x_sum = torch.zeros((batch, num_particles), dtype=self.dtype, device=self.device)
            _grads_logpsi_params_x_v2_sum = torch.zeros((batch, num_particles), dtype=self.dtype, device=self.device)

            for logpsi_params_x_key, vec1_key, vec2_key in zip(self.per_sample_logpsi_grad_params_x, vector1, vector2):
                grad_logpsi_params1_xi = self._merge_for_group(logpsi_params_x_key, self.per_sample_logpsi_grad_params_x[logpsi_params_x_key]) #shape [batch_size, num_particles, Hout, Hin]
                grad_logpsi_params1_xi_flatten = grad_logpsi_params1_xi.reshape(batch, num_particles, -1)
                v1 = self._merge_for_group(vec1_key, vector1[vec1_key])
                v2 = self._merge_for_group(vec2_key, vector2[vec2_key])
                _v1_grads_logpsi_params_x_sum += torch.sum(v1.reshape(-1) * grad_logpsi_params1_xi_flatten, dim=-1)
                _grads_logpsi_params_x_v2_sum += torch.sum(grad_logpsi_params1_xi_flatten * v2.reshape(-1), dim=-1)

            _vmc_norm = torch.mean(torch.sum(_v1_grads_logpsi_params_x_sum * _grads_logpsi_params_x_v2_sum, dim=1), dim=0)
            return _vmc_norm

        else:
            raise NameError(f"Unknown Quadratic Model ({quadratic_model}) selected!")

    def _get_scalar_product(self, vector1: dict, vector2: dict) -> Tensor:
        _v1v2_sum = torch.as_tensor(0., dtype=self.dtype, device=self.device)
        for vec1_key, vec2_key in zip(vector1, vector2):
            v1 = self._merge_for_group(module=vec1_key, input=vector1[vec1_key])
            v2 = self._merge_for_group(module=vec2_key, input=vector2[vec2_key])
            _v1v2_sum += torch.sum(v1 * v2)
        return _v1v2_sum

    #=========================================================================================================================#
    #                                       UPDATE DAMPING AND CALCULATE REDUCTION RATIO                                      #
    #=========================================================================================================================#

    def _update_damping(self, loss_theta: Tensor, loss_std: Tensor, loss_theta_plus_delta: Tensor, raw_gradients: Dict[str, List[Tensor]], delta: Dict[str, List[Tensor]]) -> None:
        Mdelta = self._compute_unregularised_qmodel(raw_gradients=raw_gradients,
                                                    delta=delta,
                                                    loss=loss_theta)
        self.param_groups[0]['log_rho'] = self._calculate_log_rho_new(loss_theta=loss_theta,
                                                                      loss_std=loss_std,
                                                                      loss_theta_plus_delta=loss_theta_plus_delta,
                                                                      delta=delta)
        self.param_groups[0]['qmodel_change'] = Mdelta - loss_theta
        self.param_groups[0]['loss_change'] = loss_theta_plus_delta - loss_theta
        if(self.param_groups[0]['log_rho'] > -0.2876820724517809): #log(0.75)):
            self.param_groups[0]['damping'] *= (self.param_groups[0]['damping_adapt_decay']**(self.param_groups[0]['damping_adapt_interval']))
        elif(self.param_groups[0]['log_rho'] < -1.3862943611198906): #log(0.25)):
            self.param_groups[0]['damping'] /= (self.param_groups[0]['damping_adapt_decay']**(self.param_groups[0]['damping_adapt_interval']))
        else:
            self.param_groups[0]['damping'] = self.param_groups[0]['damping'] 

    def _calculate_log_rho_new(self, loss_theta, loss_std, loss_theta_plus_delta, delta):  #TODO: re-check carefully this function
        """
        Levenberg-Marquardt Adjustment rule: Overflow/Underflow protected version (More' 1978?)
        """
        if(loss_theta_plus_delta > loss_theta + 0.1 * torch.abs(loss_theta)):
            return -torch.inf
        elif(loss_theta < loss_theta_plus_delta and loss_theta_plus_delta < loss_theta + 0.1 * torch.abs(loss_theta)):
            return 0. # default to reducing lambda
        else:
            log_rho_num = torch.log(loss_theta - loss_theta_plus_delta)
            deltaT_M_delta = self._get_vector_matrix_vector_product(logpsi_grads=self.per_sample_logpsi_grads,
                                                                    elocal_grads=self.per_sample_elocal_grads,
                                                                    vector1=delta,
                                                                    vector2=delta,
                                                                    elocal=self._elocal_cache,
                                                                    quadratic_model=self.param_groups[0]['quadratic_model'])
            deltaT_delta = self._get_scalar_product(vector1=delta,
                                                    vector2=delta)
            log_rho_den = torch.log(torch.abs(0.5 * deltaT_M_delta + (self.param_groups[0]['damping'] + self.param_groups[0]['l2_reg']) * deltaT_delta))
            log_rho = log_rho_num - log_rho_den
            return log_rho

    #=========================================================================================================================#
    #                        UPDATE THE PARAMETERS AND PERFORM A SINGLE STEP OF QUASI-NEWTON KFAC                             #
    #=========================================================================================================================#
    
    def _update_parameters(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                self.state[p]['momentum_buffer'] = d_p.clone().detach()  # update the momentum buffer
                p.add_(d_p)  # theta_{t+1} = theta_{t} + delta_{t} #update is here!

    @torch.no_grad()
    def step(self, closure: Optional[Callable], elocal: Tensor, x: Tensor) -> None:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
            
            elocal (Tensor): The local energies of the many-body position vector of the hilbert space.

            X (Tensor): The current many-body position vector of the hilbert space.
        """
        raw_gradients = self.get_all_gradients()

        # Update caches of per-sample gradient of log_abs_psi and of elocal (and cache elocal and energy_mean as well)
        self._update_per_sample_gradients_cache(params=dict(self.model.named_parameters()), x=x)
        self._elocal_cache = elocal
        energy_var, energy_mean = torch.var_mean(elocal)
        energy_std = (energy_var / elocal.shape[0]).sqrt()

        self._update_epsilon()

        self._update_all_kronecker_factors(elocal=elocal, energy_mean=energy_mean, mode=self.param_groups[0]['precondition_method'])
        if(self.param_groups[0]['step'] % self.param_groups[0]['gamma_adapt_interval'] == 0 and self.param_groups[0]['adapt_gamma'] == True and self.param_groups[0]['precondition_method'] in ['KFAC']):
            gammas = [self.param_groups[0]['gamma']*(self.param_groups[0]['gamma_adapt_decay']**self.param_groups[0]['gamma_adapt_interval']),
                        self.param_groups[0]['gamma'],
                        self.param_groups[0]['gamma']/(self.param_groups[0]['gamma_adapt_decay']**self.param_groups[0]['gamma_adapt_interval'])]
            M_delta_scores = torch.zeros(len(gammas), device=self.device)
            delta_cache = [None]*len(gammas) #pre-allocate list (so I can index)
            alphas = torch.zeros(len(gammas), device=self.device)
            mus = torch.zeros(len(gammas), device=self.device)
            for i, gamma in enumerate(gammas):
                self._regularise_all_kronecker_factors(gamma=gamma) #regularise then linear solve
                Delta = self._get_all_unscaled_natural_gradients()
                delta, alpha, mu, Mdelta = self._rescale_natural_gradients(raw_gradients=raw_gradients,
                                                                            Delta=Delta,
                                                                            loss=energy_mean)
                alphas[i], mus[i] = alpha, mu
                delta_cache[i] = delta.copy()  # deepcopy dict
                M_delta_scores[i] = Mdelta
            idx=M_delta_scores.argmin()
            self.param_groups[0]['gamma'] = gammas[idx]
            delta = delta_cache[idx] 
            self.alpha_opt = alphas[idx]
            self.mu_opt = mus[idx]
        else:
            self._regularise_all_kronecker_factors(gamma=self.param_groups[0]['gamma'])  # regularise then linear solve
            Delta = self._get_all_unscaled_natural_gradients()
            delta, alpha, mu, Mdelta = self._rescale_natural_gradients(raw_gradients=raw_gradients,
                                                                    Delta=Delta,
                                                                    loss=energy_mean)
            self.alpha_opt = alpha
            self.mu_opt = mu

        if(self.param_groups[0]['use_kl_clipping']):
            DeltaT_F_Delta = self._get_vector_matrix_vector_product(logpsi_grads=self.per_sample_logpsi_grads, elocal_grads=self.per_sample_elocal_grads,
                                                                    vector1=delta, vector2=delta, elocal=self._elocal_cache, quadratic_model='Fisher')
            deltaT_delta = self._get_scalar_product(vector1=delta, vector2=delta)
            KL_div = DeltaT_F_Delta  + (self.param_groups[0]['damping'] + self.param_groups[0]['l2_reg']) * deltaT_delta  
            self.KL_factor = min(self.param_groups[0]['kl_max'], (self.param_groups[0]['norm_constraint']/KL_div)**(0.5))  # same across groups
        else:
            self.KL_factor = None

        self.set_all_gradients(gradients=delta, scale=self.KL_factor)
        self._update_parameters()  # theta_{t+1} = theta_{t} + delta_{t}
        if(self.param_groups[0]['step'] % self.param_groups[0]['damping_adapt_interval'] == 0 and self.param_groups[0]['adapt_damping']):
            self._update_damping(loss_theta=energy_mean,
                                 loss_std=energy_std,
                                 loss_theta_plus_delta=closure(),
                                 raw_gradients=raw_gradients,
                                 delta=delta)

        #Clamp lambda/gamma within their respective min/max values
        self.param_groups[0]['gamma'] = min(max(self.param_groups[0]['gamma'], self.param_groups[0]['min_gamma']), self.param_groups[0]['max_gamma'])
        self.param_groups[0]['damping'] = min(max(self.param_groups[0]['damping'], self.param_groups[0]['min_damping']), self.param_groups[0]['max_damping'])
        
        self.param_groups[0]['step'] += 1