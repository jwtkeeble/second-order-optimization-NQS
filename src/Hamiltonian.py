import torch
from torch import nn, Tensor
from torch.func import functional_call, jacrev
import numpy as np
from typing import Tuple

class calculatorLocalEnergy(nn.Module):
    def __init__(self, model: nn.Module, V0: float, sigma0: float) -> None:
        super(calculatorLocalEnergy, self).__init__()
        self.model = model
        self.V0 = V0
        self.sigma0 = sigma0
        self.gauss_const = (self.V0 / (np.sqrt(2 * np.pi) * self.sigma0))
        # Local functions helping definition of energy calculating functions 
        def calc_logabs(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
            _, logabs = functional_call(self.model, params, x)
            return logabs
        def dlogabs_dx(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
            grad_logabs = jacrev(calc_logabs, argnums=1, has_aux=False)(params, x)
            return grad_logabs, grad_logabs
        def laplacian_psi(params: Tuple[Tensor], x: Tensor) -> Tuple[Tensor]:
            grad2_logabs, grad_logabs = jacrev(dlogabs_dx, argnums=1, has_aux=True)(params, x)
            return grad2_logabs.diagonal(0,-2,-1).sum() + grad_logabs.pow(2).sum()
        # Functional attributes
        self.kinetic_from_log_fn = lambda params, x: -0.5 * laplacian_psi(params, x)
        self.potential_fn = lambda x: 0.5 * (x.pow(2).sum(-1))
        self.gauss_int_fn = lambda x: self.gauss_const * (torch.exp(-(x.unsqueeze(-2) - x.unsqueeze(-1))**2 / (2 * sigma0**2)).triu(diagonal=1).sum(dim=(-2,-1)))

    def forward(self, params: Tuple[Tensor], x: Tensor) -> Tensor:
        _kin = self.kinetic_from_log_fn(params, x)
        _pot = self.potential_fn(x)
        _int = self.gauss_int_fn(x)
        _eloc = _kin + _pot + _int
        return _eloc