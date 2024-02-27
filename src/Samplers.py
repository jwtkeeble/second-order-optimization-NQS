#Pytorch package
from ast import Call
import torch
import torch.nn as nn

#Typecasting
from typing import Tuple, Callable
from torch import Tensor

from Models import vLogHarmonicNet

def rw_metropolis_kernel(logpdf: Callable, position: Tensor, log_prob: Tensor, sigma: float):
    delta_position = sigma * torch.randn_like(position)
    proposal = position + delta_position
    proposal_logprob = logpdf(proposal)

    log_uniform = torch.log(torch.rand_like(proposal_logprob))
    accept = log_uniform < (proposal_logprob - log_prob)
    
    acceptance_rate = accept.float().mean()

    position += accept[:, None] * delta_position
    log_prob = torch.where(accept, proposal_logprob, log_prob)
    return position, log_prob, acceptance_rate


class MetropolisHastings(nn.Module):
    def __init__(self, network: vLogHarmonicNet, dof: int, nwalkers: int, target_acceptance: float) -> None: #TODO rename dof into number of dof or sth
        super(MetropolisHastings, self).__init__()
        self.network = network  # /!\ Must be the same optimizer as in Optimizer's attribute
        self.dof = dof
        self.nwalkers = nwalkers
        self.target_acceptance = target_acceptance
        self.device = next(self.network.parameters()).device
        self.dtype = next(self.network.parameters()).dtype

        self.sigma = torch.tensor(1.0, device=self.device, dtype=self.dtype) #TODO more expressive name for sigma (to avoid confusion with other sigma's)
        self.acceptance_rate = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.chain_positions = torch.randn(size=(self.nwalkers, self.dof),
                                  device=self.device,
                                  dtype=self.dtype,
                                  requires_grad=False)  # isotropic initialisation.
        _pretrain = self.network.pretrain
        self.network.pretrain = False
        self.log_prob = self.network(self.chain_positions)[1].mul(2)
        self.network.pretrain = _pretrain

    def log_pdf(self, x: Tensor) -> Tensor:
      _pretrain = self.network.pretrain
      self.network.pretrain = False
      _, logabs = self.network(x)
      self.network.pretrain = _pretrain
      return 2.*logabs

    def _rescaled_sigma(self, acceptance_rate: Tensor) -> Tensor:
        return torch.as_tensor(max(acceptance_rate.item(), 0.05) * self.sigma / self.target_acceptance, device=self.device)

    @torch.no_grad()
    def forward(self, n_sweeps: int) -> Tuple[Tensor, Tensor]:
        for _ in range(n_sweeps):
            self.chain_positions, self.log_prob, self.acceptance_rate = rw_metropolis_kernel(logpdf=self.log_pdf, #TODO change name log_prob into log_probabilities (+ adapt rest of source)
                                                                                    position=self.chain_positions,
                                                                                    log_prob=self.log_prob,
                                                                                    sigma=self.sigma)
            if (self.target_acceptance is not None):
                self.sigma = torch.as_tensor(min(max(self._rescaled_sigma(self.acceptance_rate).item(), 0.01), 10.0), device=self.device) # Clamping sigma to avoid walkers to get stuck
        return self.chain_positions, self.log_prob