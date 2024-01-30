import torch
from torch.autograd import Function
from torch import Tensor
from typing import Tuple, Any

from Functions_utils import get_log_gamma, get_log_rho, get_Xi_diag, get_off_diagonal_elements

#Our custom function!
class custom_function(Function): 

  generate_vmap_rule = True

  @staticmethod
  def forward(matrices: Tensor, log_envs: Tensor) -> Tuple[Tensor, Tensor]:
    """The forward call (or evaluation) of the custom `torch.autograd.Function` which computes a Signed-Logabs value for a LogSumExp function (with max substraction)
    :param matrices: A Tensor representing the set of D matrcies from the last Equivariant Layer
    :type matrices: class: `torch.Tensor`
    :param log_envs: A Tensor representing the set of D matrcies from the Log-Envelope function
    :type log_envs: class: `torch.Tensor`
    :return out: A Tuple containing the global sign and global logabs value of the Signed-Log Summed Determinant function
    :type out: `Tuple[torch.Tensor, torch.Tensor]`
    """
    sgns, logabss = torch.slogdet(matrices * torch.exp(log_envs))
    max_logabs_envs = torch.max(logabss, keepdim=True, dim=-1)[0] #grab the value only (no indices)
    scaled_dets = sgns*torch.exp( logabss - max_logabs_envs )  #max subtraction
    summed_scaled_dets = scaled_dets.sum(keepdim=True, dim=-1) #sum
    global_logabs = (max_logabs_envs + (summed_scaled_dets).abs().log()).squeeze(-1) #add back in max value and take logabs
    global_sgn = summed_scaled_dets.sign().squeeze(-1) #take sign
    return global_sgn, global_logabs

  @staticmethod
  def setup_context(ctx, inputs, output):
    matrices, log_envs = inputs
    global_sgn, global_logabs = output
    ctx.mark_non_differentiable(global_sgn)
    ctx.save_for_backward(global_sgn, global_logabs, matrices, log_envs)

  @staticmethod
  def backward(ctx: Any, grad_global_sgn: Tensor, grad_global_logabs: Tensor) -> Tuple[Tensor, Tensor]:
    global_sgn, global_logabs, matrices, log_envs = ctx.saved_tensors
    dLoss_dA, dLoss_dS, U, S, VT, detU, detVT, normed_G, sgn_prefactor, \
    U_normed_G_VT, U_normed_G_VT_exp_log_envs \
    = custom_functionBackward.apply(matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs)
    return dLoss_dA, dLoss_dS

class custom_functionBackward(Function):

  generate_vmap_rule = True

  @staticmethod
  def forward(matrices: Tensor, log_envs: Tensor, global_sgn: Tensor, global_logabs: Tensor, grad_global_sgn: Tensor, grad_global_logabs: Tensor) -> Tuple[Tensor, Tensor]:
    U, S, VT = torch.linalg.svd(matrices * torch.exp(log_envs))  #all shape [B, D, A, A]
    detU, detVT = torch.linalg.det(U), torch.linalg.det(VT)      #both shape [B,D]
    log_G = get_log_gamma(S.log())                               #shape [B, D, A] (just the diagonal)
    normed_G = torch.exp( log_G - global_logabs.unsqueeze(-1).unsqueeze(-1) )   #shape [B,D,A]
    U_normed_G_VT = U @ normed_G.diag_embed() @ VT
    U_normed_G_VT_exp_log_envs = torch.exp(log_envs)*U_normed_G_VT
    sgn_prefactor = ((grad_global_logabs * global_sgn).unsqueeze(-1) * detU * detVT)
    dLoss_dA = sgn_prefactor.unsqueeze(-1).unsqueeze(-1) * U_normed_G_VT_exp_log_envs
    dLoss_dS = matrices * dLoss_dA
        
    return dLoss_dA, dLoss_dS, U, S, VT, detU, detVT, normed_G, sgn_prefactor, \
                          U_normed_G_VT, U_normed_G_VT_exp_log_envs

  @staticmethod
  def setup_context(ctx: Any, inputs, output):
    matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs = inputs
    dLoss_dA, dLoss_dS, U, S, VT, detU, detVT, normed_G, sgn_prefactor, \
                          U_normed_G_VT, U_normed_G_VT_exp_log_envs = output
    ctx.mark_non_differentiable(grad_global_sgn, global_sgn)
    ctx.save_for_backward(dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
                          U_normed_G_VT, U_normed_G_VT_exp_log_envs, grad_global_sgn, grad_global_logabs, global_sgn, global_logabs)

  @staticmethod
  def backward(ctx: Any, grad_G: Tensor, grad_H: Tensor, grad_U: Tensor, grad_S: Tensor, grad_VT: Tensor, \
               grad_detU: Tensor, grad_detVT: Tensor, grad_normed_G: Tensor, grad_sgn_prefactor: Tensor, grad_U_normed_G_VT: Tensor, grad_U_normed_G_VT_exp_log_envs: Tensor, \
               ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
    U_normed_G_VT, U_normed_G_VT_exp_log_envs, grad_global_sgn, grad_global_logabs, global_sgn, global_logabs = ctx.saved_tensors
    dF_dA, dF_dS, dF_dsgn_Psi, dF_dlogabs_Psi, dF_dgrad_sgn_Psi, dF_dgrad_logabs_Psi = \
        custom_functionDoubleBackward.apply(matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs, \
                                            dLoss_dA, dLoss_dS, U, S, VT, detU, detVT, normed_G, sgn_prefactor, U_normed_G_VT, U_normed_G_VT_exp_log_envs, \
                                            grad_G, grad_H)
    return dF_dA, dF_dS, dF_dsgn_Psi, dF_dlogabs_Psi, dF_dgrad_sgn_Psi, dF_dgrad_logabs_Psi

class custom_functionDoubleBackward(Function):
   
    generate_vmap_rule = True

    @staticmethod
    def forward(matrices: Tensor, log_envs: Tensor, global_sgn: Tensor, global_logabs: Tensor, grad_global_sgn: Tensor, grad_global_logabs: Tensor, \
                dLoss_dA: Tensor, dLoss_dS: Tensor, U: Tensor, S: Tensor, VT: Tensor, detU: Tensor, detVT: Tensor, normed_G: Tensor, sgn_prefactor: Tensor, \
                U_normed_G_VT: Tensor, U_normed_G_VT_exp_log_envs: Tensor, \
                grad_G: Tensor, grad_H: Tensor):
        log_envs_max = torch.max(torch.max(log_envs, keepdim=True, dim=-2)[0], keepdim=True, dim=-1)[0]
        grad_K = (grad_G + grad_H * matrices) * torch.exp(log_envs - log_envs_max)
        M = VT @ grad_K.transpose(-2,-1) @ U

        #Calculate normed rho matrices
        log_rho = get_log_rho(S.log())
        normed_rho = torch.exp( log_rho - global_logabs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + log_envs_max)

        #Calculate normed Xi matrices
        Xi_diag = get_Xi_diag(M, normed_rho)
        Xi_off_diag = get_off_diagonal_elements(-M*normed_rho)
        normed_Xi = Xi_diag + Xi_off_diag #perhaps 1 operation?

        #calculate c constant; sum(dim=(-2,-1)) is summing over kl or mn ; sum(..., dim=-1) is summing over determinants
        c = global_sgn * torch.sum( detU * detVT * torch.sum( (grad_G + matrices * grad_H ) * U_normed_G_VT_exp_log_envs, dim=(-2,-1) ), dim=-1)
        normed_Xi_minus_c_normed_G = (normed_Xi - c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*normed_G.diag_embed())
        U_Xi_c_G_VT =  U @ normed_Xi_minus_c_normed_G @ VT
        U_Xi_c_G_VT_exp_log_envs = U_Xi_c_G_VT * torch.exp(log_envs)
        dF_dA = sgn_prefactor.unsqueeze(-1).unsqueeze(-1) * (U_Xi_c_G_VT_exp_log_envs + grad_H * U_normed_G_VT_exp_log_envs)
        dF_dS = sgn_prefactor.unsqueeze(-1).unsqueeze(-1) * (matrices * U_Xi_c_G_VT_exp_log_envs + (grad_G + grad_H * matrices)*U_normed_G_VT_exp_log_envs)

        dF_dsgn_Psi = None
        dF_dlogabs_Psi = None
        dF_dgrad_sgn_Psi = None
        dF_dgrad_logabs_Psi = c

        return dF_dA, dF_dS, dF_dsgn_Psi, dF_dlogabs_Psi, dF_dgrad_sgn_Psi, dF_dgrad_logabs_Psi

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        matrices, log_envs, global_sgn, global_logabs, grad_global_sgn, grad_global_logabs, dLoss_dA, dLoss_dS, U, S, VT, detU, detVT, normed_G, sgn_prefactor, U_normed_G_VT, U_normed_G_VT_exp_log_envs, grad_G, grad_H = inputs
        dF_dA, dF_dS, dF_dsgn_psi, dF_dlogabs_psi, dF_dgrad_sgn_psi, dF_dgrad_logabs_psi = output

        ctx.mark_non_differentiable(global_sgn, grad_global_sgn)
        ctx.save_for_backward(dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
                              U_normed_G_VT, U_normed_G_VT_exp_log_envs, grad_global_sgn, grad_global_logabs, global_sgn, global_logabs, \
                              dF_dA, \
                              grad_G, grad_H)
    @staticmethod
    def backward(ctx: Any, grad_dF_dA: Tensor, grad_dF_dS: Tensor, grad_dF_dsgn_Psi: Tensor, grad_dF_dlogabs_Psi: Tensor, grad_dF_dgrad_sgn_Psi: Tensor, grad_dF_dgrad_logabs_psi: Tensor):
        dLoss_dA, dLoss_dS, U, S, VT, matrices, log_envs, detU, detVT, normed_G, sgn_prefactor, \
        U_normed_G_VT, U_normed_G_VT_exp_log_envs, grad_global_sgn, grad_global_logabs, global_sgn, global_logabs, \
        dF_dA, \
        grad_G, grad_H = ctx.saved_tensors
                
        def stable_XinvT_t_grad_M_I_minus_MT_D_for(U: Tensor, VT: Tensor, log_rho: Tensor, grad_M: Tensor, log_normalization: Tensor, global_sign):
          temp_M = VT @ grad_M.transpose(-2,-1) @ U
          normed_rho = torch.exp(log_rho - log_normalization.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
          Xi_M_diag = get_Xi_diag(temp_M, normed_rho)
          Xi_M_off_diag = get_off_diagonal_elements(-temp_M*normed_rho)
          normed_Xi_M = Xi_M_diag + Xi_M_off_diag #perhaps 1 operation?
          return global_sign.unsqueeze(-1).unsqueeze(-1) * (U @ normed_Xi_M @ VT)

        #our intermediate objects
        X = (matrices * torch.exp(log_envs))
        sgns, logabss = torch.slogdet(X)
        
        log_rho = get_log_rho(S.log())
        log_G = get_log_gamma(S.log()) 

        D = (detU * detVT).unsqueeze(-1).unsqueeze(-1) * U @ log_G.exp().diag_embed() @ VT
        psi_normed_D = (detU * detVT * global_sgn.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1) * U_normed_G_VT

        grad_K = grad_G * torch.exp(log_envs) + grad_H * X
        grad_N = grad_dF_dA * torch.exp(log_envs) + grad_dF_dS * X
        grad_Q = (grad_dF_dgrad_logabs_psi/grad_global_logabs) * grad_K + grad_H * grad_N + grad_dF_dS * grad_G * torch.exp(log_envs)

        psi_normed_t_grad_N = torch.diagonal(grad_N.transpose(-2,-1) @ psi_normed_D, 0, -2, -1).sum(dim=-1)
        psi_normed_t_grad_K = torch.diagonal(grad_K.transpose(-2,-1) @ psi_normed_D, 0, -2, -1).sum(dim=-1)
        psi_normed_t_grad_Q = torch.diagonal(grad_Q.transpose(-2,-1) @ psi_normed_D, 0, -2, -1).sum(dim=-1)

        detX_normed_D = (detU * detVT * sgns).unsqueeze(-1).unsqueeze(-1) * (U @ ((log_G - logabss.unsqueeze(-1)).exp().diag_embed()) @ VT)

        detX_normed_t_grad_N = torch.diagonal(grad_N.transpose(-2,-1) @ detX_normed_D, 0, -2, -1).sum(dim=-1)
        detX_normed_t_grad_K = torch.diagonal(grad_K.transpose(-2,-1) @ detX_normed_D, 0, -2, -1).sum(dim=-1)
        # detX_normed_t_grad_Q = torch.diagonal(grad_Q.transpose(-2,-1) @ detX_normed_D, 0, -2, -1).sum(dim=-1)

        grad_C_part1 = grad_N * detX_normed_t_grad_K.unsqueeze(-1).unsqueeze(-1) + grad_K * detX_normed_t_grad_N.unsqueeze(-1).unsqueeze(-1) \
                       - grad_K @ detX_normed_D.transpose(-2,-1) @ grad_N - grad_N @ detX_normed_D.transpose(-2,-1) @ grad_K
        grad_C_part2 = -torch.sum(psi_normed_t_grad_K.unsqueeze(-1).unsqueeze(-1) * grad_N + psi_normed_t_grad_N.unsqueeze(-1).unsqueeze(-1) * grad_K, dim=-3) \
                       + grad_Q
        grad_C = grad_C_part1 + grad_C_part2
        t_grad_C = torch.diagonal(grad_C.transpose(-2,-1) @ D, 0, -2, -1).sum(dim=-1)

        stable_U_Xi_psi_normed_N_VT = stable_XinvT_t_grad_M_I_minus_MT_D_for(U, VT, log_rho, grad_N, logabss, global_sgn.unsqueeze(-1) * detU * detVT) 
        stable_U_Xi_psi_normed_K_VT = stable_XinvT_t_grad_M_I_minus_MT_D_for(U, VT, log_rho, grad_K, logabss, global_sgn.unsqueeze(-1) * detU * detVT) 
        stable_U_Xi_psi_normed_C_VT = stable_XinvT_t_grad_M_I_minus_MT_D_for(U, VT, log_rho, grad_C, logabss, global_sgn.unsqueeze(-1) * detU * detVT) 

        si_part1 = detX_normed_t_grad_N * detX_normed_t_grad_K \
                    - torch.diagonal(grad_N.transpose(-2,-1) @ detX_normed_D @ grad_K.transpose(-2,-1) @ detX_normed_D, offset=0, dim1=-2, dim2=-1).sum(dim=-1)
        si_part2 = detX_normed_t_grad_N * psi_normed_t_grad_K \
                    - torch.diagonal(grad_N.transpose(-2,-1) @ psi_normed_D @ grad_K.transpose(-2,-1) @ detX_normed_D, offset=0, dim1=-2, dim2=-1).sum(dim=-1)
        si = 2 * torch.sum(psi_normed_t_grad_N, dim=-1) * torch.sum(psi_normed_t_grad_K, dim=-1) - torch.sum(psi_normed_t_grad_Q, dim=-1) \
              - torch.sum(si_part2, dim=-1, keepdim=True) - si_part1
        
        normed_dG_dA_part1 = stable_U_Xi_psi_normed_C_VT + si.unsqueeze(-1).unsqueeze(-1) * psi_normed_D
        normed_dG_dA_part2 = grad_dF_dS * (stable_U_Xi_psi_normed_K_VT - (torch.sum(psi_normed_t_grad_K, dim=-1)) * psi_normed_D + grad_H * psi_normed_D)
        normed_dG_dA_part3 = grad_H * (stable_U_Xi_psi_normed_N_VT - (torch.sum(psi_normed_t_grad_N, dim=-1) - grad_dF_dgrad_logabs_psi/grad_global_logabs) * psi_normed_D)

        normed_dG_dgrad_G_part1 = stable_U_Xi_psi_normed_N_VT
        normed_dG_dgrad_G_part2 = ((grad_dF_dgrad_logabs_psi/grad_global_logabs) - torch.sum(psi_normed_t_grad_N,dim=-1)) * psi_normed_D #vmap ok, batch no
        normed_dG_dgrad_G_part3 = grad_dF_dS * psi_normed_D

        #The derivatives!
        dG_dA = grad_global_logabs * torch.exp(log_envs) * (normed_dG_dA_part1 + normed_dG_dA_part2 + normed_dG_dA_part3)
        dG_dS = (matrices * dG_dA) + (grad_dF_dA * dF_dA) + (grad_dF_dS * grad_G * dLoss_dA) + (grad_dF_dgrad_logabs_psi/grad_global_logabs * grad_G * dLoss_dA) + \
                grad_global_logabs * torch.exp(log_envs) * grad_G * (stable_U_Xi_psi_normed_N_VT - torch.sum(psi_normed_t_grad_N, dim=-1) * psi_normed_D )
        dG_dgrad_G = grad_global_logabs * torch.exp(log_envs) * (normed_dG_dgrad_G_part1 + normed_dG_dgrad_G_part2 + normed_dG_dgrad_G_part3)
        dG_dgrad_H = matrices * dG_dgrad_G + grad_dF_dA * dLoss_dA
        
        #Rest are None by definition!
        dG_dglobal_sgn = None 
        dG_dglobal_logabs = None
        dG_dgrad_global_sgn = None
        dG_dgrad_global_logabs = None

        #definitely None
        dG_dLoss_dA = None
        dG_dLoss_dS = None
        dG_dU = None
        dG_dS_from_SVD = None
        dG_dVT = None
        dG_ddetU = None
        dG_ddetVT = None
        dG_dnormed_G = None
        dG_dsgn_prefactor = None
        dG_dU_normed_G_VT = None
        dG_dU_normed_G_VT_exp_log_envs = None

        return dG_dA, dG_dS, dG_dglobal_sgn, dG_dglobal_logabs, dG_dgrad_global_sgn, dG_dgrad_global_logabs, \
               dG_dLoss_dA, dG_dLoss_dS, dG_dU, dG_dS_from_SVD, dG_dVT, dG_ddetU, dG_ddetVT, dG_dnormed_G, dG_dsgn_prefactor, dG_dU_normed_G_VT, dG_dU_normed_G_VT_exp_log_envs, dG_dgrad_G, dG_dgrad_H
