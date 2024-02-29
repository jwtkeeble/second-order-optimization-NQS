if(__name__ == '__main__'):
    
    import torch
    from torch import nn
    from torch.func import vmap

    #default torch options
    torch.manual_seed(238472394)
    torch.set_printoptions(4)
    torch.backends.cudnn.benchmark=True
    torch.set_default_dtype(torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.get_default_dtype()
    dtype_str=str(torch.get_default_dtype()).split('.')[-1]
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype_str}")

    import sys
    DIR="./" #change absolute path here
    sys.path.append(DIR+"src/")

    from Models import vLogHarmonicNet
    from Samplers import MetropolisHastings
    from Hamiltonian import calculatorLocalEnergy
    from Pretraining import HermitePolynomialMatrix
    from HartreeFock import HFsolver

    from utils import calc_pretraining_loss
    from utils import load_dataframe, load_model, get_groundstate
    from utils import sync_time, clip
    from utils import generate_final_energy, round_to_err, str_with_err
    
    import numpy as np
    
    import argparse

    parser = argparse.ArgumentParser(description='Second-order Optimization NQS')
    #https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers/14117567

    parser.add_argument("-N", "--num_fermions",        type=int,   default=2,              help="Number of fermions in physical system")
    parser.add_argument("-H", "--num_hidden",          type=int,   default=64,             help="Number of hidden neurons per layer")
    parser.add_argument("-L", "--num_layers",          type=int,   default=2,              help="Number of layers within the network")
    parser.add_argument("-D", "--num_dets",            type=int,   default=1,              help="Number of determinants within the network's final layer")
    parser.add_argument("-V", "--V0",                  type=float, default=0.,             help="Interaction strength (in harmonic units)")
    parser.add_argument("-S", "--sigma0",              type=float, default=0.5,            help="Interaction distance (in harmonic units")
    parser.add_argument("--preepochs",                 type=int,   default=1000,           help="Number of pre-epochs for the pretraining phase")
    parser.add_argument("--epochs",                    type=int,   default=1000,           help="Number of epochs for the energy minimisation phase")
    parser.add_argument("-QM","--quadratic_model",     type=str,   default='QuasiHessian', help="Type of Quadratic Model")
    parser.add_argument("-PM","--precondition_method", type=str,   default='KFAC',         help="Type of Preconditioning")
    parser.add_argument("-MR","--number_of_minres_it", type=int,   default=0,              help='Number of MinRes iteration')

    args = parser.parse_args()

    # Network arguments 
    nfermions = args.num_fermions  # number of input nodes
    num_hidden = args.num_hidden   # number of hidden nodes per layer
    num_layers = args.num_layers   # number of layers in network
    num_dets = args.num_dets       # number of determinants (accepts arb. value)
    act_func = nn.Tanh()  # activation function between layers
    pretrain = True   # pretraining output shape? (TODO improve comment)
    # Sampler arguments
    nwalkers = 4096
    n_sweeps = 400
    std = 1. # (TODO rename)
    target_acceptance = 0.5
    # Interaction arguments
    V0 = args.V0
    sigma0 = args.sigma0
    # Optimizer arguments
    pt_save_every_ith = 10
    em_save_every_ith = 1  # 100
    clip_factor = 5

    preepochs = args.preepochs
    epochs = args.epochs

    gs_energy = nfermions**2 / 2.

    #objects
    net = vLogHarmonicNet(num_input=nfermions,
                        num_hidden=num_hidden,
                        num_layers=num_layers,
                        num_dets=num_dets,
                        act_func=act_func,  # TODO rename func in activation function or sth 
                        pretrain=pretrain)
    net = net.to(device=device,dtype=dtype)

    sampler = MetropolisHastings(network=net,
                                dof=nfermions,
                                nwalkers=nwalkers,
                                target_acceptance=target_acceptance)

    calc_elocal = calculatorLocalEnergy(model=net,
                                        V0=V0,
                                        sigma0=sigma0)

    HO = HermitePolynomialMatrix(num_particles=nfermions)

    optim = torch.optim.Adam(params=net.parameters(), lr=1e-4)

    groundstate_energy = nfermions**2/2  # TODO clarify gs_energy vs groundstate_energy
    gs_energy_CI = get_groundstate(A=nfermions, V0=V0, datapath=DIR+"groundstate/")

    if(V0!=0):
        HF = HFsolver(A=nfermions,
                    xL=6,
                    Nx=240,
                    V0=V0,
                    s=sigma0,
                    device=torch.device('cpu'),
                    itermax=10_000)
        enerhf, enerhfp, ekin0hf, eho, epot0hf, esum0hf = HF()  # run Hartree-Fock
    else:
        enerhf=torch.Tensor([groundstate_energy])

    ###############################################################################################################################################
    #####                                           PRE-TRAINING LOOP                                                                         #####
    ###############################################################################################################################################

    model_path_pt = DIR+"results/pretrain/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s_chkp.pt" % \
                    (nfermions, num_hidden, num_layers, num_dets, act_func.__class__.__name__, nwalkers, preepochs, optim.__class__.__name__, True, device, dtype_str)
    filename_pt = DIR+"results/pretrain/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s.csv" % \
                    (nfermions, num_hidden, num_layers, num_dets, act_func.__class__.__name__, nwalkers, preepochs, optim.__class__.__name__, True, device, dtype_str)

    net.pretrain = True

    output_dict = load_model(model_path=model_path_pt, device=device, net=net, optim=optim, sampler=sampler)
    writer_pt = load_dataframe(filename_pt)

    start = output_dict['start'] # TODO choose a better name for 'start' (~last_epoch or sth)
    net = output_dict['net']
    optim = output_dict['optim']
    sampler = output_dict['sampler']

    #Pre-training
    for preepoch in range(start, preepochs+1):  # TODO preepochs naming to make clear it's maximum of preepochs
        stats = {}
        
        t0 = sync_time()

        x, _ = sampler(n_sweeps=n_sweeps)
        
        network_orbitals = net(x)
        target_orbitals = HO(x) # no_grad op
        
        mean_preloss, stddev_preloss = calc_pretraining_loss(network_orbitals, target_orbitals)

        optim.zero_grad()
        mean_preloss.backward()  
        optim.step()

        t1 = sync_time()

        stats['epoch'] = preepoch
        stats['loss_mean'] = mean_preloss.item()
        stats['loss_std'] = stddev_preloss.item()
        stats['proposal_width'] = sampler.sigma.item()
        stats['acceptance_rate'] = sampler.acceptance_rate.item()

        stats['walltime'] = t1 - t0

        writer_pt(stats)  # push data to Writer

        if(preepoch % pt_save_every_ith == 0):
            torch.save({'epoch':preepoch,
                        'model_state_dict':net.state_dict(),
                        'optim_state_dict':optim.state_dict(),
                        'loss':mean_preloss.item(),
                        'chain_positions':sampler.chain_positions.detach(),
                        'log_prob':sampler.log_prob.detach(),
                        'sigma':sampler.sigma.item()},
                        model_path_pt)
            writer_pt.write_to_file(filename_pt)
            # write data here?

        sys.stdout.write("Epoch: %6i | Loss: %6.4f +/- %6.4f | Walltime: %4.2e (s)      \r" % (preepoch, mean_preloss, stddev_preloss, t1 - t0))
        sys.stdout.flush()

    print("\nPretraining is complete")

    ###############################################################################################################################################
    #####                                           ENERGY-MINIMISATION LOOP                                                                  #####
    ###############################################################################################################################################

    net.pretrain = False

    # We have 2 copies of the energy estimator (as one cannot compute derivatives for an nn.Module object that have backward hooks in PyTorch2.1+, previous versions silently skipped this error)
    # So we have one copy with hooks (to precondition with KFAC-like optimisers) and one for re-evaluating the energy with the Levenberg-Marquadt rule.
    from copy import deepcopy
    cnet = deepcopy(net)
    calc_local_no_hooks = calculatorLocalEnergy(model=cnet, V0=V0, sigma0=sigma0)

    def closure():
        x, _ = sampler(n_sweeps=n_sweeps)
        with torch.enable_grad():
            params = dict(net.named_parameters()) 
            elocal = vmap(calc_local_no_hooks, in_dims=(None, 0))(params, x) 
        return elocal.mean()

    from SecondOrderOptimizer import SecondOrderOpt

    optim = SecondOrderOpt(model=net,
                
                        damping=1e3,
                        adapt_damping=True,
                        damping_adapt_interval=5,
                        damping_adapt_decay=(19/20),
                        min_damping=1e-7,
                        max_damping=1e20,
                            
                        l2_reg=0,
                            
                        #gamma_init=None,
                        adapt_gamma=True, 
                        gamma_adapt_interval=5,
                        gamma_adapt_decay=(19/20)**(0.5),  
                        min_gamma=1e-4, 
                        max_gamma=1e20,
                            
                        max_epsilon=0.9,

                        quadratic_model = args.quadratic_model,
                        precondition_method = args.precondition_method,
                        number_of_minres_it = args.number_of_minres_it,

                        use_effective_gradients=False,  

                        use_kl_clipping=False,
                        norm_constraint=1e-2,
                        kl_max=1.) 

    optim_name = optim.__class__.__name__ \
                + "-" + args.quadratic_model + '-' + args.precondition_method + "-MR-" + str(args.number_of_minres_it)

    model_path = DIR+"results/energy/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_chkp.pt" % \
                    (nfermions, num_hidden, num_layers, num_dets, act_func.__class__.__name__, nwalkers, preepochs, V0, sigma0, optim_name, False, device, dtype_str)
    filename = DIR+"results/energy/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s.csv" % \
                    (nfermions, num_hidden, num_layers, num_dets, act_func.__class__.__name__, nwalkers, preepochs, V0, sigma0, optim_name, False, device, dtype_str)

    writer = load_dataframe(filename)
    output_dict = load_model(model_path=model_path, device=device, net=net, optim=optim, sampler=sampler)

    # Loading initial model
    start = output_dict['start']
    net = output_dict['net']
    optim = output_dict['optim']
    sampler = output_dict['sampler']

    for epoch in range(start, epochs+1):
        stats = {}
        t0 = sync_time()

        # Markov-Chain Monte-Carlo propagation
        x, _ = sampler(n_sweeps=n_sweeps) 

        # Loss (i.e. energy) gradient calculation
        params = dict(net.named_parameters())
        elocal = vmap(calc_local_no_hooks, in_dims=(None, 0))(params, x)
        with torch.no_grad():
            energy_var, energy_mean = torch.var_mean(elocal)
            energy_std = (energy_var / nwalkers).sqrt()
        elocal = clip(elocal, 5)  # clip local values to within 5 times the l1-variation of the median
        mean_centred_elocal = (elocal - elocal.mean()).detach()
        _, log_psi = net(x)  # forward hooks are triggered
        mean_centred_log_psi = (log_psi - log_psi.mean())
        aux_local_loss = (mean_centred_elocal * mean_centred_log_psi)
        aux_loss = 2.0 * torch.mean(aux_local_loss)
        if(aux_loss.isnan()):
            print(f"\nAuxiliary loss {aux_loss} is NaN - quitting program")
            print("\nAuxiliary local loss:")
            print(aux_local_loss)
            print("\nmean_centred_elocal:")
            print(mean_centred_elocal)
            print("\nmean_centred_log_psi:")
            print(mean_centred_log_psi)
            print("\nlog_psi:")
            print(log_psi)
            print("\nelocal:")
            print(elocal)
            sys.exit()
        optim.zero_grad()
        aux_loss.backward(retain_graph=True)  # populate grad attributes (and triggers backward hooks)
        energy_gradients = optim.get_all_gradients()  # compute energy gradients

        # Per sample gradient of log_abs_psi calculation. Stored in the hooks
        _, log_psi = net(x)
        mean_log_psi = torch.mean(log_psi)
        optim.zero_grad()
        mean_log_psi.backward()  # populate correct hooks, but wrong grad attribute
        optim.set_all_gradients(gradients=energy_gradients)  # restore correct .grad but keep correct hooks
        
        # Compute 2-norm of energy grad to compare with final update one
        raw_energy_gradient_vector = torch.cat([p.grad.flatten() for p in net.parameters()], dim=-1)
        norm2_energy_grad = raw_energy_gradient_vector.pow(2).sum(-1).sqrt()

        # Combining per sample grad(log_abs_psi) stored in optimizer.state by hooks with loss (i.e. energy) gradient stored in .grad attribute to update parameters
        optim.step(closure=closure, elocal=elocal, x=x)  

        # Saving data
        t1 = sync_time()
        parameters_update_vector = torch.cat([p.grad.flatten() for p in net.parameters()], dim=-1)
        norm2 = parameters_update_vector.pow(2).sum(-1).sqrt()
        stats['epoch'] = epoch
        stats['loss'] = aux_loss.item()
        stats['energy_mean'] = energy_mean.item()  # this slows it down, no? we're syncing
        stats['energy_std'] = (energy_var / nwalkers).sqrt().item()  # TODO refactorize
        stats['GS'] = gs_energy
        stats['CI'] = gs_energy_CI
        stats['HF'] = enerhf.item()  # maybe item func?
        stats['proposal_width'] = sampler.sigma.item()
        stats['acceptance_rate'] = sampler.acceptance_rate.item()
        stats['walltime'] = t1 - t0
        stats['l2norm'] = norm2.item()
        stats['l2norm_energy_grad'] = norm2_energy_grad.item()
        stats['alpha'] = optim.alpha_opt.item()
        stats['mu'] = optim.mu_opt.item()
        stats['l_infinity'] = max([torch.max(p.grad.abs()).item() for p in net.parameters()])
        for key in optim.param_groups[0]:
            if(key != 'params'):
                value = optim.param_groups[0][key]
                if(isinstance(value, torch.Tensor)):
                    stats[key] = value.cpu().detach().numpy()
                elif(isinstance(value, torch.nn.Module)):
                    stats[key] = value.__class__.__name__
                else:
                    stats[key] = value
        writer(stats)
        if(epoch % em_save_every_ith == 0): 
            torch.save({'epoch':epoch,
                        'model_state_dict':net.state_dict(),
                        'optim_state_dict':optim.state_dict(),
                        'loss':aux_loss.item(),
                        'energy':energy_mean.item(),
                        'energy_std':(energy_var  / nwalkers).sqrt().item(), #TODO refactorize
                        'chain_positions':sampler.chain_positions.detach(),
                        'log_prob':sampler.log_prob.detach(),
                        'sigma':sampler.sigma.item()},
                        model_path)
            writer.write_to_file(filename)
        # Outputting current state
        sys.stdout.write("(Energy-Train) Epoch %6i/%6i | Energy: %6.4f +/- %6.4f | Rate: %2.2f | λ: %4.2e | γ: %4.2e | ln(ρ): %4.2e | α: %4.2e | μ: %4.2e | CI: %6.4f | HF: %6.4f | Walltime: %4.2e%s\r" % \
                        (epoch, epochs, energy_mean, energy_std, sampler.acceptance_rate, optim.param_groups[0]['damping'], optim.param_groups[0]['gamma'], optim.param_groups[0]['log_rho'], optim.alpha_opt, optim.mu_opt, gs_energy_CI, enerhf, t1 - t0, "s"+10*" "))
        sys.stdout.flush()

    print("\n")


    n_batches = 10_000
    energy_stats = generate_final_energy(calc_elocal=calc_local_no_hooks,
                                        sampler=sampler,
                                        n_batches=n_batches,
                                        chunk_size=None, #full-batch vectorization
                                        n_sweeps=1,     #10 is fine, 400 is too much 
                                        storage_device=torch.device('cpu')) #store on cpu to save memory for GPU
    energy_mean=energy_stats['mean']
    error_of_mean=energy_stats['error_of_mean']
    batch_variance=energy_stats['batch_variance']
    variance=energy_stats['variance']
    R_hat=energy_stats['R_hat']

    x, dx = round_to_err(energy_mean.item(), error_of_mean.item())
    energy_str = str_with_err(x, dx)
    print(f"Energy: {energy_str} | R_hat: {R_hat:6.4f}")

    final_energy_str = DIR+"results/final/FINAL_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s.npz" % \
                    (nfermions, num_hidden, num_layers, num_dets, act_func.__class__.__name__, nwalkers, preepochs, V0, sigma0, optim_name, False, device, dtype)
    print(f"Saving to {final_energy_str}")
    data = {'energy_mean': x,
            'error_of_mean': dx,
            'energy_str':energy_str,
            'batch_variance':batch_variance,
            'variance':variance,
            'R_hat':R_hat,
            'HF':enerhf.cpu(),
            'CI':gs_energy_CI}
    np.savez(final_energy_str, **data)