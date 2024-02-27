if(__name__ == '__main__'):

    import argparse

    parser = argparse.ArgumentParser(description='Adam Optimization NQS')
    #https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers/14117567

    parser.add_argument("-N", "--num_fermions", type=int,   default=2,     help="Number of fermions in physical system")
    parser.add_argument("-H", "--num_hidden",   type=int,   default=64,    help="Number of hidden neurons per layer")
    parser.add_argument("-L", "--num_layers",   type=int,   default=2,     help="Number of layers within the network")
    parser.add_argument("-D", "--num_dets",     type=int,   default=1,     help="Number of determinants within the network's final layer")
    parser.add_argument("-V", "--V0",           type=float, default=0.,    help="Interaction strength (in harmonic units)")
    parser.add_argument("-S", "--sigma0",       type=float, default=0.5,   help="Interaction distance (in harmonic units")
    parser.add_argument("--preepochs",          type=int,   default=1000,  help="Number of pre-epochs for the pretraining phase")
    parser.add_argument("--epochs",             type=int,   default=10000, help="Number of epochs for the energy minimisation phase")
    parser.add_argument("-C", "--chunks",       type=int,   default=1,     help="Number of chunks for vectorized operations")
    #parser.add_argument("--dtype",              type=str,   default='float64',      help='Default dtype')

    args = parser.parse_args()

    import torch
    from torch import nn, Tensor
    from torch.func import vmap #functorch

    #default torch options
    torch.manual_seed(238472394)
    torch.set_printoptions(4)
    torch.backends.cudnn.benchmark=True
    torch.set_default_dtype(torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.get_default_dtype()
    dtype_str = str(torch.get_default_dtype()).split('.')[-1]

    import sys
    DIR="./"
    sys.path.append(DIR+"src/")

    from Models import vLogHarmonicNet
    from Samplers import MetropolisHastings
    from Hamiltonian import calculatorLocalEnergy
    from Pretraining import HermitePolynomialMatrix #as HO
    from HartreeFock import HFsolver

    from utils import calc_pretraining_loss
    from utils import load_dataframe, load_model, count_parameters, get_groundstate
    from utils import sync_time, clip
    from utils import generate_final_energy, round_to_err, str_with_err
    import numpy as np

    #args
    nfermions = args.num_fermions #number of input nodes
    num_hidden = args.num_hidden  #number of hidden nodes per layer
    num_layers = args.num_layers  #number of layers in network
    num_dets = args.num_dets      #number of determinants (accepts arb. value)
    func = nn.Tanh()  #activation function between layers
    pretrain = True   #pretraining output shape?

    nwalkers=4096
    n_sweeps=400 #n_discard
    std=1.
    target_acceptance=0.5

    V0 = args.V0
    sigma0 = args.sigma0

    pt_save_every_ith=500
    em_save_every_ith=500
    nchunks=1 #deprecated
    clip_factor=5

    preepochs=args.preepochs
    epochs=args.epochs

    gs=nfermions**2/2.

    #objects
    net = vLogHarmonicNet(num_input=nfermions,
                          num_hidden=num_hidden,
                          num_layers=num_layers,
                          num_dets=num_dets,
                          func=func,
                          pretrain=pretrain)
    net=net.to(device=device, dtype=dtype)

    sampler = MetropolisHastings(network=net,
                                 dof=nfermions,
                                 nwalkers=nwalkers,
                                 target_acceptance=target_acceptance)

    calc_elocal = calculatorLocalEnergy(model=net,
                                        V0=V0,
                                        sigma0=sigma0)

    HO = HermitePolynomialMatrix(num_particles=nfermions)

    optim = torch.optim.Adam(params=net.parameters(), lr=1e-4, fused=torch.cuda.is_available())

    groundstate=nfermions**2/2
    gs_energy_CI = get_groundstate(A=nfermions, V0=V0, datapath=DIR+"groundstate/")

    if(V0!=0):
        HF = HFsolver(A=nfermions,
                      xL=6,
                      Nx=240,
                      V0=V0,
                      s=sigma0,
                      device=torch.device('cpu'),
                      itermax=10000)
        enerhf, enerhfp, ekin0hf, eho, epot0hf, esum0hf = HF() #run Hartree-Fock
    else:
        enerhf=torch.Tensor([groundstate])

    ###############################################################################################################################################
    #####                                           PRE-TRAINING LOOP                                                                         #####
    ###############################################################################################################################################

    model_path_pt = DIR+"results/pretrain/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s_chkp.pt" % \
                     (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, optim.__class__.__name__, True, device, dtype_str)
    filename_pt = DIR+"results/pretrain/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_%s_PT_%s_device_%s_dtype_%s.csv" % \
                     (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, optim.__class__.__name__, True, device, dtype_str)

    net.pretrain = True
    writer_pt = load_dataframe(filename_pt)
    output_dict = load_model(model_path=model_path_pt, device=device, net=net, optim=optim, sampler=sampler)

    start=output_dict['start']
    net=output_dict['net']
    optim=output_dict['optim']
    sampler=output_dict['sampler']

    #Pre-training
    for preepoch in range(start, preepochs+1):
        stats={}

        t0=sync_time()

        x, _ = sampler(n_sweeps=n_sweeps)

        network_orbitals = net(x)
        target_orbitals = HO(x) #no_grad op

        mean_preloss, stddev_preloss = calc_pretraining_loss(network_orbitals, target_orbitals)

        optim.zero_grad()
        mean_preloss.backward()  
        optim.step()

        t1=sync_time()

        stats['epoch'] = preepoch
        stats['loss_mean'] = mean_preloss.item()
        stats['loss_std'] = stddev_preloss.item()
        stats['proposal_width'] = sampler.sigma
        stats['acceptance_rate'] = sampler.acceptance_rate.item()

        stats['walltime'] = t1-t0

        writer_pt(stats) #push data to Writer

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
            #write data here?

        sys.stdout.write("Epoch: %6i | Loss: %6.4f +/- %6.4f | Walltime: %4.2e (s)      \r" % (preepoch, mean_preloss, stddev_preloss, t1-t0))
        sys.stdout.flush()

    print("\nPretraining is complete")

    ###############################################################################################################################################
    #####                                           ENERGY-MINIMISATION LOOP                                                                  #####
    ###############################################################################################################################################

    net.pretrain = False
    optim = torch.optim.Adam(params=net.parameters(), lr=1e-4, fused=torch.cuda.is_available()) #new optimizer

    model_path = DIR+"results/energy/checkpoints/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s_chkp.pt" % \
                    (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, optim.__class__.__name__, False, device, dtype_str)
    filename = DIR+"results/energy/data/A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s.csv" % \
                    (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, optim.__class__.__name__, False, device, dtype_str)

    writer = load_dataframe(filename)
    output_dict = load_model(model_path=model_path, device=device, net=net, optim=optim, sampler=sampler)

    start=output_dict['start']
    net=output_dict['net']
    optim=output_dict['optim']
    sampler=output_dict['sampler']

    print("params dtype: ",next(net.parameters()).dtype)

    #Energy Minimisation
    for epoch in range(start, epochs+1):
        stats={}

        t0=sync_time()

        x, _ = sampler(n_sweeps)

        params = dict(net.named_parameters())

        elocal = vmap(calc_elocal, in_dims=(None, 0))(params, x)
        #elocal = vmap(torch.func.functional_call, in_dims=(None,None,0))(calc_elocal, params, x)
        elocal = clip(elocal=elocal, clip_factor=clip_factor)

        #_, logabs = net(x) #could probably use has_aux on elocal evaluation to remove additional call?
        #loss_elocal = 2.*((elocal - torch.mean(elocal)).detach() * logabs)
        _, log_psi = net(x)
        mean_centred_logpsi = (log_psi - log_psi.mean())
        mean_centred_elocal = (elocal - elocal.mean()).detach()
        loss_values = (mean_centred_elocal * mean_centred_logpsi)
        loss_mean = 2.0 * torch.mean(loss_values)

        with torch.no_grad():
            energy_var, energy_mean = torch.var_mean(elocal)

        #loss = torch.mean(loss_elocal)

        optim.zero_grad()
        loss_mean.backward()  #populates leafs with grads
        optim.step()

        # Compute 2-norm of energy grad to compare with final update one
        raw_energy_gradient_vector = torch.cat([p.grad.flatten() for p in net.parameters()], dim=-1)
        norm2_energy_grad = raw_energy_gradient_vector.pow(2).sum(-1).sqrt()

        t1 = sync_time()
        stats['epoch'] = epoch
        stats['loss'] = loss_mean.item()
        stats['energy_mean'] = energy_mean.item()
        stats['energy_std'] = (energy_var / nwalkers).sqrt().item()
        stats['GS'] = gs
        stats['CI'] = gs_energy_CI
        stats['HF'] = enerhf.item() #maybe item func?
        stats['proposal_width'] = sampler.sigma.item()
        stats['acceptance_rate'] = sampler.acceptance_rate.item()
        
        stats['l2norm_energy_grad'] = norm2_energy_grad.item()
        stats['alpha'] = optim.param_groups[0]['lr']
        stats['l_infinity'] = max([torch.max(p.grad.abs()).item() for p in net.parameters()])

        stats['walltime'] = t1-t0

        writer(stats)

        #async save?
        if(epoch % em_save_every_ith == 0):
            torch.save({'epoch':epoch,
                        'model_state_dict':net.state_dict(),
                        'optim_state_dict':optim.state_dict(),
                        'loss':loss_mean,
                        'energy':energy_mean,
                        'energy_std':(energy_var / nwalkers).sqrt(),
                        'chain_positions':sampler.chain_positions.detach(),
                        'log_prob':sampler.log_prob.detach(),
                        'sigma':sampler.sigma},
                        model_path)
            writer.write_to_file(filename)
            #write data here?

        sys.stdout.write("Epoch: %6i | Energy: %6.4f +/- %6.4f | CI: %6.4f | HF: %6.4f | Walltime: %4.2e (s)        \r" % (epoch, energy_mean, (energy_var / nwalkers).sqrt(), gs_energy_CI, enerhf, t1-t0))
        sys.stdout.flush()

    print("\nEnergy Minimization is complete")


    n_batches = 10_000
    energy_stats = generate_final_energy(calc_elocal=calc_elocal,
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

    optim_name=optim.__class__.__name__
    final_energy_str = DIR+"results/final/FINAL_A%02i_H%03i_L%02i_D%02i_%s_W%04i_P%06i_V%4.2e_S%4.2e_%s_PT_%s_device_%s_dtype_%s.npz" % \
                    (nfermions, num_hidden, num_layers, num_dets, func.__class__.__name__, nwalkers, preepochs, V0, sigma0, optim_name, False, device, dtype)
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
