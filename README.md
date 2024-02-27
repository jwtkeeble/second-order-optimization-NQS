# Second-Order Optimization NQS

This repository contains the associated code for '[Second-order Optimisation strategies for neural network quantum states](https://arxiv.org/abs/2401.17550)', and focuses on second-order based optimisation schemes applied towards neural-network quantum states (NQS) for Variational Monte Carlo (VMC) calculations.

This work focuses on expanding upon our previous work of '[Machine learning one-dimensional spinless trapped fermionic systems with neural-network quantum states](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.108.063320)' by focusing on novel second-order optimisation schemes for NQSs, rather than modifying the NQS itself.

These novel second-order optimisation schemes can achieve accelerated rates of convergence beyond standard second-order methods, and highlight key limitations of existing methods that can be solved by developing novel optimisation schemes.

## Installation

You can reproduce the results of the paper by cloning the repository with,

`git clone https://github.com/jwtkeeble/second-order-optimization-NQS.git`

and running the `run_second_order_opt.py` script as shown below in the [Usage](#usage) section to benchmark the variety of optimisers stated within the paper.

## Requirements

The requirements in order to run this script can be found in `requirements.txt` and can be installed via `pip` or `conda`.

**note** This package requires torch2.1+ in order to use the `torch.func` namespace to efficiently compute per-sample gradients.

## Usage

The arguments for the `run_second_order_opt.py` script are as follows:

| Argument                      | Type    | Default      | Description                                               |
|-------------------------------|---------|--------------|-----------------------------------------------------------|
| `-N`/`--num_fermions`         | `int`   | 2            | Number of fermions in physical system                     |
| `-H`/`--num_hidden`           | `int`   | 64           | Number of hidden neurons per layer                        |
| `-L`/`--num_layers`           | `int`   | 2            | Number of layers within the network                       |
| `-D`/`--num_dets`             | `int`   | 1            | Number of determinants within the network's final layer   |
| `-V`/`--V0`                   | `float` | 0            | Interaction strength (in harmonic units)                  |
| `-S`/`--sigma0`               | `float` | 0.5          | Interaction distance (in harmonic units)                  |
| `--preepochs`                 | `int`   | 1000         | Number of pre-epochs for the pretraining phase            |
| `--epochs`                    | `int`   | 1000         | Number of epochs for the energy minimisation phase        |
| `-QM`/`--quadratic_model`     | `str`   | QuasiHessian | Type of Quadratic Model ['Fisher', 'QuasiHessian', 'VMC'] |
| `-PM`/`--precondition_method` | `str`   | KFAC         | Type of Preconditioning ['KFAC', 'Fisher', 'VMC']         |
| `-MR`/`--number_of_minres_it` | `int`   | 50           | Number of maximum MinRes iteration                        |


One can run the KFAC, QN-KFAC, QN-MR-KFAC, NGD, and DGD optimisers of the paper via the `run_second_order_opt.py` script with the corresponding flags of the table below.

| Optimiser  | -QM           | -PM    | -MR |
|------------|---------------|--------|-----|
| KFAC       | Fisher        | KFAC   | 0   |
| QN-KFAC    | Quasi-Hessian | KFAC   | 0   |
| QN-MR-KFAC | Quasi-Hessian | KFAC   | > 0 |
| NGD        | Fisher        | Fisher | 0   |
| DGD        | VMC           | VMC    | > 0 |

For example, running the DGD MR=50 optimiser for a NQS with 64 hidden nodes, 2 layers, and a single determinant can be ran via the following command,

```bash

python run_second_order_opt.py -N 2 -H 64 -L 2 -D 1 -V -20 -S 0.5 -QM VMC -PM VMC -MR 50

```

## Results

The results of this simulation are stored within the `results/` directory for both pretraining (`results/pretrain/`) and the energy minimisation (`results/energy/`). Within each of these directories there exists a `data/` and `checkpoints/` directory, which stores the convergence data and the final variational state (as well as sampler state) respectively. 

1. The convergence data is stored as a `.csv` file, which can be easily manipulated by the `Pandas` library for data analysis and visualisation. 

2. The variational state is stored as a `.pt` file, following PyTorch convention, and stores the final NQS state as well as its MCMC sampler state.

## Warnings

**PyTorch Version**: This optimiser is only compatible with **PyTorch 2.1+** (due to the `torch.func` namespace for efficiently computing per-sample gradients).

**Layers**: The current state of the optimiser only supports NQSs that are comprised of `nn.Linear` objects and will silently skip over other layer types.

However, you can extend the optimiser to support other layer types by including definitions for 'merging'/'spliting' layer weights and how to perform the KFAC approximation. 

## License 

The license of this repositry is [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/).
