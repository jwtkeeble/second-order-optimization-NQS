# Second-Order Optimization NQS

This repository contains the associated code for 'Second-order Optimisation strategies for neural network quantum states', henceforth the paper,
and focuses on second-order based optimisation schemes applied towards neural-network quantum states that can accelerate convergence beyond pre-existing methods.

## Installation

You can reproduce the results of the paper by cloning the repositry with,

`git clone https://github.com/jwtkeeble/second-order-optimization-NQS.git`

and running the `run_qnkfac.py` script as shown below in the [Usage](#usage) section to benchmark the variety of optimisers stated within the paper.

## Requirements

The requirements in order to run this script can be found in `requirements.txt` and can be installed via `pip` or `conda`.

Please **note**: this optimiser is only compatible with pytorch 2.1 and above (due to reliance on the `torch.func` namespace for efficiently computing per-sample gradients)

## Usage

The arguments for the `run_qnkfac.py` script are as follows:
1. `-N`/`--num_fermions`         : (type `int`)    Number of fermions in physical system
2. `-H`/`--num_hidden`           : (type `int`)    Number of hidden neurons per layer
3. `-L`/`--num_layers`           : (type `int`)    Number of layers within the network
4. `-D`/`--num_dets`             : (type `int`)    Number of determinants within the network's final layer
5. `-V`/`--V0`                   : (type `float`)  Interaction strength (in harmonic units)
6. `-S`/`--sigma0`               : (type `float`)  Interaction distance (in harmonic units)
7. `--preepochs`                 : (type `int`)    Number of pre-epochs for the pretraining phase
8. `--epochs`                    : (type `int`)    Number of epochs for the energy minimisation phase
9. `-QM`/`--quadratic_model`     : (type `str`)    Type of Quadratic Model
10. `-PM`/`--precondition_method` : (type `str`)    Type of Preconditioning
11. `-MR`/`--number_of_minres_it` : (type `int`)    Number of MinRes iteration

```bash

python run_qnkfac.py -N 2 -V -20 -S 0.5 -QM VMC -PM VMC -MR 50

```

## License 

The license of this repositry is [Apache Licensse 2.0](https://choosealicense.com/licenses/apache-2.0/).