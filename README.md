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

| Argument                      | Type    | Description                                             |
|-------------------------------|---------|---------------------------------------------------------|
| `-N`/`--num_fermions`         | `int`   | Number of fermions in physical system                   |
| `-H`/`--num_hidden`           | `int`   | Number of hidden neurons per layer                      |
| `-L`/`--num_layers`           | `int`   | Number of layers within the network                     |
| `-D`/`--num_dets`             | `int`   | Number of determinants within the network's final layer |
| `-V`/`--V0`                   | `float` | Interaction strength (in harmonic units)                |
| `-S`/`--sigma0`               | `float` | Interaction distance (in harmonic units)                |
| `--preepochs`                 | `int`   | Number of pre-epochs for the pretraining phase          |
| `--epochs`                    | `int`   | Number of epochs for the energy minimisation phase      |
| `-QM`/`--quadratic_model`     | `str`   | Type of Quadratic Model                                 |
| `-PM`/`--precondition_method` | `str`   | Type of Preconditioning                                 |
| `-MR`/`--number_of_minres_it` | `int`   | Number of MinRes iteration                              |


One can run the KFAC, QN-KFAC, QN-MR-KFAC, NGD, and DGD optimisers via the `run_qnkfac.py` script below with the corresponding flags

| Optimsier  | QM            | PM     | MR |
|------------|---------------|--------|----|
| KFAC       | Fisher        | KFAC   | 0  |
| QN-KFAC    | Quasi-Hessian | KFAC   | 0  |
| QN-MR-KFAC | Quasi-Hessian | KFAC   | >0 |
| NGD        | Fisher        | Fisher | 0  |
| DGD        | VMC           | VMC    | >0 |

```bash

python run_qnkfac.py -N 2 -V -20 -S 0.5 -QM VMC -PM VMC -MR 50

```

## License 

The license of this repositry is [Apache Licensse 2.0](https://choosealicense.com/licenses/apache-2.0/).