# Second-Order Optimization NQS

This repository contains the associated code for 'Second-order Optimisation strategies for neural network quantum states', henceforth the paper,
and focuses on second-order based optimisation schemes applied towards neural-network quantum states that can accelerate convergence beyond pre-existing methods.

## Installation

You can reproduce the results of the paper by cloning the repositry with,

`git clone https://github.com/jwtkeeble/second-order-optimization-NQS.git`

and running the `run_qnkfac.py` script as shown below in the Usage section.

## Requirements

The requirements can be found in `requirements.txt` and can be installed via `pip` or `conda`.

Please note this optimiser is only compatible with pytorch 2.1 and above (due to reliance on the `torch.func` namespace for efficiently computing per-sample gradients)

## Usage

```python

python run_qnkfac.py -N 2 -V -20 -S 0.5 -QM VMC -PM VMC -MR 50

```

## License 

This repositry uses the [Apache Licensse 2.0](https://choosealicense.com/licenses/apache-2.0/)