# FINE
## Introduction:
This repository provides an electric-field-aware implementation of Node-Equivariant Message Passing Neural Networks (NEMP) for representing potential energy surfaces, built on the JAX framework. FINE keeps the node-level tensor-product structure of the newer NEMP codebase and injects an external electric field through field-direction spherical harmonics and field-intensity neural coefficients.

Molecular dynamics (MD) simulations can be performed using the JAX-MD package to fully leverage GPU acceleration and achieve optimal performance. For more details about the methodology and benchmarks, please refer to the associated paper.
## Requirements:
* jax+flax
* optax
* ASE
* JAX-MD

## Examples:
Training can be performed by running the following command:
```
python3 $path "train"
```
where $path is the directory where the code is located. This command executes the train.py script inside the train folder.

All training parameters are specified in the config.json file, and the accepted input data format is extended XYZ. Each frame can include `Field="Ex Ey Ez"` in the header. If `dipole_table` is enabled, the header can include `dipole="mux muy muz"`. If `bec_table` is enabled, each atomic line is expected to contain 9 Born-effective-charge components before the force columns.

## References
If you use this package, please cite these works.
1. NEMP model: Yaolong Zhang and Hua Guo. [abs/2508.16086](https://arxiv.org/abs/2508.16086)
