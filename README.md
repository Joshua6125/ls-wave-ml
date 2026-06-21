# Neural Approximations of the Wave Equation using First-Order Systems Least-Squares

This repository contains the implementation used for the numerical results used in the thesis 'Neural Approximations of the Wave Equation using First-Order Systems Least-Squares' of Joshua van Rooij.

Don't hesitate to reach in case of any questions: joshuavanrooij@gmail.com.

## Overview

The general structure is as follows. See files themselves for more details.

```text
neural-pde/
├── src/                            # Contains the training pipeline.
│   ├── training/                   # Module of the main training loop.
│   ├── models/                     # Module for the neural model definitions, configs, and builder.
│   ├── loss_functions/             # Module for all loss formulations.
│   │   ├── fosls/                  # First-Order Systems Least-Squares
│   │   ├── pinn/                   # PINN
│   │   ├── gpinn/                  # Gradient-PINN
│   │   └── vpinn/                  # Variational-PINN
│   ├── integration/                # Contains the objects needed for integration on N-D hypercubes.
│   └── training.py                 # EP for starting training.
├── experiments/                    # Experiment setup and configs for those used in the Thesis.
│   ├── configs/                    # Configs per experiment.
│   ├── scripts/                    # Experiment definitions. Both DOF sweeps and training-time based.
│   └── main.py/                    # Main EP for starting an experiment.
├── tests/                          # Testing of training framework. Structure is closely mirrored of src/
├── README.md                       # This README.
├── run_all.sh                      # Used to run all paper specific models.
├── requirements-dev.txt            # Requirements needed to testing + running.
└── requirements.txt                # Requirements to run experiments.
```

## How to run

First install the required dependencies from ```requirements.txt```. This only contains the CPU-based JAX package. See the JAX documentation for GPU specific instructions.

To only generate the plots used in the thesis, run ```run_all.sh``` in the main directory.

For generating new data/models, seek ```\experiments\main.py``` for further instructions.
