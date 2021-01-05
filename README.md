# Flowfield approximation with Gaussian Ridge Functions

This repository contains companion code for the AIAA Scitech 2021 paper:

Scillitoe, A., Seshadri, P. and Wong, C., 2021. Instantaneous Flowfield Estimation with Gaussian Ridges. In: AIAA SciTech. AIAA. Available at: <https://arc.aiaa.org/doi/abs/10.2514/6.2021-1138>.

The code generates *embedded Gaussian ridge functions* (GRF's) which allow for rapid flowfield estimation.

## Instructions

The workflow consists of three key steps; 1) preprocessing, 2) obtaining GRF's, 3) postprocessing/prediction. The code is contained in python scripts, with input settings set in JSON `*.in` files.

### 1) Preprocessing

1. Generate base cartesian mesh with `python make_basemesh.py settings.in`. 

2. Preprocess CFD data with `python proc_data.py settings.in`. This will take the M number of CFD solution files, resample them onto the cartesian mesh generated in step 1, normalise the flow variables, and create the data matrix D. The CFD solutions are split into train and test sets here, and covariance matrices are built from the training data. 

### 2) Obtain ridge functions

Embedded GRF's can be obtained at subsampled points by running `python get_subspaces.py settings.in`. The resulting GRF's are saved in a numpy object array. 

### 3) Postprocessing/prediction

The obtained GRF's can be used to predict new flowfields by running `python postproc.py settings.in`. Predictions can also be visualised at given locations in the form of *sufficient summary plots*, and train and test errors can be calculated.  

## Caveats 

* At present, the CFD data must be `*.vtk` files obtained by running the SU2 CFD code with the incompressible solver.

* The flowfields are currently the 2D flow around a set of aerofoils, with the aerofoils parameterised with 50 Hickes-Henne bump functions. The framework is readily extendable to other flows, as long as the geometry can be suitably parameterised. Example `*.cfg` files and associated scripts to generate the deformed aerofoils, and run the CFD, are included in this repo. 

* The framework is parallelised into multiple worker jobs by setting `njobs>1`. It is recommended to run on as many cores as available in order to achieve reasonable run times. We ran on a 72 core virtual machine on Microsoft Azure for the results in the paper. (Note that currently the `loky` backend and the `joblib` python package are used for parallelisation. Future work may involve a custom mpi implementation for this, in order to allow for running across multiple compute nodes in more traditional HPC settings). 

## Dependencies

A few readily available python packages are required, including `numpy`, `sklearn`, `pymanopt`, `joblib` and `pyvista`.  
