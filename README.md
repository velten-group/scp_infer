# scp-infer: Causal Inference using perturbed single-cell gene expression data
Apply a number of causal inference algorithms to experimental perturbed single-cell gene exression data.
Built using scanpy

## 1. Installation

There are two options provied to setup the usage of this package. 

1. As a simple option we reccommend to first create a clean python envrionment, where care has to be taken to select the correct python version `3.11.13` (otherwise older versions such as `3.11.8` or `3.10.x` should work as well). Then this repository can be cloned, and the requirements listed in `requirements.txt` installed using pip, e.g:

```
conda create -n <env-name> python=3.11.13
conda activate <env-name>
pip install -r requirements.txt
```
Activation of the environment for Jupyter:
```
ipython kernel install --user --name=<env-name>
```
2. Alternatively we provide a dockerfile in `\docker-deploy` for installing the package in a containerized environment. The image can be automatically configured with the `download_infer_pkg.sh` executable, which would require `docker-compose` to be installed.

## 2. Tutorials:

under [tutorials](tutorials) there are three jupyter notebooks giving examples of how to use the package:

1. [sergio-knockdown-run](tutorials/sergio-knockdown-run.ipynb) shows how synthetic gene expression data using SERGIO can be generated.
2. [synthetic_data](tutorials/synthetic_data.ipynb) gives an example of applying the GRN inference algorithms to a synthetically genrdated dataset.
3. [experimental_data](tutorials/experimental_data.ipynb) gives an example of applying the GRN inference algorithms to experimental CRISPR screen data, where first pre-processing and annotation is carried out. The used dataset, which is already highly filtered, can be replaced with other experimental datasets, where then care has to be taken with precprocessing and feature selection.

## 2. Documentation:
For Documentation of the usage of all the included functions see: [documentation - docs/build/html/index.html](docs/build/html/scp_infer.html)

[comment]: <> (https://html-preview.github.io/?url=https://github.com/jan-spr/scp_infer/blob/main/sphinx_doc/build/html/scp_infer.html)

## 2. Contents:
### 1. [scp_infer](src/scp_infer)
Module that cointains all of the functionality for GRN inference:
#### a. adata:
Data manipulation and annotation using Anndata/scanpy
#### b. inference:
Apllication of inference algorithms
#### c. eval:
Evaluation of predicted Networks using data-driven and biological approaches
#### d. utils:
A few utilites to help with the management e.g. of train and validation splits of Data and Evaluation of Networks on these.

### 1. [sergio_kd](src/sergio_kd)
Module that cointains the functionality for synthetic data generation using SERGIO.
