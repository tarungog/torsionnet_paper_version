# torsionnet_paper_version
This repository is the official PyTorch implementation of "TorsionNet: A Reinforcement Learning Approach to Sequential Conformer Search".

Tarun Gogineni, Ziping Xu, Exequiel Punzalan, Runxuan Jiang, Joshua Kammeraad, Ambuj Tewari, and Paul Zimmerman.

## Installation in Great Lakes Cluster
**Part 1: Prerequisites**
1. Dependencies and versions are stored in `environment.yml`. To create a conda environment with the dependencies, run:
    - `conda env create -f environment.yml`

2. Install customized agents:
    > cd rl-agents
    > pip install -e .

3. Install PyTorch Geometric:
    - `pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-geometric`
    - [Pytorch Geometric Official Installation Instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

4. Create "Data" Folder:
    > cd conformer_generation
    > mkdir data


## Run
The code is meant to be run on the Great Lakes cluster, although can be easily modified to run on other compute grids. The key Slurm script to run a training job is located at `gpu_batch_run.sh`, and all other scripts are based off of this one.

It calls the python file `run_batch_train.py`, which is where all details of the experiment must be set before running the job script. Here, we set the train and validation gym environments, along with the algorithmic hyperparameters.