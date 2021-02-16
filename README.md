# TorsionNet Paper

This repository is the official PyTorch implementation of ["TorsionNet: A Reinforcement Learning Approach to Sequential Conformer Search"](https://arxiv.org/abs/2006.07078).

Tarun Gogineni, Ziping Xu, Exequiel Punzalan, Runxuan Jiang, Joshua Kammeraad, Ambuj Tewari, and Paul Zimmerman.

## Installation

1. Anaconda should be install in order to create a Conda environment with the required dependencies. Anaconda can be installed [here](https://www.anaconda.com/products/individual).

2. Dependencies and versions are stored in `environment.yml`. To create a conda environment with the dependencies, run:
    ```
    conda env create -f environment.yml
    ```

3. Install PyTorch Geometric. Due to version issues, PyTorch Geometric must be installed manually. You can find instructions for installing it [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).



## File Layout

* The `TorsionNet` directory contains scripts for running each of the three experiments mentioned in the TorsionNet paper: {alkane, lignin, and t_chains}. For more details on running the scripts, see the [run](##Run) section below.
* Scripts for generating the molecule files for each of the three experiments are located in the `TorsionNet` directory and are named `generate_{branched_alkanes, lignin, t_chain}.py` corresponding to each experiment.
* The agents, models, and environments are stored in the directory `TorsionNet/main`.
    * `TorsionNet/main/agents` contains implementations for the custom agents. The main agent used is PPO, which is stored in the file `PPO_recurrent_agent.py`. Some of the code for the agents is roughly based off of the RL framework [DeepRL](https://github.com/ShangtongZhang/DeepRL).
    * `TorsionNet/main/environments` contains implementations for the reinforcement learning environments used in each experiment. Most environments are stored in `graphenvironments.py`.
    * The file `models.py` in `TorsionNet/main` contains the implementation for the neural network used in most experiments, RTGNBatch.
* Pre-trained model parameters for each of the three experiments are stored in `TorsionNet/trained_models`.

## Run

Train and evaluation python scripts are located in the `TorsionNet` directory for all experiments: {alkane, lignin, t_chain}.

Scripts for training agents for each experiment are named `train_[experiment_name].py`. For example, to run the lignin experiments, run
 ```
 cd TorsionNet/
 python train_lignin.py
 ```
NOTE: for training the alkane environment, unzip the file `huge_hc_set.zip` first.

Model parameters are saved in the `TorsionNet/data` directory. Tensorboard is available to monitor the training process:
```
cd TorsionNet/
tensorboard --logdir tf_log/
```

Evaluation scripts are available for each of the experiments and are named `eval_[experiment name].py`. To run the evaluation script, we provide sample pre-trained model parameters. If training from scratch, first replace the path parameter in the "torch.load" function in the script with the path of the model weights that perform best on the validation environment. This can be checked via tensorboard. Model weights are stored at the same cadence as validation runs. After replacing the path parameter, the eval script can be run e.g.
```
cd TorsionNet/
python eval_lignin.py
```

## Results

This is a best effort reproduction of our implementation. There may be some nondeterminism.
