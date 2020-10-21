# TorsionNet Paper
This repository is the official PyTorch implementation of ["TorsionNet: A Reinforcement Learning Approach to Sequential Conformer Search"](https://arxiv.org/abs/2006.07078).

Tarun Gogineni, Ziping Xu, Exequiel Punzalan, Runxuan Jiang, Joshua Kammeraad, Ambuj Tewari, and Paul Zimmerman.

## Code description
- The main code for running the experiments and tuning the hyperparameters are `conformer_generation/lignin_train.py`. 
- The molecule environment code is in `conformer_generation/graphenvironments.py`.
- The models are stored in `conformer_generation/models.py`
- The training agents are stored in `rl_agent/agent/PPO_recurrent_agent.py`. Our PPO implementation is built on top of the RL framework [DeepRL](https://github.com/ShangtongZhang/DeepRL).


## Installation
1. Dependencies and versions are stored in `environment.yml`. To create a conda environment with the dependencies, run:
    - `conda env create -f environment.yml`

2. Install customized agents:
    ```
    cd rl-agents
    pip install -e .
    ```

3. Install PyTorch Geometric:
    - `pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html`
    - `pip install torch-geometric`
    - [Pytorch Geometric Official Installation Instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)


## Run
Train and evaluation python scripts are located in the conformer_generation directory for all experiments.
- Run lignin experiments:
 - For training, run lignin_train.py
 ```
 cd conformer_generation
 python lignin_train.py
 ```

Tensorboard is available to monitor the training process:
```
tensorboard --logdir tf_log/
```

Model weights are saved in the data directory. Evaluation scripts are available for each of the experiments {lignin, branched_alkanes, t-chains} and are named [experiment name]_eval.py. To run the evaluation script, first replace the path parameter in the "torch.load" function within the script with the path of where the model weights to evaluate are stored. Then run
```
python lignin_eval.py
```
