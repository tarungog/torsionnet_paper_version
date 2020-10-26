# TorsionNet Paper
This repository is the official PyTorch implementation of ["TorsionNet: A Reinforcement Learning Approach to Sequential Conformer Search"](https://arxiv.org/abs/2006.07078).

Tarun Gogineni, Ziping Xu, Exequiel Punzalan, Runxuan Jiang, Joshua Kammeraad, Ambuj Tewari, and Paul Zimmerman.

## Code description
- The main code for running the experiments and tuning the hyperparameters are `conformer_generation/lignin_train.py`. 
- The molecule environment code is in `conformer_generation/graphenvironments.py`.
- The models are stored in `conformer_generation/models.py`
- The training agents are stored in `rl_agent/agent/PPO_recurrent_agent.py`. Our PPO implementation is built on top of the RL framework [DeepRL](https://github.com/ShangtongZhang/DeepRL).
- Scripts for obtaining benchmarks from OpenBabel and Rdkit are in `conformer_generation/benchmark_openbabel.py` and `conformer_generation/benchmark_rdkit.py`.


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
 - For training, run train_lignin.py
 ```
 cd conformer_generation
 python train_lignin.py
 ```

Tensorboard is available to monitor the training process:
```
tensorboard --logdir tf_log/
```

Model weights are saved in the data directory. Evaluation scripts are available for each of the experiments {lignin, branched_alkanes, t-chains} and are named [experiment name]_eval.py. To run the evaluation script, we will provide sample model weights. If training from scratch, first replace the path parameter in the "torch.load" function in the script with the path of the model weights that perform best on the validation environment. This can be checked via tensorboard. Model weights are stored at the same cadence as validation runs. After replacing the path parameter, the eval script can be run e.g.
```
python eval_lignin.py
```

## Results
This is a best effort reproduction of our implementation. There may be some nondeterminism.