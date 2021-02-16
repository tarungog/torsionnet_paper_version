import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import logging
import torch
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
import torch_geometric.nn as gnn

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from main.utils import *
from main.models import *
from main.environments import Task

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)

from concurrent.futures import ProcessPoolExecutor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loaded_policy(model, env):
    num_envs = 1
    single_process = (num_envs == 1)

    env = Task(env, seed=random.randint(0,7e4), num_envs=num_envs, single_process=single_process)
    state = env.reset()
    total_reward = 0
    start = True
    done = False
    step = 0
    while step < 200:
        with torch.no_grad():
            if start:
                prediction, rstates = model(state)
                start = False
            else:
                prediction, rstates = model(state, rstates)

        choice = prediction['a']
        step += 1
        state, rew, done, info = env.step(to_np(choice))
        total_reward += float(rew)

    if isinstance(info, tuple):
        for i, info_ in enumerate(info):
            print('episodic_return', info_['episodic_return'])
            episodic_return = info_['episodic_return']
    else:
        print('episodic_return', info['episodic_return'])
        episodic_return = info['episodic_return']
    return episodic_return


if __name__ == '__main__':
    model = RTGNBatch(6, 128, edge_dim=1)

    mean_outputs = []
    std_outputs = []
    max_outputs = []
    min_outputs = []

    full_mean = []
    full_std = []
    full_min = []
    full_max = []

    for i in range(0, 10):
        model.load_state_dict(torch.load(f'transfer_test_t_chain/models/{i}.model', map_location=device))
        model.to(device)

        for j in range(0, 10):
            samples = []

            for _ in range(5):
                samples.append(loaded_policy(model, f'TChainTest3-v{j}'))

            mean_output = np.array(samples).mean()
            std_output = np.array(samples).std()
            max_output = np.array(samples).max()
            min_output = np.array(samples).min()

            mean_outputs.append(mean_output)
            std_outputs.append(std_output)
            max_outputs.append(max_output)
            min_outputs.append(min_output)

        full_mean.append(mean_outputs)
        full_std.append(std_outputs)
        full_max.append(max_outputs)
        full_min.append(min_outputs)

        mean_outputs = []
        std_outputs = []
        max_outputs = []
        min_outputs = []

    ans = np.array(full_mean)
    df = DataFrame(ans)
    df.to_csv('energy_difference_t_chains_multisample_means.csv')
    print(df)

    ans = np.array(full_std)
    df = DataFrame(ans)
    df.to_csv('energy_difference_t_chains_multisample_std.csv')

    ans = np.array(full_min)
    df = DataFrame(ans)
    df.to_csv('energy_difference_t_chains_multisample_min.csv')

    ans = np.array(full_max)
    df = DataFrame(ans)
    df.to_csv('energy_difference_t_chains_multisample_max.csv')