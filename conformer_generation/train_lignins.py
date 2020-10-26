from os import environ

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

from utils import *

import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_rl import *

from deep_rl.component.envs import DummyVecEnv, make_env

import envs
from models import *

class Curriculum():
    def __init__(self, win_cond=0.7, success_percent=0.7, fail_percent=0.2, min_length=100):
        self.win_cond = win_cond
        self.success_percent = success_percent
        self.fail_percent = fail_percent
        self.min_length = min_length

    def return_win_cond():
        return self.win_cond

def ppo_feature(env_name, args, **kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 35
    single_process = (config.num_workers == 1)
    config.linear_lr_scale = False

    base_lr = args.learning_rate if args.learning_rate else 5e-6

    if config.linear_lr_scale:
        lr = base_lr * config.num_workers
    else:
        lr = base_lr * np.sqrt(config.num_workers)

    config.curriculum = Curriculum(min_length=config.num_workers)

    config.task_fn = lambda: AdaTask(env_name, num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process) # causes error

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.network = model
    config.hidden_size = model.dim
    config.discount = 0.9999
    config.use_gae = True
    config.gae_tau = 0.95
    config.state_normalizer = DummyNormalizer()
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.001
    config.gradient_clip = 0.5
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    # config.mini_batch_size = config.rollout_length * config.num_workers
    config.mini_batch_size = 25
    config.ppo_ratio_clip = 0.2
    config.save_interval = config.num_workers * 1000 * 2
    config.eval_interval = config.num_workers * 1000 * 2
    config.eval_episodes = 1
    config.eval_env = AdaTask('LigninPruningSkeletonValidationLong-v0', seed=random.randint(0,7e4))
    config.state_normalizer = DummyNormalizer()
    run_steps(PPORecurrentEvalAgent(config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run batch training')

    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-s', '--starter')

    args = parser.parse_args()
    
    model = RTGNBatch(6, 128, edge_dim=6, point_dim=5)
    env_name = 'LigninAllSetPruningLogSkeletonCurriculumLong-v0'
    
    if args.starter:
        print(f'loading {args.starter}')
        model.load_state_dict(torch.load(f'data/{args.starter}'))

    model.to(torch.device('cuda'))
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    select_device(0)
    tag = 'train_lignins'
    agent = ppo_feature(env_name, args, tag=tag)
    logging.info(tag)
    run_steps(agent)
