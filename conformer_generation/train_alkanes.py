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
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_rl import *

from deep_rl.component.envs import DummyVecEnv, make_env

import envs
from models import *

class Curriculum():
    def __init__(self, win_cond=0.5, success_percent=0.7, fail_percent=0.2, min_length=100):
        self.win_cond = win_cond
        self.success_percent = success_percent
        self.fail_percent = fail_percent
        self.min_length = min_length

    def return_win_cond():
        return self.win_cond

def ppo_feature(args, **kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 35
    single_process = (config.num_workers == 1)
    config.linear_lr_scale = False

    lr_base = args.learning_rate if args.learning_rate else 6.32456E-06

    if config.linear_lr_scale:
        lr = lr_base * config.num_workers
    else:
        lr = lr_base * np.sqrt(config.num_workers)

    win_cond = args.win_cond if args.win_cond else 1.4

    config.curriculum = Curriculum(min_length=config.num_workers, win_cond=win_cond)
    config.task_fn = lambda: AdaTask(config.env_name, num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process) # causes error

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
    config.mini_batch_size = 5 * config.num_workers
    config.ppo_ratio_clip = 0.2
    config.save_interval = config.num_workers * 200 * 5
    config.eval_interval = config.num_workers * 200 * 5
    config.eval_episodes = 1

    eval_env = args.eval_env if args.eval_env else 'AlkaneValidation10-v0'
    config.eval_env = AdaTask(eval_env, seed=random.randint(0,7e4))

    config.state_normalizer = DummyNormalizer()
    run_steps(PPORecurrentEvalAgent(config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run batch training')

    parser.add_argument('-w', '--win_cond', type=float, default=1.4)

    parser.add_argument('-s', '--starter')

    parser.add_argument('-lr', '--learning_rate', type=float)

    args = parser.parse_args()

    mkdir('log')
    mkdir('tf_log')

    if args.point_dim:
        point_dim = args.point_dim
    else:
        point_dim = 5

    model_type = None

\
    model = RTGNBatch(6, 128, edge_dim=6, point_dim=point_dim)
    model_type = 'rtgn'

    if args.starter:
        print(f'loading {args.starter}')
        model.load_state_dict(torch.load(f'data/{args.starter}'))

    model.to(torch.device('cuda'))
    set_one_thread()
    select_device(0)
    tag = 'alkanes train'

    env_name = 'TenTorsionSetCurriculumPoints-v0'

    logging.info(env_name)

    agent = ppo_feature(args, tag=tag, env_name=env_name)
    logging.info('using ppo')
    logging.info(f'using {model_type}')
    logging.info(env_name)
    logging.info(tag)
    
    run_steps(agent)
