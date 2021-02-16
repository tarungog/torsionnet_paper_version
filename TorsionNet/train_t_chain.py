import numpy as np
import random
import argparse
import torch
import logging

from main.config import Config
from main.utils import *
from main.models import *
from main.environments import Task
from main.agents import A2CRecurrentCurriculumAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def a2c_feature(tag, env_name, model):
    config = Config()
    config.tag=tag
    config.env_name = env_name

    config.num_workers = 1
    single_process = (config.num_workers == 1)
    config.train_env = Task(config.env_name, num_envs=config.num_workers, seed=random.randint(0,1e5), single_process=single_process)
    config.linear_lr_scale = False
    if config.linear_lr_scale:
        lr = 7e-5 * config.num_workers
    else:
        lr = 7e-5 * np.sqrt(config.num_workers)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)
    config.network = model
    config.discount = 0.9999 # gamma
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.0001 #ent_coef
    config.rollout_length = 5 # n_steps
    config.gradient_clip = 0.5 #max_grad_norm
    config.max_steps = 5000000
    config.save_interval = config.num_workers * 200 * 5

    return A2CRecurrentCurriculumAgent(config)

if __name__ == '__main__':
    model = RTGNBatch(6, 128, edge_dim=1)


    mkdir('log')
    mkdir('tf_log')
    mkdir('data')

    model.to(device)
    set_one_thread()
    select_device(0)
    tag = "tchain train"

    env_name = 'TChainTrain-v0'
    agent = a2c_feature(tag=tag, env_name=env_name, model=model)
    logging.info(env_name)
    logging.info(tag)

    agent.run_steps()