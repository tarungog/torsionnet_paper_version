#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .agent_utils import Storage
from .base_agent import BaseAgent
from ..utils import to_np, tensor, mkdir
import time
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance
from collections import deque
from rdkit import Chem

import numpy as np
import numpy.random

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CRecurrentCurriculumAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.train_env
        if config.network:
            self.network = config.network
        else:
            self.network = config.network_fn()
        self.network.to(device)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.recurrent_states = None
        self.done = True
        self.smh = None
        self.choice_ind = 0
        self.good_episodes = 0
        self.reward_buffer = deque([], maxlen=100)
        torch.autograd.set_detect_anomaly(True)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            start = time.time()
            if self.done:
                print('done1')
                prediction, self.recurrent_states = self.network(states)
            else:
                prediction, self.recurrent_states = self.network(states, self.recurrent_states)
            end = time.time()


            self.writer.add_scalar('forward_pass_time', end-start, self.total_steps)
            print('forward time', end-start)

            self.done = False

            start = time.time()
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            end = time.time()

            # rew = info[0]

            self.writer.add_scalar('env_step_time', end-start, self.total_steps)
            print('step time', end-start)
            for idx, infoDict in enumerate(info):
                if infoDict['episodic_return'] is not None:
                    print('logging episodic return train...', self.total_steps)
                    self.writer.add_scalar('episodic_return_train', infoDict['episodic_return'], self.total_steps)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1).to(device),
                         'm': tensor(1 - terminals).unsqueeze(-1).to(device)})

            states = next_states
            self.total_steps += config.num_workers

            # for ifs in info:
            #     if ifs['episodic_return'] is not None:
            #         self.reward_buffer.append(ifs['episodic_return'])
            ifs = info[0]
            if ifs['episodic_return'] is not None:
                rew = ifs['episodic_return']

        if self.total_steps % (config.num_workers * 200) == 0:
            print('real rewards', rew)
            if rew > 7.5:
                self.good_episodes += 1
            else:
                self.good_episodes = 0

            print('good_episodes', self.good_episodes)

            if self.good_episodes >= 10:
                filename = f'{self.choice_ind}'
                self.choice_ind += 1
                torch.save(self.network.state_dict(), f'transfer_test_t_chain/models/{filename}.model')
                self.good_episodes = 0

                if self.choice_ind == 10:
                    raise Exception('finished')

        self.states = states
        prediction, self.recurrent_states = self.network(states)
        # self.smh = [s.detach() for s in self.smh]

        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1))).to(device)
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        entropy_loss = entropy.mean()

        self.writer.add_scalar('advantages', advantages.mean(), self.total_steps)
        self.writer.add_scalar('policy_loss', policy_loss, self.total_steps)
        self.writer.add_scalar('value_loss', value_loss, self.total_steps)
        self.writer.add_scalar('entropy_loss', entropy_loss, self.total_steps)

        start = time.time()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss +
         config.value_loss_weight * value_loss).backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.writer.add_scalar('grad_norm', grad_norm, self.total_steps)
        self.optimizer.step()

        end = time.time()
        self.writer.add_scalar('backwards_pass_time', end-start, self.total_steps)
        # [rs.detach_() for rs in self.recurrent_states]
        # self.recurrent_states = [rs.detach() for rs in self.recurrent_states]
