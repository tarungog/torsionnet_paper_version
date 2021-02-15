import gym
import numpy as np
import logging


# documentation for SubprocVecEnv: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from .dummy_vec_env import DummyVecEnv

from ..utils import mkdir, random_seed

class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    # def change_level(self, level):
    #     return self.env.change_level(level)

    def reset(self):
        return self.env.reset()


def make_env(env_id, seed, rank, episode_life=True):
    def _thunk():
        random_seed(seed + rank)
        env = gym.make(env_id)
        env = OriginalReturnWrapper(env)
        return env

    return _thunk


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)

        logging.info(f'seed is {seed}')

        envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
            self.env = DummyVecEnv(envs)
        else:
            self.env = SubprocVecEnv(envs)
        self.name = name

    def change_level(self, xyz):
        self.env_method('change_level', xyz)

    def env_method(self, method_name, xyz):
        return self.env.env_method(method_name, xyz)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)