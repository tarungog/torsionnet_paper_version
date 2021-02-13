import gym
import sys
import inspect


from . import graphenvironments
from . import tchain_envs
from .environment_wrapper import Task

clsmembers = inspect.getmembers(sys.modules['main.environments.graphenvironments'], inspect.isclass)
for i, j in clsmembers:
     if issubclass(j, gym.Env):
          gym.envs.register(
               id=f'{i}-v0',
               entry_point=f'main.environments.graphenvironments:{i}',
               max_episode_steps=1000,
          )

gym.envs.register(
     id='TestBestGibbs-v0',
     entry_point='main.environments.tchain_envs:TestBestGibbs',
     max_episode_steps=1000,
)

gym.envs.register(
     id='TChainTrain-v0',
     entry_point='main.environments.tchain_envs:TChainTrain',
     max_episode_steps=1000,
)

for i in range(0, 10):
     gym.envs.register(
          id=f'TChainTest-v{i}',
          entry_point='main.environments.tchain_envs:TChainTest',
          max_episode_steps=1000,
          kwargs={'ind_select': i}
     )

for i in range(0, 10):
     gym.envs.register(
          id=f'TChainTest2-v{i}',
          entry_point='main.environments.tchain_envs:TChainTest2',
          max_episode_steps=1000,
          kwargs={'ind_select': i}
     )

for i in range(0, 10):
     gym.envs.register(
          id=f'TChainTest3-v{i}',
          entry_point='main.environments.tchain_envs:TChainTest3',
          max_episode_steps=1000,
          kwargs={'ind_select': i}
     )


