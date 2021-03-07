import argparse
import numpy as np
import random
import torch
import torch.multiprocessing as mp

from model import Model
from worker import Worker
from args import get_args
from Environment.carla_environment_wrapper import CarlaEnvironmentWrapper
import gym
from env_wrapper import VectorEnv


def get_params(make_env_fn):
    env = make_env_fn()
    params = {
            'observation': env.reset().shape,
            'action': env.action_space.n ,
            'max_timestep': env._max_episode_steps,
            'reward_range': 200
            }
    env.close()
    return params

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    args = get_args()

    make_env_fn = lambda: gym.make('LunarLander-v2')
    # make_env_fn = lambda: gym.make('CartPole-v1')
    vec_envs = VectorEnv(make_env_fn, n=args.num_envs)
    seeds = [random.randint(0, 9999) for _ in range(args.num_envs)]
    vec_envs.seed(seeds)

    test_env = make_env_fn()

    env_params = get_params(make_env_fn)

    Worker(vec_envs, test_env, args, env_params)

