import argparse
import numpy as np
import random
import torch
import torch.multiprocessing as mp

from models.dqn_model import DQNModel
from agents.dqn_worker import DQNWorker
from args import get_args
from Environment.carla_environment_wrapper import CarlaEnvironmentWrapper
import gym


def get_params():
    env = gym.make('CartPole-v0')
    params = {
            'observation': env.reset().shape,
            'action': 2,
            'max_timestep': env._max_episode_steps
            }
    return params

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    args = get_args()
    env_params = get_params()
    shared_model = DQNModel(env_params)
    shared_model.share_memory()
    lock = mp.Lock()
    manager = mp.Manager()
    global_results = manager.dict()
    global_results['rewards'] = [manager.list() for x in range(args.episodes)]
    global_results['loss'] = [manager.list() for x in range(args.episodes)]
    global_results['episode_length'] = [
            manager.list() for x in range(args.episodes)
            ]

    DQNWorker(
            shared_model,
            lock,
            global_results,
            0,
            args,
            env_params)

    # processes = [
            # mp.Process(target=DQNWorker, args=(
                # shared_model,
                # lock,
                # global_results,
                # idx,
                # args,
                # env_params))
            # for idx in range(args.num_workers)
            # ]
    # [p.start() for p in processes]
    # [p.join() for p in processes]
