import cv2
import random
import time
import numpy as np
import time
import torch
import random

from Environment.carla_environment_wrapper import CarlaEnvironmentWrapper
from model import Model
from buffer import ReplayBuffer
from CustomTensorboard import ModifiedTensorBoard


class DQNAgent:
    def __init__(self, args, env_params):
        self.env = CarlaEnvironmentWrapper(cameras='Depth')
        self.env_params = env_params
        self.args = args
        self.device = torch.device(actor_optim)
        self.device = torch.device(
                "cuda:0" if self.args.cuda else "cpu")

        self.replay_buffer = ReplayBuffer(max_mem=1e6, self.env_params)
        self.model = Model(self.env_params).to(self.device)
        self.target_model = Model(self.env_params).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.tensorboard = ModifiedTensorBoard(f"{time.time()}")

        self.epsilon = self.args.epsilon

    def model_eval(self):
        self.model.eval()
        self.target_model.eval()

    def model_train(self):
        self.model.train()
        self.target_model.train()

    def greedy_action(self, state):
        self.model_eval()
        with torch.no_grad():
            state = torch.tensor(state).to(self.device)
            action = self.model(state).detach().cpu().numpy()
        return action.unsqueeze()

    def noisey_action(self, state):
        if self.epsilon > random.random():
            return random.randint(self.env_params['actions'])
        else:
            with torch.no_grad():
                state = torch.tensor(state).to(self.device)
                action = self.target_model(state).detach().cpu().numpy()
            return action.unsqueeze()

    def train(self):
        pass

    def explore(self):
        pass

    def store_transition(self, transition):
        pass


def preprocess_depth_map(normalized_depth):
    logdepth = np.ones(normalized_depth.shape) + \
        (np.log(normalized_depth) / 5.70378)
    logdepth = np.clip(logdepth, 0.0, 1.0)
    # Expand to three colors.
    return logdepth

# env1 = CarlaEnv(cameras=['Depth'])
# env2 = CarlaEnv(cameras=['Depth'])
# while True:
    # done = False
    # obs = env1.reset()
    # obs = env2.reset()

    # while not done:
        # random_action1 = random.randint(0,8)
        # random_action2 = random.randint(0,8)
        # new_obs1, reward, done, info = env1.step(random_action1)
        # new_obs2, reward, done, info = env2.step(random_action2)
        # cv2.imshow("1", preprocess_depth_map(new_obs1['depth_map']))
        # cv2.imshow("2", preprocess_depth_map(new_obs2['depth_map']))
        # cv2.waitKey(0)


