import cv2
import random
import time
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random
import gym
import math

from model import Model
from buffer import ReplayBuffer
from CustomTensorboard import ModifiedTensorBoard
from env_wrapper import VectorEnv


class Worker:
    def __init__(
            self,
            env,
            test_env,
            args,
            env_params
            ):
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.args = args
        self.device = torch.device(
                "cuda:0" if self.args.cuda else "cpu")
        self.model = Model(self.env_params).to(self.device)
        self.target_model = Model(self.env_params).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.0005)
        self.replay_buffer = ReplayBuffer(
                self.args.buffer_size,
                self.env_params)
        self.tensorboard = ModifiedTensorBoard(f"{time.time()}")
        self.run()

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
        return action.argmax()

    def noisey_action(self, state):
        self.model_eval()
        with torch.no_grad():
            state = torch.tensor(state).to(self.device)
            action = self.model(state).detach().cpu()
            probs = F.softmax(action, 0)
            probs = torch.distributions.Categorical(probs)
        return np.array(probs.sample())

    def update_model(self, episode):
        self.model_train()
        losses = []
        for _ in range(self.args.num_envs):
            state, action, reward, next_state, done =\
                    self.replay_buffer.sample_buffer(self.args.batch_size)
            state = torch.tensor(state, device=self.device)
            action = torch.tensor(action, device=self.device)
            reward = torch.tensor(reward, device=self.device)
            next_state = torch.tensor(next_state, device=self.device)
            done = torch.tensor(done, device=self.device)

            predicted_q = self.model.forward(state)[
                    np.arange(0, self.args.batch_size),
                    action.long()]
            with torch.no_grad():
                next_state_q = self.model.forward(next_state)
                best_action = torch.argmax(next_state_q, axis=1)
                q_next = self.target_model.forward(next_state)[
                        np.arange(0, self.args.batch_size),
                        best_action]
                target_q = reward + (1 - done) * self.args.gamma * q_next
                target_q = target_q.float()

            loss = F.mse_loss(predicted_q, target_q)

            self.optim.zero_grad()
            # Compute gradients
            loss.backward()

            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optim.step()
            losses.append(loss.item())

        final_loss = np.mean(losses)
        self.tensorboard.update_stats(Loss=final_loss)
        self.soft_update_target(self.model, self.target_model)

    def soft_update_target(self, source, target):
        for target_param, param in zip(
                                        target.parameters(),
                                        source.parameters()):
            target_param.data.copy_(
                    (1 - self.args.polyak) * param.data +
                    self.args.polyak * target_param.data
                    )

    def run(self):
        for episode in range(self.args.episodes):
            start_time = time.time()
            for cycle in range(self.args.n_cycles):
                done = False
                state = self.env.reset()
                for _ in range(self.env_params['max_timestep']):
                    action = self.noisey_action(state)
                    new_state, reward, done, info = self.env.step(action)
                    self.replay_buffer.store_transition(
                            state,
                            action,
                            reward / self.env_params['reward_range'],
                            new_state,
                            done)
                    state = new_state
            self.update_model(episode)
            avg_reward = self.evaluate(episode)
            elapsed_time = time.time() - start_time
            print(
                f"Epoch {episode} of total of {self.args.episodes +1}",
                f"epochs, average reward is: {avg_reward}.",
                f"Elapsedtime: {int(elapsed_time /60)} minutes ",
                f"{int(elapsed_time %60)} seconds")
            self.tensorboard.step = episode

    def evaluate(self, episode):
        self.model_eval()
        total_rewards = []
        total_steps = []
        for _ in range(self.args.num_eval_eps):
            avg_reward = 0
            step = 0
            done = False
            state = self.test_env.reset()
            ep_reward = 0
            for _ in range(self.env_params['max_timestep']):
                action = self.greedy_action(state)
                new_state, reward, done, _ = self.test_env.step(action)
                ep_reward += reward
                state = new_state
                step += 1
                if done:
                    break
            total_steps.append(step)
            total_rewards.append(ep_reward)
        avg_reward = np.mean(total_rewards)
        min_reward = np.min(total_rewards)
        max_reward = np.max(total_rewards)
        episode_length = np.mean(total_steps)
        self.tensorboard.update_stats(
                AvgReward=avg_reward,
                MinReward=min_reward,
                MaxReward=max_reward,
                EpisodeLength=episode_length)
        return int(avg_reward)


def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def preprocess_depth_map(normalized_depth):
    logdepth = np.ones(normalized_depth.shape) + \
        (np.log(normalized_depth) / 5.70378)
    logdepth = np.clip(logdepth, 0.0, 1.0)
    return logdepth


