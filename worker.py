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

from Environment.carla_environment_wrapper import CarlaEnvironmentWrapper
from model import Model
from buffer import ReplayBuffer
from CustomTensorboard import ModifiedTensorBoard


class Worker:
    def __init__(
            self,
            shared_model,
            lock,
            global_results,
            worker_id,
            args,
            env_params
            ):
        # self.env = CarlaEnvironmentWrapper(cameras='Depth')
        self.env = gym.make('CartPole-v0')
        self.env_params = env_params
        self.args = args
        self.device = torch.device(
                "cuda:0" if self.args.cuda else "cpu")
        self.global_results = global_results
        self.worker_id = worker_id
        self.lock = lock
        self.shared_model = shared_model
        self.model = Model(self.env_params).to(self.device)
        self.model.load_state_dict(self.shared_model.state_dict())
        self.target_model = Model(self.env_params).to(self.device)
        self.target_model.load_state_dict(self.shared_model.state_dict())
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.0005)
        self.replay_buffer = ReplayBuffer(
                self.args.buffer_size,
                self.env_params)
        if self.worker_id == 0:
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
        return probs.sample().item()

    def copy_gradients(self, target, source):
        for shared_param, param in zip(
                target.parameters(),
                source.parameters()):
            if param.grad is not None:
                shared_param._grad = param.grad.clone().cpu()

    def update_model(self, episode):
        self.model_train()
        losses = []
        for _ in range(self.args.n_batches):
            state, action, reward, next_state, done =\
                    self.replay_buffer.sample_buffer(self.args.batch_size)
            state = torch.tensor(state, device=self.device)
            action = torch.tensor(action, device=self.device)
            reward = torch.tensor(reward, device=self.device)
            next_state = torch.tensor(next_state, device=self.device)
            done = torch.tensor(done, device=self.device)

            with torch.no_grad():
                next_action = self.target_model(next_state).detach().max(1)[0]
                target_q = reward + (1 - done) * self.args.gamma * next_action

            predicted_q = self.model(state).gather(1, action.long()).max(1)[0]

            loss = F.smooth_l1_loss(predicted_q.float(), target_q.float())

            self.optim.zero_grad()
            # Compute gradients
            loss.backward()

            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)

            # The critical section begins
            self.lock.acquire()
            # self.copy_gradients(self.shared_model, self.model)
            self.optim.step()
            self.lock.release()
            losses.append(loss.item())

        final_loss = np.sum(losses)
        self.lock.acquire()
        self.global_results['loss'][episode].append(final_loss)
        self.lock.release()
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
                            reward,
                            new_state,
                            done)
                    state = new_state
                    if done:
                        break
            self.update_model(episode)
            avg_reward = self.evaluate(episode)
            elapsed_time = time.time() - start_time
            if self.worker_id == 0:
                print(f"Epoch {episode} of total of {self.args.episodes +1}",
                        f"epochs, average reward is: {avg_reward}.",
                        f"Elapsedtime: {int(elapsed_time /60)} minutes ",
                        f"{int(elapsed_time %60)} seconds")
                self.tensorboard.step = episode

    def evaluate(self, episode):
        self.model_eval()
        avg_reward = 0
        step = 0
        done = False
        state = self.env.reset()
        ep_reward = 0
        for _ in range(self.env_params['max_timestep']):
            action = self.greedy_action(state)
            new_state, reward, done, _ = self.env.step(action)
            ep_reward += reward
            state = new_state
            step += 1
            if done:
                break
        self.lock.acquire()
        self.global_results['rewards'][episode].append(ep_reward)
        self.global_results['episode_length'][episode].append(step)
        self.lock.release()
        if self.worker_id == 0:
            avg_loss = np.mean(self.global_results['loss'][episode])
            avg_reward = np.mean(self.global_results['rewards'][episode])
            min_reward = np.min(self.global_results['rewards'][episode])
            max_reward = np.max(self.global_results['rewards'][episode])
            episode_length = np.mean(
                    self.global_results['episode_length'][episode]
                    )
            self.tensorboard.update_stats(
                    Loss=avg_loss,
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


