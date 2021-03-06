import torch
import numpy as np
import pdb


class ReplayBuffer:
    def __init__(self, max_mem, env_params):
        self.max_mem = int(max_mem)
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.max_mem, *env_params['observation']))
        self.new_state_mem = np.zeros(
                (self.max_mem, *env_params['observation'])
                )
        self.action_mem = np.zeros((self.max_mem, 1))
        self.reward_mem = np.zeros(self.max_mem)
        self.done_mem = np.zeros(self.max_mem)

    def __len__(self):
        return min(self.mem_cntr, self.max_mem)

    def store_transition(self, state, action, reward, nextstate, done):
        index = self.mem_cntr % self.max_mem
        self.state_mem[index] = state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.new_state_mem[index] = nextstate
        self.done_mem[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_batch = min(self.mem_cntr, self.max_mem)
        batch = np.random.choice(max_batch, batch_size)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        nextstates = self.new_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, nextstates, dones
