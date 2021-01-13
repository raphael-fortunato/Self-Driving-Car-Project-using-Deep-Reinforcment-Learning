import numpy as np
import torch
import torch.functional as F
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, env_params):
        super(Model, self).__init__()

        # self.conv1 = nn.Conv2d(
                # in_channels=env_params['observation'][0],
                # out_channels=32,
                # kernel_size=8,
                # stride=4)
        # self.conv2 = nn.Conv2d(
                # in_channels=32,
                # out_channels=64,
                # kernel_size=4,
                # stride=2)
        # self.conv3 = nn.Conv2d(
                # in_channels=64,
                # out_channels=64,
                # kernel_size=3,
                # stride=1)

        self.fc1 = nn.Linear(in_features=env_params['observation'][0], out_features=128)
        self.fc2 = nn.Linear(
                in_features=128,
                out_features=env_params['action'])

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # def _get_conv_out(self, shape):
        # o = self.conv(torch.zeros(1, *shape).unsqueeze(0))
        # return int(np.prod(o.size()))
