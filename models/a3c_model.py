import numpy as np
import torch
import torch.functional as F
import torch.nn as nn


class A3CModel(nn.Module):
    def __init__(self, env_params):
        super(A3CModel, self).__init__()
        w, h = env_params['input_shape']
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        self.gru = nn.GRU(convh * convw * 32, 256)

        self.v_fc1 = nn.Linear(features=256, out_features=128)
        self.v_fc2 = nn.Linear(
                in_features=128,
                out_features=1)
        self.p_fc1 = nn.Linear(features=256, out_features=128)
        self.p_fc2 = nn.Linear(
                in_features=128,
                out_features=env_params['actions'])

    def forward(self, inputs):
        x, hx = inputs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        hx = self.gru(x, hx)
        x = hx

        v = F.relu(self.v_fc1(x))
        v = F.relu(self.v_fc2(x))

        p = F.relu(self.p_fc1(x))
        p = F.relu(self.p_fc2(x))
        return v, p, hx
