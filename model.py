import numpy as np
import torch
import torch.functional as F
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, env_params):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                env_params['observation'][0],
                32, kernel_size=5,
                padding=2,
                stride=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(256),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(
                in_features=self._get_conv_out(env_params['observation']),
                out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=env_params['actions'])
                )

    def __call__(self, x):
        x = self.conv(x)
        x = torch.flatten(x)
        x = self.fully_connected(x)
        return x

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
