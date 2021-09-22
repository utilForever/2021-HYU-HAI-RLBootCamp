import math

from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as f

import numpy as np

BETA_START = 0.4
BETA_FRAMES = 100000


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)

        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)

        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)

            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)

        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, x):
        self.epsilon_weight.normal_()

        bias = self.bias

        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        v = self.sigma_weight * self.epsilon_weight.data + self.weight

        return f.linear(x, v, bias)

    def _forward_unimplemented(self, *input_forward: Any) -> None:
        pass


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, num_actions)
        ]

        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]

    def _forward_unimplemented(self, *input_forward: Any) -> None:
        pass


class PrioritizedReplayBuffer:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size,), dtype=np.float32)
        self.beta = BETA_START

    def update_beta(self, idx):
        v = BETA_START + idx * (1.0 - BETA_START) / BETA_FRAMES
        self.beta = min(1.0, v)

        return self.beta

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_priority = self.priorities.max(initial=1.0) if self.buffer else 1.0

        for _ in range(count):
            sample = next(self.exp_source_iter)

            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample

            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probabilities = priorities ** self.prob_alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority
