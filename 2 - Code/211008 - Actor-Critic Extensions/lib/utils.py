import sys
import time
import numpy as np

import torch
import torch.nn as nn

from typing import Any


class A2C(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)

    def _forward_unimplemented(self, *input_forward: Any) -> None:
        pass


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.ts = 0
        self.ts_frame = 0
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)

        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()

        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon

        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()

        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)

        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True

        return False


def unpack_batch(_batch, _net, last_val_gamma, _device='cpu'):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for idx, _exp in enumerate(_batch):
        states.append(np.array(_exp.state, copy=False))
        actions.append(int(_exp.action))
        rewards.append(_exp.reward)

        if _exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(_exp.last_state, copy=False))

    _states_v = torch.FloatTensor(np.array(states, copy=False)).to(_device)
    _actions_t = torch.LongTensor(actions).to(_device)

    rewards_np = np.array(rewards, dtype=np.float32)

    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(_device)
        last_vals_v = _net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(_device)

    return _states_v, _actions_t, ref_vals_v
