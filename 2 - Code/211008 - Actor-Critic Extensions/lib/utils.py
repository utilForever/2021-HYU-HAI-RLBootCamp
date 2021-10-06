import sys
import time
import numpy as np


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
