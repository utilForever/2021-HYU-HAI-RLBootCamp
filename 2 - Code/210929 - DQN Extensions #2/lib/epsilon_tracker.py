from types import SimpleNamespace

import ptan


class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector, params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)
