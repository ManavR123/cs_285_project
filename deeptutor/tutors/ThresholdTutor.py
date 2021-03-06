import copy

import numpy as np

from deeptutor.infrastructure.utils import *
from deeptutor.tutors.Tutor import Tutor


class ThresholdTutor(Tutor):
    """review item with recall likelihood closest to some threshold"""

    def __init__(self, n_items, env, init_timestamp=0):
        self.n_items = n_items
        self.threshold = None
        self.env = copy.deepcopy(env)
        self.env.reset()

    def next_item(self):
        return np.argmin(np.abs(self.env.recall_likelihoods() - self.threshold))

    def update(self, item, outcome, timestamp, delay):
        self.env.update_model(item, outcome, timestamp, delay)
        self.env.curr_step += 1
        self.env.now += delay

    def reset(self):
        self.env.reset()

    def train(self, env, n_eps=10):
        thresholds = np.arange(0, 1, 1 / n_eps)
        n_eps_per_thresh = n_eps // thresholds.size
        assert n_eps_per_thresh > 0
        best_reward = None
        best_thresh = None
        for thresh in thresholds:
            self.threshold = thresh
            reward = np.mean(run_eps(self, env, n_eps=n_eps_per_thresh))
            if best_reward is None or reward > best_reward:
                best_thresh = thresh
                best_reward = reward
        self.threshold = best_thresh
        return run_eps(self, env, n_eps=n_eps)
