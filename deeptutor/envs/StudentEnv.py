import sys

import gym
import numpy as np
from gym import spaces

from deeptutor.infrastructure.utils import sample_const_delay


class StudentEnv(gym.Env):
    def __init__(
        self,
        n_items=10,
        n_steps=100,
        discount=1.0,
        sample_delay=None,
        reward_func="likelihood",
    ):
        if sample_delay is None:
            self.sample_delay = sample_const_delay(1)
        else:
            self.sample_delay = sample_delay
        self.curr_step = None
        self.n_steps = n_steps
        self.n_items = n_items
        self.now = 0
        self.curr_item = 0
        self.curr_outcome = None
        self.curr_delay = None
        self.discount = discount
        self.reward_func = reward_func

        self.action_space = spaces.Discrete(n_items)
        self.observation_space = spaces.Box(
            np.zeros(4), np.array([n_items - 1, 1, sys.maxsize, sys.maxsize])
        )

    def _recall_likelihoods(self):
        raise NotImplementedError

    def _recall_log_likelihoods(self, eps=1e-9):
        return np.log(eps + self._recall_likelihoods())

    def _update_model(self, item, outcome, timestamp, delay):
        raise NotImplementedError

    def _obs(self):
        timestamp = self.now - self.curr_delay
        return np.array(
            [self.curr_item, self.curr_outcome, timestamp, self.curr_delay], dtype=int
        )

    def _rew(self):
        if self.reward_func == "likelihood":
            return self._recall_likelihoods().mean()
        elif self.reward_func == "log_likelihood":
            return self._recall_log_likelihoods().mean()
        else:
            raise ValueError

    def _step(self, action):
        if self.curr_step is None or self.curr_step >= self.n_steps:
            raise ValueError

        if action < 0 or action >= self.n_items:
            raise ValueError

        self.curr_item = action
        self.curr_outcome = (
            1 if np.random.random() < self._recall_likelihoods()[action] else 0
        )

        self.curr_step += 1
        self.curr_delay = self.sample_delay()
        self.now += self.curr_delay

        self._update_model(self.curr_item, self.curr_outcome, self.now, self.curr_delay)

        obs = self._obs()
        r = self._rew()
        done = self.curr_step == self.n_steps
        info = {}

        return obs, r, done, info

    def _reset(self):
        self.curr_step = 0
        self.now = 0
        return self._step(np.random.choice(range(self.n_items)))[0]
