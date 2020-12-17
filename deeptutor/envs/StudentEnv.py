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
        dynamic=False,
        add_rate=0,
    ):
        if sample_delay is None:
            self.sample_delay = sample_const_delay(1)
        else:
            self.sample_delay = sample_delay
        self.curr_step = None
        self.n_steps = n_steps
        self.orig_n_items = n_items
        self.n_items = n_items
        self.now = 0
        self.curr_item = 0
        self.curr_outcome = None
        self.curr_delay = None
        self.discount = discount
        self.reward_func = reward_func
        self.dynamic = dynamic
        self.add_rate = add_rate

        if self.dynamic and self.add_rate <= 0:
            raise ValueError(
                "Add rate must be greater than 0 when environment is dynamic"
            )

        self.action_space = spaces.Discrete(n_items)
        self.observation_space = spaces.Box(
            np.zeros(4), np.array([n_items - 1, 1, sys.maxsize, sys.maxsize])
        )

    def recall_likelihoods(self):
        raise NotImplementedError

    def recall_log_likelihoods(self, eps=1e-9):
        return np.log(eps + self.recall_likelihoods())

    def update_model(self, item, outcome, timestamp, delay):
        raise NotImplementedError

    def obs(self):
        timestamp = self.now - self.curr_delay
        return np.array(
            [self.curr_item, self.curr_outcome, timestamp, self.curr_delay], dtype=int
        )

    def rew(self):
        if self.reward_func == "likelihood":
            return self.recall_likelihoods().mean()
        elif self.reward_func == "log_likelihood":
            return self.recall_log_likelihoods().mean()
        else:
            raise ValueError

    def step(self, action):
        if self.curr_step is None or self.curr_step >= self.n_steps:
            raise ValueError

        if action < 0 or action >= self.n_items:
            raise ValueError

        self.curr_item = action
        self.curr_outcome = (
            1 if np.random.random() < self.recall_likelihoods()[action] else 0
        )

        self.curr_step += 1
        self.curr_delay = self.sample_delay()
        self.now += self.curr_delay

        if self.dynamic and self.curr_step and self.curr_step % self.add_rate == 0:
            self.update_items()

        self.update_model(self.curr_item, self.curr_outcome, self.now, self.curr_delay)

        obs = self.obs()
        r = self.rew()
        done = self.curr_step == self.n_steps
        info = {}

        return obs, r, done, info

    def reset(self):
        self.curr_step = 0
        self.now = 0
        self.n_items = self.orig_n_items
        self.action_space = spaces.Discrete(self.n_items)
        self.observation_space = spaces.Box(
            np.zeros(4), np.array([self.n_items - 1, 1, sys.maxsize, sys.maxsize])
        )
        return self.step(np.random.choice(range(self.n_items)))[0]

    def update_items(self):
        self.n_items = int(self.n_items * 1.1)
        self.action_space = spaces.Discrete(self.n_items)
        self.observation_space = spaces.Box(
            np.zeros(4), np.array([self.n_items - 1, 1, sys.maxsize, sys.maxsize])
        )
