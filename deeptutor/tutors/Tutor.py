from deeptutor.infrastructure.utils import *


class Tutor(object):
    def __init__(self, dynamic=False, n_items=10):
        self.dynamic = dynamic
        self.n_items = n_items

    def next_item(self):
        raise NotImplementedError

    def update(self, item, outcome, timestamp, delay):
        raise NotImplementedError

    def act(self, obs):
        self.update(*list(obs))
        return self.next_item()

    def learn(self, r):
        pass

    def train(self, env, n_eps=10):
        return run_eps(self, env, n_eps=n_eps)

    def reset(self):
        raise NotImplementedError

    def update_items(self, num_items):
        self.n_items = num_items
