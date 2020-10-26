class Tutor(object):
    def __init__(self):
        pass

    def _next_item(self):
        raise NotImplementedError

    def _update(self, item, outcome, timestamp, delay):
        raise NotImplementedError

    def act(self, obs):
        self._update(*list(obs))
        return self._next_item()

    def learn(self, r):
        pass

    def train(self, env, n_eps=10):
        return run_eps(self, env, n_eps=n_eps)

    def reset(self):
        raise NotImplementedError
