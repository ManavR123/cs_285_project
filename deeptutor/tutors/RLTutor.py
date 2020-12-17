from deeptutor.tutors.Tutor import Tutor


class RLTutor(Tutor):
    def __init__(self, init_timestamp=0, **kwargs):
        super(RLTutor, self).__init__(**kwargs)
        self.algo = None
        self.curr_obs = None

    def train(self, gym_env, n_eps=10, seed=0):
        raise NotImplementedError

    def reset(self):
        self.curr_obs = None
        self.algo.reset()

    def _next_item(self):
        if self.curr_obs is None:
            raise ValueError
        return self.algo.get_action(self.curr_obs)[0]

    def _update(self, obs):
        self.curr_obs = self.vectorize_obs(obs)

    def act(self, obs):
        self._update(obs)
        return self._next_item()

    def reset(self):
        self.algo.reset()
