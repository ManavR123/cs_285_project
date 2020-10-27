import copy

import numpy as np

from deeptutor.envs.StudentEnv import StudentEnv
from deeptutor.infrastructure.utils import sample_item_decay_rates


class EFCEnv(StudentEnv):
    """exponential forgetting curve"""

    def __init__(self, item_decay_rates=None, **kwargs):
        super(EFCEnv, self).__init__(**kwargs)

        if item_decay_rates is None:
            self.item_decay_rates = sample_item_decay_rates(self.n_items)
        else:
            self.item_decay_rates = item_decay_rates

        self.tlasts = None
        self.strengths = None
        self.init_tlasts = np.exp(np.random.normal(0, 1, self.n_items))
        self._init_params()

    def _init_params(self):
        # self.tlasts = np.ones(self.n_items) * -sys.maxsize
        self.tlasts = copy.deepcopy(self.init_tlasts)
        self.strengths = np.ones(self.n_items)

    def _recall_likelihoods(self):
        return np.exp(
            -self.item_decay_rates * (self.now - self.tlasts) / self.strengths
        )

    def _update_model(self, item, outcome, timestamp, delay):
        # self.strengths[item] = max(1, self.strengths[item] + 2 * outcome - 1) # fictional Leitner system
        self.strengths[item] += 1  # num attempts
        self.tlasts[item] = timestamp

    def _reset(self):
        self._init_params()
        return super(EFCEnv, self)._reset()
