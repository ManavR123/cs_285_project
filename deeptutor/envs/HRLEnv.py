import numpy as np
from StudentEnv import StudentEnv
from utils import sample_loglinear_coeffs


class HLREnv(StudentEnv):
    """exponential forgetting curve with log-linear memory strength"""

    def __init__(self, loglinear_coeffs=None, **kwargs):
        super(HLREnv, self).__init__(**kwargs)

        if loglinear_coeffs is None:
            self.loglinear_coeffs = sample_loglinear_coeffs(self.n_items)
        else:
            self.loglinear_coeffs = loglinear_coeffs
        assert self.loglinear_coeffs.size == 3 + self.n_items

        self.tlasts = None
        self.loglinear_feats = None
        self.init_tlasts = np.exp(np.random.normal(0, 1, self.n_items))
        self._init_params()

    def _init_params(self):
        # self.tlasts = np.ones(self.n_items) * -sys.maxsize
        self.tlasts = copy.deepcopy(self.init_tlasts)
        self.loglinear_feats = np.zeros(
            (self.n_items, 3)
        )  # n_attempts, n_correct, n_incorrect
        self.loglinear_feats = np.concatenate(
            (self.loglinear_feats, np.eye(self.n_items)), axis=1
        )

    def _strengths(self):
        return np.exp(np.einsum("j,ij->i", self.loglinear_coeffs, self.loglinear_feats))

    def _recall_likelihoods(self):
        return np.exp(-(self.now - self.tlasts) / self._strengths())

    def _update_model(self, item, outcome, timestamp, delay):
        self.loglinear_feats[item, 0] += 1
        self.loglinear_feats[item, 1 if outcome == 1 else 2] += 1
        self.tlasts[item] = timestamp

    def _reset(self):
        self._init_params()
        return super(HLREnv, self)._reset()
