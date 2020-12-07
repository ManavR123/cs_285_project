import copy

import numpy as np

from deeptutor.envs.StudentEnv import StudentEnv
from deeptutor.infrastructure.utils import (
    sample_delay_coef,
    sample_item_decay_exps,
    sample_item_difficulties,
    sample_student_ability,
    sample_student_decay_exp,
    sample_window_cw,
    sample_window_nw,
)


class DASHEnv(StudentEnv):
    def __init__(
        self,
        n_windows=5,
        item_difficulties=None,
        student_ability=None,
        window_cw=None,
        window_nw=None,
        item_decay_exps=None,
        student_decay_exp=None,
        delay_coef=None,
        **kwargs
    ):
        super(DASHEnv, self).__init__(**kwargs)

        if item_difficulties is None:
            self.item_difficulties = sample_item_difficulties(self.n_items)
        else:
            if len(item_difficulties) != self.n_items:
                raise ValueError
            self.item_difficulties = item_difficulties

        if student_ability is None:
            self.student_ability = sample_student_ability()
        else:
            self.student_ability = student_ability

        if item_decay_exps is None:
            self.item_decay_exps = sample_item_decay_exps(self.n_items)
        else:
            if len(item_decay_exps) != self.n_items:
                raise ValueError
            self.item_decay_exps = item_decay_exps

        if student_decay_exp is None:
            self.student_decay_exp = sample_student_decay_exp()
        else:
            self.student_decay_exp = student_decay_exp

        if delay_coef is None:
            self.delay_coef = sample_delay_coef()
        else:
            self.delay_coef = delay_coef

        if self.n_steps % n_windows != 0:
            raise ValueError

        self.n_windows = n_windows
        self.window_size = self.n_steps // self.n_windows
        self.n_correct = None
        self.n_attempts = None

        if window_cw is None:
            window_cw = sample_window_cw(self.n_windows)
        if window_nw is None:
            window_nw = sample_window_nw(self.n_windows)
        if len(window_cw) != n_windows or len(window_nw) != n_windows:
            raise ValueError

        self.window_cw = np.tile(window_cw, self.n_items).reshape(
            (self.n_items, self.n_windows)
        )
        self.window_nw = np.tile(window_nw, self.n_items).reshape(
            (self.n_items, self.n_windows)
        )

        self.init_tlasts = np.exp(np.random.normal(0, 0.01, self.n_items))
        self.init_params()

    def init_params(self):
        self.n_correct = np.zeros((self.n_items, self.n_windows))
        self.n_attempts = np.zeros((self.n_items, self.n_windows))
        # self.tlasts = np.ones(self.n_items) * -sys.maxsize
        self.tlasts = copy.deepcopy(self.init_tlasts)

    def current_window(self):
        return min(self.n_windows - 1, self.curr_step // self.window_size)

    def recall_likelihoods(self):
        curr_window = self.current_window()
        study_histories = (
            self.window_cw[:, :curr_window]
            * np.log(1 + self.n_correct[:, :curr_window])
            + self.window_nw[:, :curr_window]
            * np.log(1 + self.n_attempts[:, :curr_window])
        ).sum(axis=1)
        m = 1 / (
            1
            + np.exp(-(self.student_ability - self.item_difficulties + study_histories))
        )
        f = np.exp(self.student_decay_exp - self.item_decay_exps)
        delays = self.now - self.tlasts
        return m / (1 + self.delay_coef * delays) ** f

    def update_model(self, item, outcome, timestamp, delay):
        curr_window = self.current_window()
        if outcome == 1:
            self.n_correct[item, curr_window] += 1
        self.n_attempts[item, curr_window] += 1
        self.tlasts[item] = timestamp

    def reset(self):
        self.init_params()
        return super(DASHEnv, self).reset()
