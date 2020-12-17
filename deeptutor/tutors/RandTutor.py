import numpy as np

from deeptutor.tutors.Tutor import Tutor


class RandTutor(Tutor):
    """sample item uniformly at random"""

    def __init__(self, init_timestamp=0, **kwargs):
        super(RandTutor, self).__init__(**kwargs)

    def next_item(self):
        return np.random.choice(range(self.n_items))

    def update(self, item, outcome, timestamp, delay):
        pass

    def reset(self):
        pass
