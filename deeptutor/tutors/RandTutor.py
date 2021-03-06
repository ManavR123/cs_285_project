import numpy as np

from deeptutor.tutors.Tutor import Tutor


class RandTutor(Tutor):
    """sample item uniformly at random"""

    def __init__(self, n_items, init_timestamp=0):
        self.n_items = n_items

    def next_item(self):
        return np.random.choice(range(self.n_items))

    def update(self, item, outcome, timestamp, delay):
        pass

    def reset(self):
        pass
