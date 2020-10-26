class Card(object):
    def __init__(self, _id):
        self.grade = 0
        self.next_rep = 0
        self.last_rep = 0
        self.easiness = 0
        self.acq_reps = 0
        self.acq_reps_since_lapse = 0
        self.ret_reps = 0
        self.ret_reps_since_lapse = 0
        self.lapses = 0
        self._id = _id
