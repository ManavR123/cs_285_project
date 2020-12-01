from deeptutor.tutors.Tutor import Tutor

class DummyTutor(Tutor):
    def __init__(self, policy):
        self.policy = policy

    def act(self, obs):
        return self.policy(obs)

    def reset(self):
        pass
