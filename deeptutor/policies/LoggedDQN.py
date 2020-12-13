from deeptutor.infrastructure.utils import run_rl_ep
from deeptutor.tutors.DummyTutor import DummyTutor
from garage.torch.algos import DQN


class LoggedDQN(DQN):
    def __init__(self, env, *args, **kwargs):
        super(LoggedDQN, self).__init__(*args, **kwargs)
        self.env = env
        self.rew_chkpts = []

    def train(self, trainer):
        ret = None
        for _ in trainer.step_epochs():
            trainer.step_path = trainer.obtain_episodes(trainer.step_itr)
            ret = self._train_once(trainer.step_itr, trainer.step_path)
            my_policy = lambda obs: self.policy.get_action(obs)[0]
            r, _ = run_rl_ep(DummyTutor(my_policy), self.env)
            self.rew_chkpts.append(r)
            trainer.step_itr += 1
        return ret
