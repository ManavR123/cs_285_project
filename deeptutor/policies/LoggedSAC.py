from deeptutor.infrastructure.utils import run_rl_ep
from deeptutor.tutors.DummyTutor import DummyTutor
from garage.torch.algos import SAC
from garage import log_performance, obtain_evaluation_episodes, StepType

import numpy as np


class LoggedSAC(SAC):
    def __init__(self, env, *args, **kwargs):
        super(LoggedSAC, self).__init__(*args, **kwargs)
        self.env = env
        self.rew_chkpts = []

    def train(self, trainer):
        self.to()
        for _ in trainer.step_epochs():
            for _ in range(self._steps_per_epoch):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                trainer.step_path = trainer.obtain_episodes(trainer.step_itr)
                trainer.step_path = trainer.obtain_samples(
                    trainer.step_itr, batch_size)
                path_returns = []
                for path in trainer.step_path:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'].reshape(-1, 1),
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=np.array([
                                 step_type == StepType.TERMINAL
                                 for step_type in path['step_types']
                             ]).reshape(-1, 1)))
                    path_returns.append(sum(path['rewards']))
                assert len(path_returns) == len(trainer.step_path)
                self.episode_rewards.append(np.mean(path_returns))
                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()
            last_return = self._evaluate_policy(trainer.step_itr)
            self._log_statistics(policy_loss, qf1_loss, qf2_loss)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)
            my_policy = lambda obs: self.policy.get_action(obs)[0]
            r, _ = run_rl_ep(DummyTutor(my_policy), self.env)
            self.rew_chkpts.append(r)
            trainer.step_itr += 1
