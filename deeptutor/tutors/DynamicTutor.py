import torch

from deeptutor.envs.MyGymEnv import MyGymEnv
from deeptutor.policies.DynamicPolicy import DynamicPolicy
from deeptutor.policies.LoggedPPOTorch import LoggedPPO
from deeptutor.tutors.RLTutor import RLTutor
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer


class DynamicTutor(RLTutor):
    def __init__(self, **kwargs):
        super(DynamicTutor, self).__init__(**kwargs)

    def train(self, gym_env, n_eps=10, seed=0):
        @wrap_experiment(archive_launch_repo=False, snapshot_mode="none")
        def train_dynamic(ctxt=None):
            set_seed(seed)
            trainer = Trainer(ctxt)
            env = MyGymEnv(gym_env, max_episode_length=100)
            policy = DynamicPolicy(env)
            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                worker_class=FragmentWorker,
            )
            value_function = GaussianMLPValueFunction(env_spec=env.spec)
            self.algo = LoggedPPO(
                env=env,
                env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                sampler=sampler,
                discount=0.99,
                center_adv=False,
            )
            trainer.setup(self.algo, env)
            trainer.train(n_epochs=n_eps, batch_size=4000)

            return self.algo.rew_chkpts

        return train_dynamic()
