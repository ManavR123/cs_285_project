import tensorflow as tf
import torch

from deeptutor.envs.MyGymEnv import MyGymEnv
from deeptutor.policies.LoggedPPO import LoggedPPO
from deeptutor.tutors.RLTutor import RLTutor
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.policies import CategoricalGRUPolicy
# from garage.torch.policies import CategoricalGRUPolicy
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer, TFTrainer
from garage.torch import prefer_gpu, global_device
from garage.tf.optimizers import FirstOrderOptimizer


class PPOTutor(RLTutor):
    def __init__(self, n_items, init_timestamp=0):
        super().__init__(n_items)
    
    def train(self, gym_env, n_eps=10, seed=0):
        @wrap_experiment(archive_launch_repo=False, snapshot_mode="none")
        def train_ppo(ctxt=None):
            set_seed(seed)
            with TFTrainer(ctxt) as trainer:
                env = MyGymEnv(gym_env, max_episode_length=100)
                policy = CategoricalGRUPolicy(name='policy', env_spec=env.spec, state_include_action=False)
                baseline = LinearFeatureBaseline(env_spec=env.spec)
                sampler = LocalSampler(
                    agents=policy,
                    envs=env,
                    max_episode_length=env.spec.max_episode_length,
                    worker_class=FragmentWorker,
                    is_tf_worker=True,
                )
                self.algo = LoggedPPO(
                    env=env,
                    env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=0.99,
                    center_adv=False,
                )

                trainer.setup(self.algo, env)
                trainer.train(n_epochs=n_eps, batch_size=4000)
                return self.algo.rew_chkpts

        return train_ppo()
