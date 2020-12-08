import numpy as np
import tensorflow as tf
import torch

from deeptutor.envs.MyGymEnv import MyGymEnv
from deeptutor.policies.LoggedDQN import LoggedDQN
from deeptutor.policies.LoggedTRPO import LoggedTRPO
from deeptutor.tutors.Tutor import Tutor
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch.policies import DiscreteQFArgmaxPolicy, CategoricalGRUPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.trainer import Trainer, TFTrainer
from garage.tf.policies import CategoricalMLPPolicy
from garage.np.baselines import LinearFeatureBaseline

class RLTutor(Tutor):
    def __init__(self, n_items, init_timestamp=0):
        self.algo = None
        self.curr_obs = None

    def train(self, gym_env, algo="DQN", n_eps=10, seed=0):
        @wrap_experiment(archive_launch_repo=False, snapshot_mode="none")
        def train_dqn(ctxt=None):
            set_seed(seed)
            trainer = Trainer(ctxt)
            env = MyGymEnv(gym_env, max_episode_length=100)
            steps_per_epoch = 10
            sampler_batch_size = 4000
            num_timesteps = n_eps * steps_per_epoch * sampler_batch_size
            replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
            qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(8, 5))
            policy = DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)
            exploration_policy = EpsilonGreedyPolicy(
                env_spec=env.spec,
                policy=policy,
                total_timesteps=num_timesteps,
                max_epsilon=1.0,
                min_epsilon=0.01,
                decay_ratio=0.4,
            )
            sampler = LocalSampler(
                agents=exploration_policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                worker_class=FragmentWorker,
            )
            self.algo = LoggedDQN(
                env=env,
                env_spec=env.spec,
                policy=policy,
                qf=qf,
                exploration_policy=exploration_policy,
                replay_buffer=replay_buffer,
                sampler=sampler,
                steps_per_epoch=steps_per_epoch,
                qf_lr=5e-5,
                discount=0.99,
                min_buffer_size=int(1e4),
                n_train_steps=500,
                target_update_freq=30,
                buffer_batch_size=64,
            )
            trainer.setup(self.algo, env)
            trainer.train(n_epochs=n_eps, batch_size=sampler_batch_size)

            return self.algo.rew_chkpts

        @wrap_experiment(archive_launch_repo=False, snapshot_mode="none")
        def train_mlp_trpo(ctxt=None):
            set_seed(seed)
            with TFTrainer(snapshot_config=ctxt) as trainer:
                env = MyGymEnv(gym_env, max_episode_length=100)
                policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      )
                baseline = LinearFeatureBaseline(env_spec=env.spec)
                sampler = LocalSampler(
                    agents=policy,
                    envs=env,
                    max_episode_length=env.spec.max_episode_length,
                    worker_class=FragmentWorker,
                )
                self.algo = LoggedTRPO(
                    env=env,
                    env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=0.99,
                    max_kl_step=0.01,
                )

                trainer.setup(self.algo, env)
                trainer.train(n_epochs=n_eps, batch_size=4000)

        if algo == "DQN":
            return train_dqn()
        if algo == "MLP_TRPO":
            return train_mlp_trpo()
        raise ValueError(f"{algo} is not suppported")

    def reset(self):
        self.curr_obs = None
        self.algo.reset()

    def _next_item(self):
        if self.curr_obs is None:
            raise ValueError
        return self.algo.get_action(self.curr_obs)[0]

    def _update(self, obs):
        self.curr_obs = self.vectorize_obs(obs)

    def act(self, obs):
        self._update(obs)
        return self._next_item()

    def reset(self):
        self.algo.reset()
