import numpy as np
import tensorflow as tf

from garage.tf.policies import CategoricalGRUPolicy
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline

from deeptutor.envs.MyGymEnv import MyGymEnv
from deeptutor.tutors.Tutor import Tutor
from deeptutor.policies.LoggedTRPO import LoggedTRPO


class RLTutor(Tutor):
    def __init__(self, n_items, init_timestamp=0):
        self.raw_policy = None
        self.curr_obs = None

    def train(self, gym_env, n_eps=10):
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()

        with sess.as_default():
            env = MyGymEnv(gym_env, max_episode_length=None)
            policy = CategoricalGRUPolicy(env_spec=env.spec, hidden_dim=32, state_include_action=False)
            self.raw_policy = LoggedTRPO(
                env_spec=env,
                policy=policy,
                baseline=LinearFeatureBaseline(env_spec=env.spec),
                discount=0.99
                # batch_size=4000,
                # max_path_length=env.env.n_steps,
                # n_itr=n_eps,
                # step_size=0.01,
                # verbose=False
            )
            self.raw_policy.train()

        sess.close()
        return self.raw_policy.rew_chkpts

    def reset(self):
        self.curr_obs = None
        self.raw_policy.reset()

    def _next_item(self):
        if self.curr_obs is None:
            raise ValueError
        return self.raw_policy.get_action(self.curr_obs)[0]

    def _update(self, obs):
        self.curr_obs = self.vectorize_obs(obs)

    def act(self, obs):
        self._update(obs)
        return self._next_item()

    def reset(self):
        self.raw_policy.reset()
