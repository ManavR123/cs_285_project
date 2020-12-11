import tensorflow as tf


from deeptutor.envs.MyGymEnv import MyGymEnv
from deeptutor.policies.LoggedTRPO import LoggedTRPO
from deeptutor.tutors.RLTutor import RLTutor
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.policies import CategoricalGRUPolicy
from garage.trainer import TFTrainer
from garage.tf.optimizers import FiniteDifferenceHVP


class GRUTRPOTutor(RLTutor):
    def __init__(self, n_items, init_timestamp=0):
        super().__init__(n_items)
    
    def train(self, gym_env, n_eps=10, seed=0):
        tf.compat.v1.reset_default_graph()
        @wrap_experiment(archive_launch_repo=False, snapshot_mode="none")
        def train_gru_trpo(ctxt=None):
            set_seed(seed)
            with TFTrainer(snapshot_config=ctxt) as trainer:
                env = MyGymEnv(gym_env, max_episode_length=100)
                policy = CategoricalGRUPolicy(name='policy',
                                      env_spec=env.spec,
                                      state_include_action=False
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
                    optimizer_args=dict(hvp_approach=FiniteDifferenceHVP(
                        base_eps=1e-5))
                )

                trainer.setup(self.algo, env)
                trainer.train(n_epochs=n_eps, batch_size=4000)
                return self.algo.rew_chkpts

        return train_gru_trpo()