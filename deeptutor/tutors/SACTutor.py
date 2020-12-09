from torch.nn import functional as F

from deeptutor.envs.MyGymEnv import MyGymEnv
from deeptutor.policies.LoggedTRPO import LoggedTRPO
from deeptutor.tutors.RLTutor import RLTutor
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.envs import GymEnv
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
from garage.torch import prefer_gpu, global_device
from garage.torch.policies import GaussianMLPPolicy

class SACTutor(RLTutor):
    def __init__(self, n_items, init_timestamp=0):
        super().__init__(n_items)
    
    def train(self, gym_env, n_eps=10, seed=0):
        prefer_gpu()
        @wrap_experiment(archive_launch_repo=False, snapshot_mode="none")
        def train_trpo(ctxt=None):
            trainer = Trainer(ctxt)
            env = MyGymEnv(gym_env, max_episode_length=100)
            policy = GaussianMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      )
            qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                        hidden_sizes=[256, 256],
                                        hidden_nonlinearity=F.relu)

            qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                        hidden_sizes=[256, 256],
                                        hidden_nonlinearity=F.relu)
            replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

            sampler = LocalSampler(agents=policy,
                                envs=env,
                                max_episode_length=env.spec.max_episode_length,
                                worker_class=FragmentWorker)
            self.algo = SAC(env_spec=env.spec,
                                policy=policy,
                                qf1=qf1,
                                qf2=qf2,
                                sampler=sampler,
                                gradient_steps_per_itr=1000,
                                max_episode_length_eval=100,
                                replay_buffer=replay_buffer,
                                min_buffer_size=1e4,
                                target_update_tau=5e-3,
                                discount=0.99,
                                buffer_batch_size=256,
                                reward_scale=1.,
                                steps_per_epoch=1).to(global_device())
            trainer.setup(algo=self.algo, env=env)
            trainer.train(n_epochs=n_eps, batch_size=4000)
            

            
                    