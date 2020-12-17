import os

import gym

from garage.envs import GymEnv
import akro
from garage import Environment, EnvSpec, EnvStep, StepType


class MyGymEnv(GymEnv):
    def __init__(
        self,
        env,
        is_image=False,
        record_video=False,
        video_schedule=None,
        log_dir=None,
        record_log=False,
        force_reset=False,
        max_episode_length=None,
    ):
        super().__init__(env, max_episode_length=max_episode_length)
        self._env = env
        self.env_id = ""
        self.max_episode_length = max_episode_length

        if log_dir is None or record_log is False:
            self.monitoring = False

        self._horizon = self._env.n_steps
        self._log_dir = log_dir
        self._force_reset = force_reset
        self.is_image = is_image

    def step(self, action):
        env_step = super().step(action)
        self._action_space = akro.from_gym(self._env.action_space)
        self._observation_space = akro.from_gym(self._env.observation_space,
                                                is_image=self.is_image)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=self._max_episode_length)
        return env_step

