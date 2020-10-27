import os

import gym
from garage.envs import GymEnv

from deeptutor.infrastructure import logger


class MyGymEnv(GymEnv):
    def __init__(
        self,
        env,
        record_video=False,
        video_schedule=None,
        log_dir=None,
        record_log=False,
        force_reset=False,
    ):
        super().__init__(env)
        self.env = env
        self.env_id = ""

        if log_dir is None or record_log is False:
            self.monitoring = False

        # self._observation_space = convert_gym_space(env.observation_space)
        # logger.log("observation space: {}".format(self._observation_space))
        # self._action_space = convert_gym_space(env.action_space)
        # logger.log("action space: {}".format(self._action_space))
        self._horizon = self.env.n_steps
        self._log_dir = log_dir
        self._force_reset = force_reset
