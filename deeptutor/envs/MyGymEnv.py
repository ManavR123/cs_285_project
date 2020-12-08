import os

import gym

from garage.envs import GymEnv


class MyGymEnv(GymEnv):
    def __init__(
        self,
        env,
        record_video=False,
        video_schedule=None,
        log_dir=None,
        record_log=False,
        force_reset=False,
        max_episode_length=None,
    ):
        super().__init__(env, max_episode_length=max_episode_length)
        self.env = env
        self.env_id = ""
        self.max_episode_length = max_episode_length

        if log_dir is None or record_log is False:
            self.monitoring = False

        self._horizon = self.env.n_steps
        self._log_dir = log_dir
        self._force_reset = force_reset
