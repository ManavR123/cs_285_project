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
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log(
                    "Warning: skipping Gym environment monitoring since snapshot_dir not configured."
                )
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        self.env = env
        self.env_id = ""

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(
                self.env, log_dir, video_callable=video_schedule, force=True
            )
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = self.env.n_steps
        self._log_dir = log_dir
        self._force_reset = force_reset
