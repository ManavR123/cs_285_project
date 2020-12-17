from garage.np.exploration_policies import EpsilonGreedyPolicy


class DynamicEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self, env, env_spec, policy, total_env_steps, **kwargs):
        super().__init__(env_spec, policy, total_env_steps, **kwargs)
        self.env = env

    def get_action(self, observation):
        self._env_spec = self.env.spec
        self._action_space = self._env_spec.action_space
        return super().get_action(observation)

    def get_actions(self, observations):
        self._env_spec = self.env.spec
        self._action_space = self._env_spec.action_space
        return super().get_action(observations)
