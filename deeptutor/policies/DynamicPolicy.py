import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from garage.torch import np_to_torch
from garage.torch.modules import MLPModule
from garage.torch.policies.policy import Policy
from garage.torch.policies.stochastic_policy import StochasticPolicy


class DynamicPolicy(StochasticPolicy):
    def __init__(self, env, name="dynamic_policy", hidden_sizes=(32,), **kwargs):
        super(DynamicPolicy, self).__init__(env.spec, name)
        self.env = env
        self._env_spec = env.spec
        self._name = name
        self._obs_dim = self._env_spec.observation_space.flat_dim
        self._module = MLPModule(
            input_dim=self._obs_dim, hidden_sizes=hidden_sizes, output_dim=1, **kwargs
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors.
                Do not need to be detached, and can be on any device.
        """
        if len(observations.size()) == 2:
            observations = observations.unsqueeze(0)
        priorities = torch.ones((observations.shape[0], self.env.n_items))
        batch_idx = list(range(observations.size(0)))
        for seq in range(observations.size(1)):
            step = observations[:, seq, :]
            out = self._module(step).squeeze()
            idx = [int(step[i][0]) for i in range(len(step))]
            priorities[batch_idx, idx] = out
        logits = self.softmax(priorities)
        return Categorical(logits=logits), {}
