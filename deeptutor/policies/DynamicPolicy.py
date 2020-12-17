import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from garage.torch.policies.policy import Policy
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.torch.modules import MLPModule
from garage.torch import np_to_torch



class DynamicPolicy(Policy):
    def __init__(self, env, name="dynamic_policy", hidden_sizes=(32,), **kwargs):
        super(DynamicPolicy, self).__init__(env.spec, name)
        self.env = env
        self._env_spec = env.spec
        self._name = name
        self._obs_dim = self._env_spec.observation_space.flat_dim
        self._module = MLPModule(
            input_dim=self._obs_dim, hidden_sizes=hidden_sizes, output_dim=1, **kwargs
        )
        self.priorities = torch.zeros((env.n_items, 4))
        self.softmax = nn.Softmax(dim=0)

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
        self.update_priorities(observations)
        out = self._module(self.priorities).squeeze()
        logits = self.softmax(out)
        return Categorical(logits=logits.unsqueeze(0)), {}

    def get_action(self, observation):
        """Get a single action given an observation.

        Args:
            observation (np.ndarray): Observation with shape :math:`(O, )`.

        Returns:
            torch.Tensor: Predicted action with shape :math:`(A, )`.
            dict: Empty since this policy does not produce a distribution.
        """
        observation = np.expand_dims(observation, axis=0)
        self.update_priorities(observation)
        act, dist = self.get_actions(observation)
        return act[0], dist

    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (np.ndarray): Batch of observations, should
                have shape :math:`(N, O)`.

        Returns:
            torch.Tensor: Predicted actions. Tensor has shape :math:`(N, A)`.
            dict: Empty since this policy does not produce a distribution.
        """
        self.update_priorities(np_to_torch(observations))
        with torch.no_grad():
            return self(self.priorities)[0].sample().cpu().numpy(), dict()
    
    def update_priorities(self, observations):
        self.priorities = torch.cat((
            self.priorities, 
            torch.zeros((self.env.n_items - len(self.priorities), 4))
        ), dim=0)
        try:
            idx = [int(i) for i in observations[:,0]]
        except:
            import pdb; pdb.set_trace()
        self.priorities[idx] = observations
