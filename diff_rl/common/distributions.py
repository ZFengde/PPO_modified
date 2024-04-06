import torch as th
from stable_baselines3.common.distributions import Distribution
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_action_dim
from torch.distributions import Normal
from diff_rl.common.diffusion_policy import Diffusion_Policy, Networks

def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

class DiagGaussianDistribution(Distribution):

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim, log_std_init: float = 0.0):

        mean_actions = nn.Linear(latent_dim, self.action_dim)
        # mean_actions = Diffusion_Policy(action_feat_dim=latent_dim, action_dim=self.action_dim, model=Networks)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions, log_std):
        
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions):
        
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self):
        return sum_independent_dims(self.distribution.entropy())

    def sample(self):
        return self.distribution.rsample()

    def mode(self):
        return self.distribution.mean

    def actions_from_params(self, mean_actions, log_std, deterministic: bool = False):
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions, log_std):
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob
