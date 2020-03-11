import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions.normal import Normal


class SquashedGaussianPolicy(nn.Module):
    def __init__(self, args):
        super(SquashedGaussianPolicy, self).__init__()
        self.args = args

        self.state_encoder = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(args.hidden_dim, args.hidden_dim))

        self.mu = nn.Linear(args.hidden_dim, args.action_dim)
        self.log_std = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, state):
        x = F.relu(self.state_encoder(state))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = th.clamp(log_std, self.args.LOG_STD_MIN, self.args.LOG_STD_MAX)
        std = th.exp(self.log_std(x))  # log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        pi = Normal(mu, std)
        return pi
