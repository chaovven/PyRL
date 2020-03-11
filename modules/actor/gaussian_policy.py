import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch as th
import torch.nn.functional as F


class GaussianPolicy(nn.Module):
    def __init__(self, args):
        super(GaussianPolicy, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.action_dim)

        log_std = -0.5 * np.ones(args.action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(th.as_tensor(log_std))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        mu = self.fc2(x)
        std = th.exp(self.log_std)
        pi = Normal(mu, std)
        return pi
