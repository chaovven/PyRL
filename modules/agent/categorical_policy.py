import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch as th
import torch.nn.functional as F


class CategoricalPolicy(nn.Module):
    def __init__(self, args):
        super(CategoricalPolicy, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        out = self.fc2(x)

        return out
