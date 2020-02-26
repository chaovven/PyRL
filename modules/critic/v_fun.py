import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VFun(nn.Module):
    def __init__(self, args):
        super(VFun, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v
