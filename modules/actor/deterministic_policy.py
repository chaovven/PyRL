import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DeterministicPolicy(nn.Module):
    def __init__(self, args):
        super(DeterministicPolicy, self).__init__()
        self.args = args

        self.max_action = th.tensor(args.max_action, dtype=th.float, device=args.device).view(1, -1)

        self.fc1 = nn.Linear(args.state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, args.action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        a = self.max_action * th.tanh(self.fc3(x))
        return a
