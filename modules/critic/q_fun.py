import torch as th
import torch.nn as nn
import torch.nn.functional as F


class QFun_continuous(nn.Module):
    """
    Q Function for continuous case,
    take state and action as input and output the Q(s,a)
    """

    def __init__(self, args):
        super(QFun_continuous, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.state_dim + args.action_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, state, action):
        x = th.cat([state, action], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class QFun_discreate(nn.Module):
    """
    Q function for discreate case,
    take state as input and output Q values for each of the actions
    """

    def __init__(self, args):
        super(QFun_discreate, self).__init__()

        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class QFun_double(nn.Module):
    """
    double Q architecture used by TD3
    """

    def __init__(self, args):
        super(QFun_double, self).__init__()

        self.q1_net = nn.Sequential(nn.Linear(args.state_dim + args.action_dim, args.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_dim, args.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_dim, 1))

        self.q2_net = nn.Sequential(nn.Linear(args.state_dim + args.action_dim, args.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_dim, args.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_dim, 1))

    def q1(self, state, action):
        input = th.cat([state, action], dim=-1)
        return self.q1_net(input)

    def q2(self, state, action):
        input = th.cat([state, action], dim=-1)
        return self.q2_net(input)


class QFun_DDPG(nn.Module):
    def __init__(self, args):
        super(QFun_DDPG, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.state_dim + args.action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = th.cat([state, action], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
