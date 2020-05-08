import torch as th
from torch.optim import Adam
from modules.actor import REGISTRY as AGENT_REGISTRY


class BaseLearner():
    def __init__(self, args, logger):
        self.logger = logger
        self.args = args

        self.actor = AGENT_REGISTRY[args.actor](args)
        self.critic = None

        self.target_actor = None
        self.target_critic = None

        self.actor_optimizer = Adam(params=self.actor.parameters(), lr=args.lr)

        self.last_log = -self.args.log_interval - 1  # log the first run

    def forward(self, s):
        s = th.tensor(s, dtype=th.float).view(1, -1).to(self.args.device)
        actor_out = self.actor(s)
        return actor_out

    def _update_target_critic(self):
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def _update_target_actor(self):
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    def cuda(self):
        self.actor.cuda()
        if self.critic is not None:
            self.critic.cuda()
        if self.target_critic is not None:
            self.target_critic.cuda()
        if self.target_actor is not None:
            self.target_actor.cuda()
