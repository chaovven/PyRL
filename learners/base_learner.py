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
        self.critic_optimizer = None

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

    def save_models(self, path):
        # save actor and critic
        th.save(self.actor.state_dict(), "{}/actor.th".format(path))
        if self.critic is not None:
            th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        # save target networks
        if self.target_actor is not None:
            th.save(self.target_actor.state_dict(), "{}/tar_actor.th".format(path))
        if self.target_critic is not None:
            th.save(self.target_critic.state_dict(), "{}/tar_critic.th".format(path))
        # save optimizers
        th.save(self.actor_optimizer.state_dict(), "{}/actor_opt.th".format(path))
        if self.critic_optimizer is not None:
            th.save(self.critic_optimizer.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        # actor & critic
        self.actor.load_state_dict(th.load("{}/actor.th".format(path), map_location=lambda storage, loc: storage))
        if self.critic is not None:
            self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # target networks
        if self.target_critic is not None:
            self.target_critic.load_state_dict(th.load("{}/tar_critic.th".format(path), map_location=lambda storage, loc: storage))
        if self.target_actor is not None:
            self.target_actor.load_state_dict(th.load("{}/tar_actor.th".format(path), map_location=lambda storage, loc: storage))
        # optimizers
        if self.actor_optimizer is not None:
            self.actor_optimizer.load_state_dict(th.load("{}/actor_opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.critic_optimizer is not None:
            self.critic_optimizer.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))