import torch as th
import torch.nn.functional as F
from modules.critic.q_fun import QFun_DDPG
from torch.optim import Adam
import copy
from .base_learner import BaseLearner


class DDPGLearner(BaseLearner):
    def __init__(self, args, logger):
        BaseLearner.__init__(self, args, logger)

        self.critic = QFun_DDPG(args)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.critic_lr)

    def train(self, batch, t_env):
        args = self.args
        s = batch['state']
        a = batch['action']
        r = batch['reward'][:, :-1]
        done = batch['done'][:, :-1]
        mask = batch['mask'][:, :-1]

        ############### optimize critic ################
        target_q = r + args.gamma * (1 - done) * self.target_critic(s[:, 1:], self.target_actor(s[:, 1:]))
        chosen_action_q = self.critic(s[:, :-1], a[:, :-1])

        td_error = chosen_action_q - target_q.detach()
        masked_td_error = td_error * mask
        critic_loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic.parameters(), args.grad_norm_clip)
        self.critic_optimizer.step()

        ############### optimize actor #################

        actor_loss = - self.critic(s, self.actor(s))[:, :-1]
        actor_loss = (actor_loss * mask).sum() / mask.sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), args.grad_norm_clip)
        self.actor_optimizer.step()

        ############### update target networks #################
        self._update_target_actor()
        self._update_target_critic()

        ############### log data #################
        if t_env - self.last_log >= args.log_interval:
            self.logger.add_scalar("actor_loss", actor_loss, t_env)
            self.logger.add_scalar("actor_grad_norm", actor_grad_norm, t_env)
            self.logger.add_scalar("critic_loss", critic_loss, t_env)
            self.logger.add_scalar("critic_grad_norm", critic_grad_norm, t_env)
            self.logger.add_scalar("td_error_abs", (td_error * mask).abs().sum() / mask.sum(), t_env)
            self.logger.add_scalar("target_mean", (target_q * mask).sum() / mask.sum(), t_env)

            self.last_log = t_env
