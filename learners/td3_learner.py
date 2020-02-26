from numpy.random import normal
import torch as th
from modules.critic.q_fun import QFun_td3
from torch.optim import Adam
import copy
from .base_learner import BaseLearner


class TD3Learner(BaseLearner):
    def __init__(self, args, logger):
        BaseLearner.__init__(self, args, logger)

        self.critic = QFun_td3(args)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.critic_lr, weight_decay=1e-2)

        self.pi_last_updated = -self.args.log_interval - 1  # last time the policy updated

    def train(self, batch, t_env):
        args = self.args
        s = batch['state']
        a = batch['action']
        r = batch['reward'][:, :-1]
        done = batch['done'][:, :-1]
        mask = batch['mask'][:, :-1]

        ############### optimize critic ################
        # trick 1: add noise to the target action
        noise = normal(0, args.max_action * args.policy_noise, size=a[:, 1:].shape).clip(-args.noise_clip,
                                                                                         args.noise_clip)
        target_action = (self.target_actor(s[:, 1:]).detach().numpy() + noise).clip(args.min_action, args.max_action)

        target_action = th.FloatTensor(target_action)

        # trick 2: use two q networks
        target_q1, target_q2 = self.target_critic(s[:, 1:], target_action)
        target_q1 = r + args.gamma * (1 - done) * target_q1
        target_q2 = r + args.gamma * (1 - done) * target_q2
        target_q = th.min(target_q1, target_q2)

        chosen_action_q1, chosen_action_q2 = self.critic(s[:, :-1], a[:, :-1])

        td_error1 = chosen_action_q1 - target_q.detach()
        td_error2 = chosen_action_q2 - target_q.detach()

        masked_td_error1 = td_error1 * mask
        masked_td_error2 = td_error2 * mask

        critic_loss1 = (masked_td_error1 ** 2).sum() / mask.sum()
        critic_loss2 = (masked_td_error2 ** 2).sum() / mask.sum()

        critic_loss = critic_loss1 + critic_loss2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic.parameters(), args.grad_norm_clip)
        self.critic_optimizer.step()

        ############### optimize actor #################

        # trick 3: delay the update of the actor
        if t_env - self.pi_last_updated >= self.args.policy_freq:
            actor_loss = - self.critic.Q1(s, self.actor(s))[:, :-1]
            actor_loss = (actor_loss * mask).sum() / mask.sum()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), args.grad_norm_clip)
            self.actor_optimizer.step()

            self._update_target_actor()
            self._update_target_critic()

            self.pi_last_updated = t_env

            self.actor_loss = actor_loss
            self.actor_grad_norm = actor_grad_norm

        ############### log data #################
        if t_env - self.last_log >= args.log_interval:
            self.logger.add_scalar("actor_loss", self.actor_loss, self.pi_last_updated)
            self.logger.add_scalar("actor_grad_norm", self.actor_grad_norm, self.pi_last_updated)
            self.logger.add_scalar("critic_loss", critic_loss, t_env)
            self.logger.add_scalar("critic_grad_norm", critic_grad_norm, t_env)
            self.logger.add_scalar("td_error1_abs", (td_error1 * mask).abs().sum() / mask.sum(), t_env)
            self.logger.add_scalar("td_error2_abs", (td_error2 * mask).abs().sum() / mask.sum(), t_env)
            self.logger.add_scalar("target_mean", (target_q * mask).sum() / mask.sum(), t_env)

            self.last_log = t_env
