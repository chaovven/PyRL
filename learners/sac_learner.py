import torch as th
from modules.critic.q_fun import QFun_double
from torch.optim import Adam
import copy
from .base_learner import BaseLearner


class SACLearner(BaseLearner):
    def __init__(self, args, logger):
        BaseLearner.__init__(self, args, logger)

        self.critic = QFun_double(args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.critic_lr, weight_decay=1e-2)

    def train(self, batch, t_env):
        args = self.args
        s = batch['state']
        a = batch['action']
        r = batch['reward'][:, :-1]
        d = batch['done'][:, :-1]
        mask = batch['mask'][:, :-1]

        ######## sample target action and calculate log probs ###########

        # sample action from current policy instead of the repaly buffer!
        pi = self.actor(s)
        targ_action = pi.rsample()  # reparameterization trick

        # calculate the log probabilities of the sampled actions
        log_prob = pi.log_prob(targ_action).sum(dim=-1, keepdim=True)
        log_prob -= th.log(1 - th.tanh(targ_action).pow(2)).sum(dim=-1, keepdim=True)

        targ_action = th.tanh(targ_action) * th.tensor(self.args.max_action, dtype=th.float)

        ############### optimize critic ################

        chosen_action_q1 = self.critic.q1(s[:, :-1], a[:, :-1])
        chosen_action_q2 = self.critic.q2(s[:, :-1], a[:, :-1])

        targ_q1 = self.target_critic.q1(s[:, 1:], targ_action[:, 1:])
        targ_q2 = self.target_critic.q2(s[:, 1:], targ_action[:, 1:])
        targ_q_min = th.min(targ_q1, targ_q2)

        if args.pi_entropy:
            target_q = r + args.gamma * (1 - d) * (targ_q_min - args.ent_coef * log_prob[:, 1:])
        else:
            target_q = r + args.gamma * (1 - d) * targ_q_min

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
        pi = self.actor(s)
        act = th.tanh(pi.rsample()) * th.tensor(self.args.max_action, dtype=th.float)

        q1_pi = self.critic.q1(s, act)[:, :-1]
        q2_pi = self.critic.q2(s, act)[:, :-1]
        actor_loss = - th.min(q1_pi, q2_pi)

        if args.pi_entropy:
            actor_loss += args.ent_coef * log_prob[:, :-1]

        actor_loss = (actor_loss * mask).sum() / mask.sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), args.grad_norm_clip)
        self.actor_optimizer.step()

        ############### update target networks #################
        self._update_target_critic()

        ############### log data #################
        if t_env - self.last_log >= args.log_interval:
            self.logger.add_scalar("actor_loss", actor_loss, t_env)
            self.logger.add_scalar("actor_grad_norm", actor_grad_norm, t_env)
            self.logger.add_scalar("critic_loss", critic_loss, t_env)
            self.logger.add_scalar("critic_grad_norm", critic_grad_norm, t_env)
            self.logger.add_scalar("td_error1_abs", (td_error1 * mask).abs().sum() / mask.sum(), t_env)
            self.logger.add_scalar("td_error2_abs", (td_error2 * mask).abs().sum() / mask.sum(), t_env)
            self.logger.add_scalar("target_mean", (target_q * mask).sum() / mask.sum(), t_env)

            self.last_log = t_env
