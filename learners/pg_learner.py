import torch as th
from modules.critic.v_fun import VFun
from torch.optim import Adam
import copy
from .base_learner import BaseLearner


class PGLearner(BaseLearner):
    def __init__(self, args, logger):
        BaseLearner.__init__(self, args, logger)

        self.critic = VFun(args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.critic_lr, weight_decay=1e-2)

    def train(self, batch, t_env):
        args = self.args
        s0 = batch['state']
        a0 = batch['action']
        r1 = batch['reward'][:, :-1]
        done = batch['done'][:, :-1]
        mask = batch['mask'][:, :-1]

        ############### optimize critic ################
        critic_stats = self.train_critic(batch)

        ############### optimize actor #################

        # log probability
        actor_out = self.actor(s0)
        if args.discrete:
            action_probs = th.nn.functional.softmax(actor_out, dim=-1)
            log_prob = th.log(action_probs)
            log_prob_taken = th.gather(log_prob, index=a0.long(), dim=-1)[:, :-1]
        else:
            log_prob_taken = actor_out.log_prob(a0)[:, :-1]

        # advantage
        V = self.critic(s0)
        adv = (r1 + (1 - done) * args.gamma * V[:, 1:] - V[:, :-1]).detach()

        # optimize actor
        actor_loss = -(log_prob_taken * adv * mask).sum() / mask.sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
        self.actor_optimizer.step()

        # log
        if t_env - self.last_log >= self.args.log_interval:
            self.logger.add_scalar("actor_loss", actor_loss, t_env)
            self.logger.add_scalar("advantage", (adv * mask).sum() / mask.sum(), t_env)
            self.logger.add_scalar("actor_grad_norm", grad_norm, t_env)

            ts_logged = len(critic_stats['critic_loss'])  # timestep logged
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs"]:
                self.logger.add_scalar(key, sum(critic_stats[key]) / ts_logged, t_env)

            self.last_log = t_env

    def train_critic(self, batch):
        s0 = batch['state']
        r1 = batch['reward'][:, :-1]
        mask = batch['mask'][:, :-1]
        done = batch['done'][:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
        }

        target_v = self.target_critic(s0)[:, 1:]
        target = (r1 + self.args.gamma * (1 - done) * target_v).detach()

        for t in range(self.args.ep_limit - 1):
            mask_t = mask[:, t]
            if mask_t.sum() == 0:
                continue

            target_t = target[:, t]
            v_t = self.critic(s0[:, t])

            td_error = target_t - v_t
            masked_td_error = td_error * mask_t

            loss = (masked_td_error ** 2).sum() / mask_t.sum()

            self.critic_optimizer.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_norm_clip)
            self.critic_optimizer.step()

            self._update_target_critic()

            running_log['critic_loss'].append(loss)
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))

        return running_log
