import torch as th
from modules.critic.v_fun import VFun
from torch.optim import Adam
import copy
from .base_learner import BaseLearner


class PPOLearner(BaseLearner):
    def __init__(self, args, logger):
        BaseLearner.__init__(self, args, logger)

        self.critic = VFun(args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.critic_lr, weight_decay=1e-2)

    def train(self, batch, t_env):

        actor_stats = self.train_actor(batch)
        critic_stats = self.train_critic(batch)

        if t_env - self.last_log >= self.args.log_interval:

            ts_logged = len(actor_stats['actor_loss'])  # timestep logged
            for key in ["actor_loss", "actor_grad_norm", "ratio", "cliped_adv"]:
                self.logger.add_scalar(key, sum(actor_stats[key]) / ts_logged, t_env)

            ts_logged = len(critic_stats['critic_loss'])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs"]:
                self.logger.add_scalar(key, sum(critic_stats[key]) / ts_logged, t_env)

            self.last_log = t_env

    def train_actor(self, batch):
        args = self.args
        state = batch['state']
        action = batch['action']
        reward = batch['reward'][:, :-1]
        mask = batch['mask'][:, :-1]
        done = batch['done'][:, :-1]
        log_probs_old = batch['log_prob']

        target = reward + self.args.gamma * (1 - done) * self.target_critic(state[:, 1:])  # TODO: use GAE
        adv = (target - self.critic(state[:, :-1])).detach()

        running_log = {
            "actor_loss": [],
            "actor_grad_norm": [],
            "ratio": [],
            "cliped_adv": []
        }

        for i in range(self.args.train_actor_iters):
            log_prob = self.get_log_prob(state, action)

            ratio = th.exp(log_prob - log_probs_old)[:, :-1]
            cliped_adv = th.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * adv

            actor_loss = -(th.min(ratio * adv, cliped_adv) * mask).sum() / mask.sum()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
            self.actor_optimizer.step()

            running_log['actor_loss'].append(actor_loss)
            running_log["actor_grad_norm"].append(grad_norm)
            mask_elems = mask.sum().item()
            running_log["ratio"].append((ratio.sum().item() / mask_elems))
            running_log["cliped_adv"].append((cliped_adv.sum().item() / mask_elems))

        return running_log

    def train_critic(self, batch):
        state = batch['state']
        reward = batch['reward'][:, :-1]
        mask = batch['mask'][:, :-1]
        done = batch['done'][:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
        }

        target = (reward + self.args.gamma * (1 - done) * self.target_critic(state[:, 1:])).detach()

        for i in range(self.args.train_critic_iters):
            v = self.critic(state[:, :-1])

            td_error = target - v
            masked_td_error = td_error * mask

            loss = (masked_td_error ** 2).sum() / mask.sum()

            self.critic_optimizer.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_norm_clip)
            self.critic_optimizer.step()

            self._update_target_critic()

            running_log['critic_loss'].append(loss)
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))

        return running_log

    def get_log_prob(self, state, action):
        actor_out = self.actor(state)

        if self.args.discrete:
            probs = th.nn.functional.softmax(actor_out, dim=-1)
            log_probs = th.log(probs)
            log_probs = th.gather(log_probs, index=action.long(), dim=-1)
        else:
            log_probs = actor_out.log_prob(action)

        return log_probs
