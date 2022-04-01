import torch as th
import copy
from .base_learner import BaseLearner


class DQNLearner(BaseLearner):
    def __init__(self, args, logger):
        BaseLearner.__init__(self, args, logger)

        self.target_actor = copy.deepcopy(self.actor)  # target q network

    def train(self, batch, t_env):
        args = self.args
        s = batch['state']
        a = batch['action'].long()
        r = batch['reward']
        next_s = batch['next_state']
        done = batch['done']

        ############### optimize actor ################
        # Q(s,a) for each a
        qvals = self.actor(s)
        # Q(s,a)
        chosen_action_qvals = th.gather(qvals, dim=-1, index=a)

        target_q = r + args.gamma * (1 - done) * self.target_actor(next_s).max(dim=-1, keepdim=True)[0]  # Q target
        td_error = chosen_action_qvals - target_q.detach()  # TD error

        # loss
        loss = (td_error ** 2).sum() / td_error.shape[0]

        self.actor_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
        self.actor_optimizer.step()

        ############### update target networks #################
        self._update_target_actor()

        ############### log data #################
        if t_env - self.last_log >= args.log_interval:
            self.logger.add_scalar("loss", loss, t_env)
            # self.logger.add_scalar("actor_grad_norm", actor_grad_norm, t_env)

            self.last_log = t_env

    def train_episode(self, batch, t_env):
        args = self.args
        s = batch['state']
        a = batch['action'].long()
        r = batch['reward'][:, :-1]
        done = batch['done'][:, :-1]
        mask = batch['mask'][:, :-1]

        qvals = self.actor(s)[:, :-1]
        chosen_action_qvals = th.gather(qvals, dim=2, index=a[:, :-1])
        target_qvals = r + args.gamma * (1 - done) * self.target_actor(s[:, 1:]).max(dim=-1, keepdim=True)[0]

        td_error = chosen_action_qvals - target_qvals.detach()
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.actor_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
        self.actor_optimizer.step()

        # log
        if t_env - self.last_log >= self.args.log_interval:
            self.logger.add_scalar("loss", loss, t_env)
            self.logger.add_scalar("grad_norm", grad_norm, t_env)
            self.logger.add_scalar("td_error_abs", (td_error * mask).abs().sum() / mask.sum(), t_env)
            self.logger.add_scalar("target_mean", (target_qvals * mask).sum() / mask.sum(), t_env)

            self.last_log = t_env

        self._update_target_actor()
