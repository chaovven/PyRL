import torch as th
import numpy as np
from torch.distributions import Categorical
from components.decay_schedules import DecaySchedule


class BaseActionSelector():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def select_action(self, agent_out, t_env, train_mode=False):
        raise NotImplementedError


class EpsilonGreedyActionSelector(BaseActionSelector):
    """
    Epsilon greedy is usually used in DQN for discrete action space.
    """

    def __init__(self, args, logger=None):
        BaseActionSelector.__init__(self, args, logger)

        self.schedule = DecaySchedule(args.eps_start, args.eps_finish, args.eps_anneal_time, decay="exp")
        self.epsilon = self.schedule.eval(0)

        self.last_log = -self.args.log_interval - 1  # last time the data logged

    def select_action(self, agent_out, t_env, train_mode=False):
        self.epsilon = self.schedule.eval(t_env)

        if not train_mode:
            self.epsilon = 0.0  # no exploration during testing

        random_numbers = th.rand_like(agent_out[:, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(th.ones_like(agent_out)).sample().long()
        a = pick_random * random_actions + (1 - pick_random) * agent_out.max(dim=-1)[1]

        if t_env - self.last_log >= self.args.log_interval and train_mode:
            self.logger.add_scalar("epsilon", self.epsilon, t_env)
            self.last_log = t_env

        return a.detach()


class GaussianActionSelector(BaseActionSelector):
    """
    This action selector is for the case of continuous action space
    in which a gaussian distribution of the action is learned.
    Used by continuous algorithms like PPO and Policy Gradient.
    """

    def __init__(self, args, logger):
        BaseActionSelector.__init__(self, args, logger)
        self.log_prob = None

    def select_action(self, action_dist, t_env, train_mode=False):
        if train_mode:
            a = action_dist.sample()
            self.log_prob = action_dist.log_prob(a).detach()
        else:
            a = action_dist.mean.detach()

        return a.detach()


class SquashedGaussianActionSelector(BaseActionSelector):
    """
    Used by Soft Actor Critc (SAC)
    """

    def __init__(self, args, logger):
        BaseActionSelector.__init__(self, args, logger)
        self.log_prob = None

    def select_action(self, action_dist, t_env, train_mode=False):
        if train_mode:
            a = action_dist.rsample()  # reparameterization trick
        else:
            a = action_dist.mean

        a = th.tanh(a) * th.tensor(self.args.max_action, dtype=th.float)  # TODO: how about just clipping like TD3?

        return a.detach()


class MultinomialActionSelector(BaseActionSelector):

    def __init__(self, args, logger):
        BaseActionSelector.__init__(self, args, logger)
        self.schedule = DecaySchedule(args.eps_start, args.eps_finish, args.eps_anneal_time, decay="exp")
        self.epsilon = self.schedule.eval(0)

        self.last_log = -self.args.log_interval - 1  # last time the data logged
        self.log_prob = None

    def select_action(self, logits, t_env, train_mode=False):
        self.epsilon = self.schedule.eval(t_env)
        probs = th.nn.functional.softmax(logits, dim=-1)

        if train_mode:
            # select a random action with epsilon probability
            probs = probs * (1 - self.epsilon) + self.epsilon / probs.shape[-1]
            picked_actions = Categorical(probs=probs).sample().long()
            # log
            if t_env - self.last_log >= self.args.log_interval:
                self.logger.add_scalar("epsilon", self.epsilon, t_env)
                self.last_log = t_env
        else:
            picked_actions = probs.max(dim=-1)[1]

        self.log_prob = th.log(probs).detach()[:, picked_actions]

        return picked_actions


class DeterministicActionSelector(BaseActionSelector):
    def __init__(self, args, logger):
        BaseActionSelector.__init__(self, args, logger)

    def select_action(self, actor_out, t_env, train_mode=False):
        if train_mode:  # add noise if training
            noise = np.random.normal(0, self.args.max_action * self.args.expl_noise, size=self.args.action_dim)
            a = (actor_out + noise).clip(self.args.min_action, self.args.max_action)
        else:
            a = actor_out

        return a.detach()


REGISTRY = {}
REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
REGISTRY["gaussian"] = GaussianActionSelector
REGISTRY["multinomial"] = MultinomialActionSelector
REGISTRY["deterministic"] = DeterministicActionSelector
REGISTRY["squashed_gaussian"] = SquashedGaussianActionSelector
