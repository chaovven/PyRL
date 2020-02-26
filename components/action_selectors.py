import torch as th
import numpy as np
from torch.distributions import Categorical
from components.decay_schedules import DecaySchedule

REGISTRY = {}


class EpsilonGreedyActionSelector():
    """
    Epsilon greedy is usually used in DQN for discrete action space.
    """

    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger

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

        return a.numpy().item()


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class GaussianActionSelector():
    """
    This action selector is for the case of continuous action space
    in which a gaussian distribution of the action is learned.
    """

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def select_action(self, action_dist, t_env, train_mode=False):
        a = action_dist.sample()

        return a.numpy().item()


REGISTRY["gaussian"] = GaussianActionSelector


class MultinomialActionSelector():

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def select_action(self, logits, t_env, train_mode=False):
        if train_mode:
            picked_actions = Categorical(logits).sample().long()
        else:
            picked_actions = logits.max(dim=-1)[1]

        return picked_actions.numpy().item()


REGISTRY["multinomial"] = MultinomialActionSelector


class DeterministicActionSelector():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def select_action(self, actor_out, t_env, train_mode=False):
        if train_mode:  # add noise if training
            noise = np.random.normal(0, self.args.max_action * self.args.expl_noise, size=self.args.action_dim)
            a = (actor_out + noise).clip(self.args.min_action, self.args.max_action)
        else:
            a = actor_out
        return a


REGISTRY["deterministic"] = MultinomialActionSelector
