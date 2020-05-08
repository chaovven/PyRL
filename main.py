import numpy as np
import datetime
import os
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
import sys
import torch as th
from run import run
from run import run_episode
import yaml
import gym
from gym.spaces import Box, Discrete

ex = Experiment("pyrl")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])

    env = gym.make(_config['env'])
    env.seed(_config["seed"])

    # add info about env
    _config['state_dim'] = env.observation_space.shape[0]
    _config['ep_limit'] = env._max_episode_steps

    if isinstance(env.action_space, Box):
        _config['discrete'] = False
        _config['action_dim'] = env.action_space.shape[0]
        _config['max_action'] = env.action_space.high
        _config['min_action'] = env.action_space.low

    elif isinstance(env.action_space, Discrete):
        _config['discrete'] = True
        _config['action_dim'] = env.action_space.n

    # fields that appear in the event filename
    critic_lr = '' if _config['learner'] in ['dqn'] else '__clr={}'.format(_config["critic_lr"])
    unique_token = '{}-{}__{}__lr={}{}__bt={}_{}'.format(_config["env"],
                                                         _config["learner"],
                                                         _config["name"],
                                                         _config["lr"],
                                                         critic_lr,
                                                         _config["buffer_type"],
                                                         _config["time"])

    _config['tb_path'] = os.path.join(dirname(abspath(__file__)), "results", _config['env'], unique_token)

    if _config['buffer_type'] == 'transition':
        run(_run, _config, _log, env)
    elif _config['buffer_type'] == 'episode':
        run_episode(_run, _config, _log, env)
    else:
        print("error: buffer_type not recognized!")


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # load default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm configs
    alg_config = _get_config(params, "--alg", 'algs')
    config_dict.update(alg_config)

    config_dict['time'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ex.observers.append(
        FileStorageObserver.create(os.path.join(dirname(abspath(__file__)), "results/sacred", config_dict['time'])))

    # add all the config to sacred
    ex.add_config(config_dict)
    ex.run_commandline(params)
