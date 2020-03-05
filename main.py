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
    use_critic = _config['learner'] not in ['dqn_learner']
    values = ['env', 'learner', 'lr']
    names = ['', '', 'lr=']
    if use_critic:
        values.append('critic_lr')
        names.append('clr=')
    unique_token = _config[values[0]]
    for i in range(1, len(names)):
        unique_token = unique_token + '__{}{}'.format(names[i], str(_config[values[i]]))
    unique_token += '__' + _config['time']

    _config['tb_path'] = os.path.join(dirname(abspath(__file__)), "results", _config['env'], unique_token)

    # run the framework
    run(_run, _config, _log, env)


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
