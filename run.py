import os.path
from utils.eval_policy import eval_policy
import numpy as np
import pprint
from tensorboardX import SummaryWriter
from learners import REGISTRY as L_REGISTRY
from components.action_selectors import REGISTRY as A_REGISTRY
from types import SimpleNamespace as SN
from utils.rl_utils import *
from components.one_hot import one_hot
from components.episode_buffer import EpisodeBuffer


def run(_run, _config, _log, env):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)  # args['example'] -> args.example
    args.device = "cuda" if args.use_cuda else "cpu"

    # show parameters in console
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # ------------------- setup tensorboard -------------------
    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=args.tb_path)

    # ------------------- setup learner -------------------
    learner = L_REGISTRY[args.learner](args, writer)  # algorithm
    if args.use_cuda:
        learner.cuda()

    # ------------------- load models -------------------
    if args.load_model_path is not "":  # load model if 'checkpoints' specified
        if not os.path.isdir(args.load_model_path):
            raise ValueError("Not a directory: " + args.load_model_path)
        learner.load_models(args.load_model_path)

    # -------------- setup action selector --------------
    action_selector = A_REGISTRY[args.action_selector](args, writer)  # action selector

    # ------------------- setup buffer -------------------
    ep_buffer = EpisodeBuffer(args)

    ep_data = ep_buffer.new_empty_batch()
    t_env_old = -args.log_interval - 1  # log the first run

    # ------------------- collect trajectories -------------------
    s0, ep_num, ep_t, ep_reward, model_saved_time = env.reset(), 0, 0, 0, 0

    for t_env in range(int(args.max_timesteps)):

        # save models
        if args.save_model and (t_env - model_saved_time >= args.model_save_interval or model_saved_time == 0):
            # model saved in "{args.tb_path}/models/{t_env}"
            model_save_path = os.path.join(args.tb_path, "models", str(t_env))
            os.makedirs(model_save_path, exist_ok=True)
            learner.save_models(model_save_path)

            model_saved_time = t_env

        # chose action
        if t_env < args.start_timesteps:  # use random policy if no sufficient transitions collected
            a0 = np.array(env.action_space.sample())
        else:
            a0 = action_selector.select_action(learner.forward(s0), t_env, train_mode=True)

        s1, r1, done, _ = env.step(a0)

        ep_data['state'][:, ep_t] = th.tensor(s0).view(1, -1)
        ep_data['action'][:, ep_t] = th.tensor(a0).view(1, -1)
        ep_data['reward'][:, ep_t] = th.tensor(r1).view(1, -1)
        if args.buf_act_logprob:
            ep_data['log_prob'][:, ep_t] = action_selector.log_prob.view(1, -1)

        s0 = s1
        ep_t += 1
        ep_reward += r1

        if done:
            print(f"Episode {ep_num + 1}: reward = {ep_reward:.3f}, timestep = {t_env + 1}")

            # log reward for both training and testing
            if t_env - t_env_old >= args.log_interval:
                writer.add_scalar("train/reward", ep_reward, t_env + 1)
                eval_reward, eval_buf = eval_policy(learner, action_selector, args)
                writer.add_scalar("test/reward", eval_reward, t_env + 1)
                t_env_old = t_env

            ep_data['done'][:, ep_t - 1] = th.tensor(1)  # -1 as ep_t+=1 before
            ep_data['mask'][:, ep_t:] = th.tensor(0)

            ep_buffer.update(ep_data)  # insert into the replay buffer

            # reset counters, env and create new empty batch
            ep_num += 1
            ep_reward = 0
            ep_t = 0
            s0, done = env.reset(), False
            ep_data = ep_buffer.new_empty_batch()

            # train policy when sufficient episodes are collected
            if t_env >= args.start_timesteps and ep_buffer._len() >= args.batch_size:
                batch_samples = ep_buffer.sample(args.batch_size)
                learner.train(batch_samples, t_env)

                # if on policy, clear the replay buffer
                if args.buffer_size == args.batch_size:
                    ep_buffer.clear()

    env.close()

    print("Exiting Main")
