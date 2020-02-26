from utils.eval_policy import eval_policy
import pprint
from components.episode_buffer import EpisodeBuffer
from tensorboardX import SummaryWriter
from learners import REGISTRY as L_REGISTRY
from components.action_selectors import REGISTRY as A_REGISTRY
from types import SimpleNamespace as SN
from utils.rl_utils import *
from components.one_hot import one_hot


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

    # tensorboard config
    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=args.tb_path)

    learner = L_REGISTRY[args.learner](args, writer)  # algorithm
    action_selector = A_REGISTRY[args.action_selector](args, writer)  # action selector

    ep_buffer = EpisodeBuffer(args)

    ep_num = 0
    ep_t = 0
    ep_reward = 0
    ep_data = ep_buffer.new_empty_batch()

    t_env_old = -args.log_interval - 1  # log the first run

    s0 = env.reset()

    for t_env in range(int(args.max_timesteps)):

        # chose action
        if t_env < args.start_timesteps:  # use random policy if no sufficient transitions collected
            a0 = env.action_space.sample()
        else:
            a0 = action_selector.select_action(learner.forward(s0), t_env, train_mode=True)

        s1, r1, done, _ = env.step(a0)

        ep_data['state'][:, ep_t] = th.tensor(s0).view(1, -1)
        ep_data['action'][:, ep_t] = th.tensor(a0).view(1, -1)
        ep_data['reward'][:, ep_t] = th.tensor(r1).view(1, -1)
        if args.discrete:
            ep_data['action_onehot'][:, ep_t] = one_hot(th.tensor(a0).view(1, -1), args.action_dim)

        s0 = s1
        ep_t += 1
        ep_reward += r1

        if done:
            print(f"Episode {ep_num +1}: reward = {ep_reward:.3f}, timestep = {t_env + 1}")

            # log reward for both training and testing
            if t_env - t_env_old >= args.log_interval:
                writer.add_scalar("training/reward", ep_reward, t_env + 1)
                evaluation = eval_policy(learner, action_selector, args)
                writer.add_scalar("testing/reward", evaluation, t_env + 1)
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

            # train policy after sampling an episode
            if ep_buffer._len() >= args.batch_size:  # can train or not
                batch_samples = ep_buffer.sample(args.batch_size)
                learner.train(batch_samples, t_env)

    env.close()

    print("Exiting Main")
