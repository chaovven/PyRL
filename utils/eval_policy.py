import gym
import torch as th
from envs import REGISTRY as ENV_REGISTRY
from components.episode_buffer import EpisodeBuffer


def eval_policy(learner, action_selector, args):
    # init env
    if args.env in ENV_REGISTRY.keys():  # customize your own environment here, stored in ./envs
        eval_env = ENV_REGISTRY[args.env]()
    else:
        eval_env = gym.make(args.env)

    eval_env.seed(args.seed + 100)

    # episode buffer
    eval_buf = EpisodeBuffer(args)
    ep_data = eval_buf.new_empty_batch()

    avg_reward = 0.
    for _ in range(args.test_nepisodes):
        s0, done, ep_reward, ep_t = eval_env.reset(), False, 0, 0
        while not done:
            a0 = action_selector.select_action(learner.forward(s0), 0, train_mode=False)
            s1, r1, done, _ = eval_env.step(a0)

            # log data
            ep_data['state'][:, ep_t] = th.tensor(s0).view(1, -1)
            ep_data['action'][:, ep_t] = th.tensor(a0).view(1, -1)
            ep_data['reward'][:, ep_t] = th.tensor(r1).view(1, -1)
            if args.buf_act_logprob:
                ep_data['log_prob'][:, ep_t] = action_selector.log_prob.view(1, -1)

            # next state
            s0 = s1
            ep_t += 1
            ep_reward += r1

        # if done
        ep_data['done'][:, ep_t - 1] = th.tensor(1)  # -1 as ep_t+=1 before
        ep_data['mask'][:, ep_t:] = th.tensor(0)
        eval_buf.update(ep_data)  # insert into the replay buffer
        ep_data = eval_buf.new_empty_batch()

        avg_reward += ep_reward

    avg_reward /= args.test_nepisodes

    print("---------------------------------------")
    print(f"Evaluation over {args.test_nepisodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, eval_buf
