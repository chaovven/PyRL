import gym
from envs import REGISTRY as ENV_REGISTRY


def eval_policy(learner, action_selector, args):
    # init env
    if args.env in ENV_REGISTRY.keys():    # customize your own environment here, stored in ./envs
        eval_env = ENV_REGISTRY[args.env]()
    else:
        eval_env = gym.make(args.env)

    eval_env.seed(args.seed + 100)

    avg_reward = 0.
    for _ in range(args.test_nepisodes):
        state, done = eval_env.reset(), False
        while not done:
            action = action_selector.select_action(learner.forward(state), 0, train_mode=False)
            action = action.item() if args.discrete else action
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= args.test_nepisodes

    print("---------------------------------------")
    print(f"Evaluation over {args.test_nepisodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
