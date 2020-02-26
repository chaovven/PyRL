import gym

def eval_policy(learner, action_selector, args):
    eval_env = gym.make(args.env)
    eval_env.seed(args.seed + 100)

    avg_reward = 0.
    for _ in range(args.test_nepisodes):
        state, done = eval_env.reset(), False
        while not done:
            agent_output = learner.forward(state)
            action = action_selector.select_action(agent_output, 0, train_mode=False)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= args.test_nepisodes

    print("---------------------------------------")
    print(f"Evaluation over {args.test_nepisodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward
