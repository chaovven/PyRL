import torch as th
import random


class EpisodeBuffer:
    def __init__(self, args):
        self.args = args
        self.capacity = args.buffer_size
        self.buffer = []

    def clear(self):
        """
        Remove all episodes in the replay buffer.
        This function should be called by on-policy algorithms to remove old episodes.
        """
        self.buffer.clear()

    def update(self, ep_data):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(ep_data)

    def _len(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        # a batch of episodes available for training
        state = th.cat([x['state'] for x in batch], dim=0)
        action = th.cat([x['action'] for x in batch], dim=0)
        reward = th.cat([x['reward'] for x in batch], dim=0)
        done = th.cat([x['done'] for x in batch], dim=0)
        mask = th.cat([x['mask'] for x in batch], dim=0)

        ep_data = {'state': state, 'action': action, 'reward': reward, 'done': done, 'mask': mask}

        if self.args.buf_act_onehot:
            ep_data['action_onehot'] = th.cat([x['action_onehot'] for x in batch], dim=0)
        if self.args.buf_act_logprob:
            ep_data['log_prob'] = th.cat([x['log_prob'] for x in batch], dim=0)

        return ep_data

    def new_empty_batch(self):
        args = self.args
        act_dim = 1 if args.discrete else args.action_dim
        ep_data = {
            'state': th.zeros([1, args.ep_limit + 1, args.state_dim], dtype=th.float, device=args.device),
            'action': th.zeros([1, args.ep_limit + 1, act_dim], dtype=th.float, device=args.device),
            'reward': th.zeros([1, args.ep_limit + 1, 1], dtype=th.float, device=args.device),
            'mask': th.ones([1, args.ep_limit + 1, 1], dtype=th.float, device=args.device),
            'done': th.zeros([1, args.ep_limit + 1, 1], dtype=th.float, device=args.device),
        }

        if self.args.buf_act_onehot:
            ep_data['action_onehot'] = th.zeros([1, args.ep_limit + 1, args.action_dim], dtype=th.float,
                                                device=args.device)
        if self.args.buf_act_logprob:
            ep_data['log_prob'] = th.zeros([1, args.ep_limit + 1, act_dim], dtype=th.float, device=args.device)

        return ep_data
