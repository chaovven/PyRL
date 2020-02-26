import torch as th
import random


class EpisodeBuffer:
    def __init__(self, args):
        self.args = args
        self.capacity = args.buffer_size
        self.buffer = []

    def update(self, transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

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

        episode_data = {'state': state, 'action': action, 'reward': reward, 'done': done, 'mask': mask}

        if self.args.discrete:
            episode_data['action_onehot'] = th.cat([x['action_onehot'] for x in batch], dim=0)

        return episode_data

    def new_empty_batch(self):
        args = self.args
        ep_data = {
            'state': th.zeros([1, args.ep_limit + 1, args.state_dim], dtype=th.float),
            'action': th.zeros([1, args.ep_limit + 1, 1], dtype=th.float),
            'reward': th.zeros([1, args.ep_limit + 1, 1], dtype=th.float),
            'mask': th.ones([1, args.ep_limit + 1, 1], dtype=th.float),
            'done': th.zeros([1, args.ep_limit + 1, 1], dtype=th.float),
        }
        if args.discrete:
            ep_data['action_onehot'] = th.zeros([1, args.ep_limit + 1, args.action_dim], dtype=th.float)
        return ep_data
