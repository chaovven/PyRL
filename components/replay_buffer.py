import numpy as np
import torch as th


class ReplayBuffer():
    def __init__(self, args):
        self.args = args
        self.ptr = 0  # pointer that points to the
        self.current_size = 0

        self.state = np.zeros((args.buffer_size, args.state_dim))
        self.action = np.zeros((args.buffer_size, args.action_dim))
        self.next_state = np.zeros((args.buffer_size, args.state_dim))
        self.reward = np.zeros((args.buffer_size, 1))
        self.done = np.zeros((args.buffer_size, 1))
        if args.buf_act_logprob:
            self.logprob = np.zeros((args.buffer_size, args.action_dim))

    def add(self, state, action, next_state, reward, done, logprob=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        if self.args.buf_act_logprob:
            self.logprob[self.ptr] = logprob

        self.ptr = (self.ptr + 1) % self.args.buffer_size
        self.current_size = min(self.current_size + 1, self.args.buffer_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.current_size, size=batch_size)

        data = {'state': th.FloatTensor(self.state[ind]).to(self.args.device),
                'action': th.FloatTensor(self.action[ind]).to(self.args.device),
                'next_state': th.FloatTensor(self.next_state[ind]).to(self.args.device),
                'reward': th.FloatTensor(self.reward[ind]).to(self.args.device),
                'done': th.FloatTensor(self.done[ind]).to(self.args.device)}

        if self.args.buf_act_logprob:
            data['logprob'] = th.FloatTensor(self.logprob[ind]).to(self.args.device)

        return data
