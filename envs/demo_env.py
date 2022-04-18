"""
You can customize your own environment following the interfaces shown in this demo.
For more examples, please visit https://github.com/openai/gym.
"""


class DemoEnv():
    def __init__(self):
        pass

    def seed(self, seed=None):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass
