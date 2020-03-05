from .pg_learner import PGLearner
from .ddpg_learner import DDPGLearner
from .td3_learner import TD3Learner
from .dqn_learner import DQNLearner
from .ppo_learner import PPOLearner

REGISTRY = {}

REGISTRY["ddpg"] = DDPGLearner
REGISTRY["pg"] = PGLearner
REGISTRY["td3"] = TD3Learner
REGISTRY["dqn"] = DQNLearner
REGISTRY["ppo"] = PPOLearner
