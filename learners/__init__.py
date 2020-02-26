from .pg_learner import PGLearner
from .ddpg_learner import DDPGLearner
from .td3_learner import TD3Learner
from .dqn_learner import DQNLearner

REGISTRY = {}

REGISTRY["ddpg_learner"] = DDPGLearner
REGISTRY["pg_learner"] = PGLearner
REGISTRY["td3_learner"] = TD3Learner
REGISTRY["dqn_learner"] = DQNLearner
