from .categorical_policy import CategoricalPolicy
from .deterministic_policy import DeterministicPolicy
from .gaussian_policy import GaussianPolicy

REGISTRY = {}

REGISTRY['categorical'] = CategoricalPolicy
REGISTRY['deterministic'] = DeterministicPolicy
REGISTRY['gaussian'] = GaussianPolicy
