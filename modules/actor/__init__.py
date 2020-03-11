from .categorical_policy import CategoricalPolicy
from .deterministic_policy import DeterministicPolicy
from .gaussian_policy import GaussianPolicy
from .squashed_gaussian_policy import SquashedGaussianPolicy

REGISTRY = {}

REGISTRY['categorical'] = CategoricalPolicy
REGISTRY['deterministic'] = DeterministicPolicy
REGISTRY['gaussian'] = GaussianPolicy
REGISTRY['squashed_gaussian'] = SquashedGaussianPolicy
