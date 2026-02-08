"""BayesianJaxModels — modular Bayesian model framework in JAX.

Built on equinox.  Provides parameter types, a Module base class with
introspection, Bayesian layers, and utility functions for freeze/unfreeze,
sampling, and entropy computation.
"""

from .parameter import (
    STDV_EPS,
    AbstractParameter,
    DeterministicParameter,
    GaussianParameter,
    LaplacianParameter,
    make_parameter,
)
from .module import Module
from .layers import BayesianLinear
from .utils import (
    freeze_stdvs,
    freeze_means,
    freeze_params,
    unfreeze_all,
    flatten_means,
    flatten_stdvs,
    get_parameter_count,
    get_parameter_groups,
    sample_all_parameters,
    gaussian_entropy,
    laplacian_entropy,
)

__all__ = [
    # Parameter types
    "STDV_EPS",
    "AbstractParameter",
    "DeterministicParameter",
    "GaussianParameter",
    "LaplacianParameter",
    "make_parameter",
    # Module base
    "Module",
    # Layers
    "BayesianLinear",
    # Utilities
    "freeze_stdvs",
    "freeze_means",
    "freeze_params",
    "unfreeze_all",
    "flatten_means",
    "flatten_stdvs",
    "get_parameter_count",
    "get_parameter_groups",
    "sample_all_parameters",
    "gaussian_entropy",
    "laplacian_entropy",
]
