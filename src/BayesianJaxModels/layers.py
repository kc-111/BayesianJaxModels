"""Bayesian layers built on the Module base class."""

from typing import Type

import jax
import jax.numpy as jnp
from jax import random

from .module import Module
from .parameter import (
    AbstractParameter,
    GaussianParameter,
    make_parameter,
)


class BayesianLinear(Module):
    """Bayesian fully-connected layer.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        use_bias: Whether to include a bias term.
        bayesian: If True (default), weights use a variational distribution.
        bayesian_bias: If True (default), bias also uses a variational
            distribution.
        param_type: ``GaussianParameter`` (default) or ``LaplacianParameter``.
        init_log_sigma: Initial value for the unconstrained log-scale
            parameter.  Effective initial stdv is ``exp(init_log_sigma)``.
        key: PRNG key for Xavier initialisation of means.
    """

    W: AbstractParameter
    b: AbstractParameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        bayesian: bool = True,
        bayesian_bias: bool = True,
        param_type: Type[AbstractParameter] = GaussianParameter,
        init_log_sigma: float = -5.0,
        key: jax.Array,
    ):
        k1, k2 = random.split(key)
        # Xavier uniform initialisation
        limit = jnp.sqrt(6.0 / (in_features + out_features))
        W_init = random.uniform(k1, (out_features, in_features), minval=-limit, maxval=limit)
        self.W = make_parameter(
            W_init, bayesian=bayesian, param_type=param_type,
            init_log_sigma=init_log_sigma,
        )

        if use_bias:
            b_init = jnp.zeros(out_features)
            self.b = make_parameter(
                b_init, bayesian=bayesian_bias, param_type=param_type,
                init_log_sigma=init_log_sigma,
            )
        else:
            self.b = None

    def __call__(self, x: jax.Array, *, key: jax.Array | None, sample: bool = True) -> jax.Array:
        """Forward pass.

        Args:
            x: Input array of shape ``(..., in_features)``.
            key: PRNG key. Required when ``sample=True``.
            sample: If True, sample weights from the variational distribution.
                If False, use the means (MAP / deterministic mode).

        Returns:
            Output array of shape ``(..., out_features)``.
        """
        if sample and key is None:
            raise ValueError("key is required when sample=True")

        if sample:
            k1, k2 = random.split(key)
            W = self.W.sample(k1)
        else:
            W = self.W.mean

        out = x @ W.T

        if self.b is not None:
            if sample:
                b = self.b.sample(k2)
            else:
                b = self.b.mean
            out = out + b

        return out
