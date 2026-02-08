"""Bayesian parameter classes built on equinox.

Each parameter stores a mean and optionally variational parameters (stdv/scale).
Sampling uses reparameterization tricks for gradient flow.
"""

from abc import abstractmethod
from typing import Type

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

STDV_EPS = 1e-4


class AbstractParameter(eqx.Module):
    """Base class for all parameters."""

    mean: jax.Array

    @abstractmethod
    def sample(self, rng: jax.Array) -> jax.Array:
        """Draw a sample via the reparameterization trick.

        Args:
            rng: PRNG key.

        Returns:
            Sampled array with the same shape as ``mean``.
        """
        ...

    @property
    def shape(self) -> tuple:
        """Shape of the parameter."""
        return self.mean.shape


class DeterministicParameter(AbstractParameter):
    """Fixed parameter (no variational distribution)."""

    def sample(self, rng: jax.Array) -> jax.Array:
        """Return the mean (no randomness).

        Args:
            rng: PRNG key (ignored).

        Returns:
            The stored mean value.
        """
        return self.mean


class GaussianParameter(AbstractParameter):
    """Gaussian variational parameter with reparameterization.

    The actual standard deviation is ``raw_stdv**2 + STDV_EPS``, which
    guarantees positivity without constrained optimisation.
    """

    raw_stdv: jax.Array

    @property
    def stdv(self) -> jax.Array:
        """Positive standard deviation derived from ``raw_stdv``."""
        return self.raw_stdv ** 2 + STDV_EPS

    def sample(self, rng: jax.Array) -> jax.Array:
        """Sample via ``mean + stdv * N(0, 1)``.

        Args:
            rng: PRNG key.

        Returns:
            Sampled array with the same shape as ``mean``.
        """
        return self.mean + self.stdv * random.normal(rng, self.mean.shape)


class LaplacianParameter(AbstractParameter):
    """Laplace variational parameter using inverse-CDF reparameterization.

    The actual scale is ``raw_scale**2 + STDV_EPS``.
    """

    raw_scale: jax.Array

    @property
    def scale(self) -> jax.Array:
        """Positive scale derived from ``raw_scale``."""
        return self.raw_scale ** 2 + STDV_EPS

    def sample(self, rng: jax.Array) -> jax.Array:
        """Sample via the inverse-CDF Laplace reparameterization.

        Args:
            rng: PRNG key.

        Returns:
            Sampled array with the same shape as ``mean``.
        """
        u = random.uniform(rng, self.mean.shape, minval=1e-7, maxval=1 - 1e-7)
        return self.mean - self.scale * jnp.sign(u - 0.5) * jnp.log1p(
            -2 * jnp.abs(u - 0.5)
        )


def make_parameter(
    value: jax.Array,
    *,
    bayesian: bool = True,
    param_type: Type[AbstractParameter] = GaussianParameter,
    init_raw_stdv: float = 0.01,
) -> AbstractParameter:
    """Wrap an arbitrary array as a Bayesian or deterministic parameter.

    Args:
        value: Initial mean value. Any shape (scalar, vector, matrix, tensor).
        bayesian: If True (default), wrap in ``param_type``. If False, wrap
            in ``DeterministicParameter``.
        param_type: ``GaussianParameter`` (default) or ``LaplacianParameter``.
        init_raw_stdv: Initial value for the raw standard-deviation / scale
            field.

    Returns:
        An ``AbstractParameter`` instance wrapping ``value``.
    """
    value = jnp.asarray(value)
    if not bayesian:
        return DeterministicParameter(mean=value)
    if param_type is GaussianParameter:
        return GaussianParameter(
            mean=value,
            raw_stdv=jnp.full_like(value, init_raw_stdv),
        )
    if param_type is LaplacianParameter:
        return LaplacianParameter(
            mean=value,
            raw_scale=jnp.full_like(value, init_raw_stdv),
        )
    raise ValueError(f"Unknown param_type: {param_type}")
