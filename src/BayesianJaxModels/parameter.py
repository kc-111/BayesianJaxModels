"""Bayesian parameter classes built on equinox.

Each parameter stores a mean and an unconstrained log-scale parameter.
Positivity of sigma / scale is guaranteed via the exp transform.
Sampling uses reparameterization tricks for gradient flow.
"""

from abc import abstractmethod
from typing import Type

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random


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

    The actual standard deviation is ``exp(log_sigma)``, which guarantees
    positivity via the exponential transform.  The unconstrained
    ``log_sigma`` is the quantity the optimiser updates directly.
    """

    log_sigma: jax.Array

    @property
    def stdv(self) -> jax.Array:
        """Positive standard deviation: ``exp(log_sigma)``."""
        return jnp.exp(self.log_sigma)

    def sample(self, rng: jax.Array) -> jax.Array:
        """Sample via ``mean + exp(log_sigma) * N(0, 1)``.

        Args:
            rng: PRNG key.

        Returns:
            Sampled array with the same shape as ``mean``.
        """
        return self.mean + self.stdv * random.normal(rng, self.mean.shape)


class LaplacianParameter(AbstractParameter):
    """Laplace variational parameter using inverse-CDF reparameterization.

    The actual scale is ``exp(log_scale)``, guaranteed positive via the
    exponential transform.
    """

    log_scale: jax.Array

    @property
    def scale(self) -> jax.Array:
        """Positive scale: ``exp(log_scale)``."""
        return jnp.exp(self.log_scale)

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
    init_log_sigma: float = -5.0,
) -> AbstractParameter:
    """Wrap an arbitrary array as a Bayesian or deterministic parameter.

    Args:
        value: Initial mean value. Any shape (scalar, vector, matrix, tensor).
        bayesian: If True (default), wrap in ``param_type``. If False, wrap
            in ``DeterministicParameter``.
        param_type: ``GaussianParameter`` (default) or ``LaplacianParameter``.
        init_log_sigma: Initial value for the unconstrained log-scale field
            (``log_sigma`` or ``log_scale``).  The effective initial standard
            deviation / scale is ``exp(init_log_sigma)``.

    Returns:
        An ``AbstractParameter`` instance wrapping ``value``.
    """
    value = jnp.asarray(value)
    if not bayesian:
        return DeterministicParameter(mean=value)
    if param_type is GaussianParameter:
        return GaussianParameter(
            mean=value,
            log_sigma=jnp.full_like(value, init_log_sigma),
        )
    if param_type is LaplacianParameter:
        return LaplacianParameter(
            mean=value,
            log_scale=jnp.full_like(value, init_log_sigma),
        )
    raise ValueError(f"Unknown param_type: {param_type}")
