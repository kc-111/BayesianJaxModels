"""Base Module class with parameter introspection utilities."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from .parameter import AbstractParameter, DeterministicParameter, GaussianParameter


class Module(eqx.Module):
    """Base class for Bayesian models.

    Provides recursive parameter introspection: ``get_parameters``,
    ``flatten_means``, ``flatten_stdvs``, and ``flatten_raw_stdvs``.
    """

    def get_parameters(self) -> dict[str, AbstractParameter]:
        """Return a flat dict mapping dotted names to parameter leaves.

        Returns:
            Dict like ``{"W": GaussianParameter(...), "b": DeterministicParameter(...)}``.
            Nested modules produce dotted names (``"layer1.W"``); lists
            produce indexed names (``"layers[0].W"``).
        """
        result: dict[str, AbstractParameter] = {}
        self._collect_parameters("", result)
        return result

    def _collect_parameters(
        self, prefix: str, result: dict[str, AbstractParameter]
    ) -> None:
        for field in dataclasses.fields(self):
            name = field.name
            full_name = f"{prefix}{name}" if prefix else name
            value = getattr(self, name)
            if isinstance(value, AbstractParameter):
                result[full_name] = value
            elif isinstance(value, Module):
                value._collect_parameters(f"{full_name}.", result)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    indexed = f"{full_name}[{i}]"
                    if isinstance(item, AbstractParameter):
                        result[indexed] = item
                    elif isinstance(item, Module):
                        item._collect_parameters(f"{indexed}.", result)

    def flatten_means(self) -> jax.Array:
        """Concatenate all parameter means into a single flat vector.

        Returns:
            1-D array of length ``sum(p.mean.size for p in parameters)``.
        """
        params = self.get_parameters()
        if not params:
            return jnp.array([])
        return jnp.concatenate(
            [p.mean.ravel() for p in params.values()]
        )

    def flatten_stdvs(self) -> jax.Array:
        """Concatenate all positive stdvs into a flat vector.

        Deterministic parameters contribute zeros.

        Returns:
            1-D array with the same length as ``flatten_means()``.
        """
        params = self.get_parameters()
        if not params:
            return jnp.array([])
        parts = []
        for p in params.values():
            if isinstance(p, GaussianParameter):
                parts.append(p.stdv.ravel())
            elif hasattr(p, "scale"):
                parts.append(p.scale.ravel())
            else:
                parts.append(jnp.zeros(p.mean.size))
        return jnp.concatenate(parts)

    def flatten_raw_stdvs(self) -> jax.Array:
        """Concatenate all raw (unconstrained) variational params into a flat vector.

        These are the values the optimiser updates directly.
        Deterministic parameters contribute zeros.

        Returns:
            1-D array with the same length as ``flatten_means()``.
        """
        params = self.get_parameters()
        if not params:
            return jnp.array([])
        parts = []
        for p in params.values():
            if hasattr(p, "raw_stdv"):
                parts.append(p.raw_stdv.ravel())
            elif hasattr(p, "raw_scale"):
                parts.append(p.raw_scale.ravel())
            else:
                parts.append(jnp.zeros(p.mean.size))
        return jnp.concatenate(parts)
