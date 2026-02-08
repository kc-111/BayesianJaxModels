"""Utility functions: freeze/unfreeze, sampling, entropy, parameter counting."""

import re

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from .module import Module
from .parameter import (
    AbstractParameter,
    DeterministicParameter,
    GaussianParameter,
    LaplacianParameter,
)


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def _path_contains(path_components: tuple, *names: str) -> bool:
    """Check whether any component of a pytree path matches one of *names*.

    Args:
        path_components: Tuple of pytree path elements.
        *names: Substrings to search for.

    Returns:
        True if any path component contains any of the given names.
    """
    for comp in path_components:
        s = str(comp)
        for name in names:
            if name in s:
                return True
    return False


def _build_filter(model: Module, predicate) -> Module:
    """Build a boolean filter_spec pytree: True where predicate(path) holds.

    Args:
        model: The model to build a filter for.
        predicate: Callable taking a pytree path and returning bool.

    Returns:
        A pytree of bools with the same structure as ``model``.
    """
    flat, tree_def = jax.tree.flatten_with_path(model)
    specs = []
    for path, _ in flat:
        specs.append(predicate(path))
    return jax.tree.unflatten(tree_def, specs)


def freeze_stdvs(model: Module) -> tuple:
    """Partition model so that raw_stdv / raw_scale fields are static.

    Args:
        model: The model to partition.

    Returns:
        Tuple of ``(dynamic, static)`` for use with ``eqx.combine``.
    """
    filter_spec = _build_filter(
        model, lambda path: not _path_contains(path, "raw_stdv", "raw_scale")
    )
    return eqx.partition(model, filter_spec)


def freeze_means(model: Module) -> tuple:
    """Partition model so that mean fields are static.

    Args:
        model: The model to partition.

    Returns:
        Tuple of ``(dynamic, static)`` for use with ``eqx.combine``.
    """
    filter_spec = _build_filter(
        model, lambda path: not _path_contains(path, ".mean", "mean")
    )
    return eqx.partition(model, filter_spec)


def freeze_params(model: Module, names: list[str]) -> tuple:
    """Freeze specific named parameters (both mean and variational fields).

    Args:
        model: The model to partition.
        names: Parameter names matching keys from ``model.get_parameters()``,
            e.g. ``["layers[0].W", "layers[1].b"]``.

    Returns:
        Tuple of ``(dynamic, static)`` for use with ``eqx.combine``.
    """
    param_names = set(names)

    def _predicate(path):
        path_str = ".".join(str(c) for c in path)
        for pname in param_names:
            if pname in path_str:
                return False
        return True

    filter_spec = _build_filter(model, _predicate)
    return eqx.partition(model, filter_spec)


def unfreeze_all(dynamic, static) -> Module:
    """Recombine a partitioned model.

    Args:
        dynamic: Dynamic partition from a freeze function.
        static: Static partition from a freeze function.

    Returns:
        Recombined model.
    """
    return eqx.combine(dynamic, static)


# ---------------------------------------------------------------------------
# Parameter extraction (module-level convenience wrappers)
# ---------------------------------------------------------------------------

def flatten_means(model: Module) -> jax.Array:
    """Concatenate all parameter means into a flat vector.

    Args:
        model: The model to extract means from.

    Returns:
        1-D array of all concatenated means.
    """
    return model.flatten_means()


def flatten_stdvs(model: Module) -> jax.Array:
    """Concatenate all positive stdvs into a flat vector.

    Args:
        model: The model to extract stdvs from.

    Returns:
        1-D array of all concatenated stdvs (zeros for deterministic params).
    """
    return model.flatten_stdvs()


def get_parameter_count(model: Module) -> dict:
    """Count parameters by type.

    Args:
        model: The model to count parameters for.

    Returns:
        Dict with keys ``"total"``, ``"bayesian"``, ``"deterministic"``.
    """
    params = model.get_parameters()
    total = 0
    bayesian = 0
    deterministic = 0
    for p in params.values():
        n = p.mean.size
        total += n
        if isinstance(p, DeterministicParameter):
            deterministic += n
        else:
            bayesian += n
    return {"total": total, "bayesian": bayesian, "deterministic": deterministic}


def get_parameter_groups(model: Module) -> dict:
    """Group parameter names by distribution type.

    Args:
        model: The model to group parameters for.

    Returns:
        Dict with keys ``"gaussian"``, ``"laplacian"``, ``"deterministic"``,
        each mapping to a list of parameter name strings.
    """
    params = model.get_parameters()
    groups: dict[str, list[str]] = {
        "gaussian": [],
        "laplacian": [],
        "deterministic": [],
    }
    for name, p in params.items():
        if isinstance(p, GaussianParameter):
            groups["gaussian"].append(name)
        elif isinstance(p, LaplacianParameter):
            groups["laplacian"].append(name)
        else:
            groups["deterministic"].append(name)
    return groups


# ---------------------------------------------------------------------------
# Sampling & entropy
# ---------------------------------------------------------------------------

def sample_all_parameters(model: Module, rng: jax.Array) -> Module:
    """Return a copy of model with all means replaced by reparameterized samples.

    The sampled values are computed via the reparameterization trick
    (``mean + stdv * noise``), so gradients flow back through the sample
    to both ``mean`` and ``raw_stdv`` of the original model. The returned
    model can be called with ``sample=False`` (it will use the sampled
    means).

    Args:
        model: The model to sample parameters for.
        rng: PRNG key.

    Returns:
        A copy of ``model`` whose parameter means are reparameterized
        samples. Useful for ODE integration or recurrent models where
        weights must be fixed across time-steps.
    """
    params = model.get_parameters()
    if not params:
        return model

    keys = random.split(rng, len(params))
    new_model = model
    for (name, param), key in zip(params.items(), keys):
        sampled_value = param.sample(key)
        new_param = _set_mean(param, sampled_value)
        new_model = _replace_parameter(new_model, name, new_param)
    return new_model


def _set_mean(param: AbstractParameter, new_mean: jax.Array) -> AbstractParameter:
    """Return a copy of param with its mean replaced.

    Preserves all other fields (raw_stdv, raw_scale) so the pytree
    structure stays intact and gradients can flow through.

    Args:
        param: The parameter to copy.
        new_mean: The new mean value.

    Returns:
        A new parameter with the same type and fields, but with
        ``mean`` set to ``new_mean``.
    """
    return eqx.tree_at(lambda p: p.mean, param, new_mean)


def _replace_parameter(model: Module, name: str, new_param: AbstractParameter) -> Module:
    """Replace a single named parameter inside model using eqx.tree_at.

    Args:
        model: The model containing the parameter.
        name: Dotted/indexed name (e.g. ``"layers[0].W"``).
        new_param: The replacement parameter.

    Returns:
        A new model with the specified parameter replaced.
    """
    parts = _parse_name(name)
    return eqx.tree_at(lambda m: _navigate(m, parts), model, new_param)


def _navigate(model: Module, parts: list):
    """Navigate the model pytree by a sequence of attribute/index accesses.

    Args:
        model: Root of the pytree.
        parts: List of str (attribute names) or int (list indices).

    Returns:
        The leaf reached by following ``parts`` from ``model``.
    """
    obj = model
    for part in parts:
        if isinstance(part, str):
            obj = getattr(obj, part)
        else:
            obj = obj[part]
    return obj


def _parse_name(name: str) -> list:
    """Parse a dotted/indexed name into a list of accessors.

    Args:
        name: e.g. ``"layers[0].W"``.

    Returns:
        List like ``["layers", 0, "W"]``.
    """
    tokens = re.split(r"\.|(?=\[)", name)
    result = []
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("[") and tok.endswith("]"):
            result.append(int(tok[1:-1]))
        else:
            result.append(tok)
    return result


def gaussian_entropy(model: Module) -> jax.Array:
    """Compute sum(log(stdv)) over all Gaussian parameters.

    This is the variable part of the Gaussian entropy
    ``H(q) = 0.5 * d * (1 + log(2*pi)) + sum(log(sigma))``.
    The caller can add the constant if needed.

    Args:
        model: The model to compute entropy for.

    Returns:
        Scalar sum of log-stdvs across all Gaussian parameters.
    """
    params = model.get_parameters()
    total = jnp.array(0.0)
    for p in params.values():
        if isinstance(p, GaussianParameter):
            total = total + jnp.sum(jnp.log(p.stdv))
    return total


def laplacian_entropy(model: Module) -> jax.Array:
    """Compute sum(log(scale)) over all Laplacian parameters.

    This is the variable part of the Laplace entropy
    ``H(q) = d * (1 + log(2)) + sum(log(b))``.
    The caller can add the constant if needed.

    Args:
        model: The model to compute entropy for.

    Returns:
        Scalar sum of log-scales across all Laplacian parameters.
    """
    params = model.get_parameters()
    total = jnp.array(0.0)
    for p in params.values():
        if isinstance(p, LaplacianParameter):
            total = total + jnp.sum(jnp.log(p.scale))
    return total
