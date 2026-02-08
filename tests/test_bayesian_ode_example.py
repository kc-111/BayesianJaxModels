"""ODE integration example using BayesianJaxModels + diffrax.

Demonstrates the 'sample once, integrate deterministically' pattern:
1. Define a Bayesian ODE function (neural ODE right-hand-side)
2. Use sample_all_parameters to fix weights for one trajectory
3. Integrate with diffrax
4. vmap over keys for posterior trajectory samples
"""

import diffrax
import jax
import jax.numpy as jnp
import pytest
from jax import random

from BayesianJaxModels import (
    BayesianLinear,
    Module,
    sample_all_parameters,
)


class BayesianODEFunc(Module):
    """Bayesian neural ODE right-hand side: dx/dt = NN(x).

    NOT part of the library — just a user-defined model for testing.
    """

    layer1: BayesianLinear
    layer2: BayesianLinear

    def __init__(self, state_dim: int, hidden_dim: int, *, key: jax.Array):
        k1, k2 = random.split(key)
        self.layer1 = BayesianLinear(state_dim, hidden_dim, key=k1)
        self.layer2 = BayesianLinear(hidden_dim, state_dim, key=k2)

    def __call__(self, x: jax.Array, *, key=None, sample: bool = False) -> jax.Array:
        if sample and key is None:
            raise ValueError("key is required when sample=True")
        if sample:
            k1, k2 = random.split(key)
        else:
            k1 = k2 = None
        h = jax.nn.tanh(self.layer1(x, key=k1, sample=sample))
        return self.layer2(h, key=k2, sample=sample)


def integrate_ode(func, y0, t0, t1, dt0, saveat_ts):
    """Integrate an ODE using diffrax with Tsit5 solver."""

    def vector_field(t, y, args):
        return func(y, key=None, sample=False)

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=saveat_ts)
    sol = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=saveat
    )
    return sol.ys


class TestBayesianODE:
    """Tests demonstrating Bayesian ODE usage patterns."""

    def test_single_trajectory(self):
        """A single sampled model should produce a deterministic trajectory."""
        func = BayesianODEFunc(state_dim=2, hidden_dim=8, key=random.key(0))

        # Sample weights once
        sampled_func = sample_all_parameters(func, random.key(1))

        # Means should differ from the original (they are samples now)
        assert not jnp.allclose(
            sampled_func.layer1.W.mean, func.layer1.W.mean
        )

        # Integrate
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0, 1, 20)
        traj = integrate_ode(sampled_func, y0, t0=0.0, t1=1.0, dt0=0.05, saveat_ts=ts)

        assert traj.shape == (20, 2)
        assert jnp.all(jnp.isfinite(traj))

    def test_trajectory_reproducibility(self):
        """Same key should produce identical trajectories."""
        func = BayesianODEFunc(state_dim=2, hidden_dim=8, key=random.key(0))
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0, 1, 10)

        s1 = sample_all_parameters(func, random.key(42))
        t1 = integrate_ode(s1, y0, 0.0, 1.0, 0.05, ts)

        s2 = sample_all_parameters(func, random.key(42))
        t2 = integrate_ode(s2, y0, 0.0, 1.0, 0.05, ts)

        assert jnp.allclose(t1, t2)

    def test_different_keys_different_trajectories(self):
        """Different sample keys should yield different trajectories."""
        func = BayesianODEFunc(state_dim=2, hidden_dim=8, key=random.key(0))
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0, 1, 10)

        s1 = sample_all_parameters(func, random.key(1))
        t1 = integrate_ode(s1, y0, 0.0, 1.0, 0.05, ts)

        s2 = sample_all_parameters(func, random.key(2))
        t2 = integrate_ode(s2, y0, 0.0, 1.0, 0.05, ts)

        assert not jnp.allclose(t1, t2)

    def test_posterior_samples_via_vmap(self):
        """vmap over sample keys to get posterior trajectory samples."""
        func = BayesianODEFunc(state_dim=2, hidden_dim=8, key=random.key(0))
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0, 1, 10)
        n_samples = 5

        def sample_and_integrate(key):
            sampled = sample_all_parameters(func, key)
            return integrate_ode(sampled, y0, 0.0, 1.0, 0.05, ts)

        keys = random.split(random.key(0), n_samples)
        trajectories = jax.vmap(sample_and_integrate)(keys)

        assert trajectories.shape == (n_samples, 10, 2)
        assert jnp.all(jnp.isfinite(trajectories))

        # Trajectories should have non-zero variance across samples
        traj_std = trajectories.std(axis=0)
        assert jnp.any(traj_std > 1e-6)

    def test_ode_deterministic_forward(self):
        """Using means directly (no sampling) should work and be consistent."""
        func = BayesianODEFunc(state_dim=2, hidden_dim=8, key=random.key(0))
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0, 1, 10)

        # Use the original model with sample=False (means)
        t1 = integrate_ode(func, y0, 0.0, 1.0, 0.05, ts)
        t2 = integrate_ode(func, y0, 0.0, 1.0, 0.05, ts)

        assert jnp.allclose(t1, t2)
        assert t1.shape == (10, 2)

    def test_gradient_through_ode(self):
        """Gradients should flow through sample -> ODE integrate -> loss.

        This is the actual training pipeline: we sample weights via the
        reparameterisation trick, integrate the ODE with those fixed
        weights, compute a loss, and differentiate back to both mean and
        raw_stdv of the original model.
        """
        func = BayesianODEFunc(state_dim=2, hidden_dim=8, key=random.key(0))
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0, 1, 5)
        target = jnp.zeros((5, 2))

        def loss_fn(model, rng):
            sampled = sample_all_parameters(model, rng)
            traj = integrate_ode(sampled, y0, 0.0, 1.0, 0.05, ts)
            return jnp.mean((traj - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(func, random.key(42))
        assert jnp.isfinite(loss)

        # flatten_with_path returns (list[(path, leaf)], treedef)
        flat_grads, _ = jax.tree.flatten_with_path(grads)

        # Grads on means should be non-zero
        mean_grads = [
            g for path, g in flat_grads
            if "mean" in str(path) and isinstance(g, jax.Array)
        ]
        assert any(jnp.any(g != 0) for g in mean_grads), \
            "Expected non-zero gradients on means"

        # Grads on raw_stdv should also be non-zero (reparameterisation trick)
        stdv_grads = [
            g for path, g in flat_grads
            if "raw_stdv" in str(path) and isinstance(g, jax.Array)
        ]
        assert any(jnp.any(g != 0) for g in stdv_grads), \
            "Expected non-zero gradients on raw_stdv (reparameterisation)"
