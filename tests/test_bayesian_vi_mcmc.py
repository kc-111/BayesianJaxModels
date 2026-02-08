"""VI validation: compare variational posterior to analytical posterior.

For Bayesian linear regression with Gaussian prior and likelihood,
the posterior is analytically tractable. We verify that our VI framework
recovers the correct posterior mean and standard deviation.

Setup
-----
- Prior: w ~ N(0, (1/alpha) * I)
- Likelihood: y | x, w ~ N(w @ x, (1/beta) * I)
- Posterior (analytical):
    Sigma_post = inv(alpha * I + beta * X^T X)
    mu_post = beta * Sigma_post @ X^T y
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from jax import random

from BayesianJaxModels import (
    BayesianLinear,
    GaussianParameter,
    freeze_stdvs,
    freeze_means,
    gaussian_entropy,
    unfreeze_all,
)


def generate_data(key, n_samples=200, n_features=3, noise_std=0.5):
    """Generate synthetic linear regression data."""
    k1, k2, k3 = random.split(key, 3)
    w_true = random.normal(k1, (n_features,))
    X = random.normal(k2, (n_samples, n_features))
    y = X @ w_true + noise_std * random.normal(k3, (n_samples,))
    return X, y, w_true


def analytical_posterior(X, y, alpha, beta):
    """Compute the analytical Gaussian posterior for Bayesian linear regression."""
    n_features = X.shape[1]
    Sigma_post_inv = alpha * jnp.eye(n_features) + beta * X.T @ X
    Sigma_post = jnp.linalg.inv(Sigma_post_inv)
    mu_post = beta * Sigma_post @ X.T @ y
    stdv_post = jnp.sqrt(jnp.diag(Sigma_post))
    return mu_post, stdv_post, Sigma_post


def elbo_loss(dynamic, static, X, y, beta, alpha, key):
    """Negative ELBO = -E_q[log p(y|w,X)] - E_q[log p(w)] + E_q[log q(w)].

    For Gaussian q and Gaussian prior/likelihood, we use the reparameterisation
    trick and entropy in closed form.
    """
    model = unfreeze_all(dynamic, static)

    # Sample weights
    y_pred = model(X, key=key, sample=True).squeeze(-1)

    # Log-likelihood: -0.5 * beta * ||y - y_pred||^2  (up to constants)
    log_lik = -0.5 * beta * jnp.sum((y - y_pred) ** 2)

    # Log-prior: -0.5 * alpha * ||w||^2
    W_sample = model.W.sample(key)
    log_prior = -0.5 * alpha * jnp.sum(W_sample ** 2)

    # Entropy of q (Gaussian)
    entropy = gaussian_entropy(model)

    # ELBO = log_lik + log_prior + entropy
    return -(log_lik + log_prior + entropy)


class TestVIvsMCMC:
    """Compare VI posterior to analytical posterior on linear regression."""

    @pytest.fixture
    def problem_setup(self):
        key = random.key(42)
        alpha = 1.0  # prior precision
        beta = 4.0   # likelihood precision (noise_std = 0.5 => beta = 1/0.25 = 4)
        noise_std = 1.0 / jnp.sqrt(beta)
        X, y, w_true = generate_data(key, n_samples=200, n_features=3, noise_std=noise_std)
        mu_post, stdv_post, _ = analytical_posterior(X, y, alpha, beta)
        return X, y, w_true, alpha, beta, mu_post, stdv_post

    def test_vi_recovers_posterior_mean(self, problem_setup):
        """VI mean should be close to analytical posterior mean."""
        X, y, _, alpha, beta, mu_post, stdv_post = problem_setup

        # Build model: single Bayesian linear layer (no bias for simplicity)
        model = BayesianLinear(
            3, 1, use_bias=False, bayesian=True, param_type=GaussianParameter,
            init_raw_stdv=0.1, key=random.key(0)
        )

        # Stage 1: MAP — freeze stdvs, optimize means
        dynamic, static = freeze_stdvs(model)
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(dynamic)

        @jax.jit
        def map_step(dynamic, opt_state, key):
            loss, grads = jax.value_and_grad(elbo_loss)(
                dynamic, static, X, y, beta, alpha, key
            )
            updates, opt_state = optimizer.update(grads, opt_state, dynamic)
            dynamic = optax.apply_updates(dynamic, updates)
            return dynamic, opt_state, loss

        key = random.key(1)
        for i in range(2000):
            key, subkey = random.split(key)
            dynamic, opt_state, loss = map_step(dynamic, opt_state, subkey)

        model = unfreeze_all(dynamic, static)

        # Check MAP estimate ≈ posterior mean
        vi_mean = model.W.mean.ravel()
        assert jnp.allclose(vi_mean, mu_post, atol=0.15), (
            f"VI mean {vi_mean} vs analytical {mu_post}"
        )

    def test_vi_recovers_posterior_stdv(self, problem_setup):
        """After VI (MAP then stdv stage), stdvs should match analytical posterior."""
        X, y, _, alpha, beta, mu_post, stdv_post = problem_setup

        model = BayesianLinear(
            3, 1, use_bias=False, bayesian=True, param_type=GaussianParameter,
            init_raw_stdv=0.1, key=random.key(0)
        )

        # Stage 1: MAP
        dynamic, static = freeze_stdvs(model)
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(dynamic)

        @jax.jit
        def map_step(dynamic, opt_state, key):
            loss, grads = jax.value_and_grad(elbo_loss)(
                dynamic, static, X, y, beta, alpha, key
            )
            updates, opt_state = optimizer.update(grads, opt_state, dynamic)
            dynamic = optax.apply_updates(dynamic, updates)
            return dynamic, opt_state, loss

        key = random.key(1)
        for i in range(2000):
            key, subkey = random.split(key)
            dynamic, opt_state, loss = map_step(dynamic, opt_state, subkey)

        model = unfreeze_all(dynamic, static)

        # Stage 2: VI — freeze means, optimize stdvs
        dynamic, static = freeze_means(model)
        optimizer2 = optax.adam(1e-3)
        opt_state2 = optimizer2.init(dynamic)

        @jax.jit
        def vi_step(dynamic, opt_state, key):
            # Average over multiple samples for lower variance
            def mc_loss(key):
                return elbo_loss(dynamic, static, X, y, beta, alpha, key)

            keys = random.split(key, 8)
            loss = jnp.mean(jax.vmap(mc_loss)(keys))
            grads = jax.grad(lambda d: jnp.mean(jax.vmap(
                lambda k: elbo_loss(d, static, X, y, beta, alpha, k)
            )(keys)))(dynamic)

            updates, opt_state = optimizer2.update(grads, opt_state, dynamic)
            dynamic = optax.apply_updates(dynamic, updates)
            return dynamic, opt_state, loss

        key = random.key(2)
        for i in range(3000):
            key, subkey = random.split(key)
            dynamic, opt_state2, loss = vi_step(dynamic, opt_state2, subkey)

        model = unfreeze_all(dynamic, static)

        vi_stdv = model.W.stdv.ravel()
        # Tolerant comparison — VI may not perfectly match analytical
        assert jnp.allclose(vi_stdv, stdv_post, atol=0.1), (
            f"VI stdv {vi_stdv} vs analytical {stdv_post}"
        )

    def test_vi_mean_better_than_random(self, problem_setup):
        """Even minimal training should improve over random initialisation."""
        X, y, _, alpha, beta, mu_post, _ = problem_setup

        model = BayesianLinear(
            3, 1, use_bias=False, bayesian=True, key=random.key(99)
        )

        # Random model prediction error
        y_rand = model(X, key=None, sample=False).squeeze(-1)
        mse_rand = jnp.mean((y - y_rand) ** 2)

        # Minimal MAP training (100 steps)
        dynamic, static = freeze_stdvs(model)
        optimizer = optax.adam(1e-2)
        opt_state = optimizer.init(dynamic)

        key = random.key(1)
        for _ in range(200):
            key, subkey = random.split(key)
            loss, grads = jax.value_and_grad(elbo_loss)(
                dynamic, static, X, y, beta, alpha, subkey
            )
            updates, opt_state = optimizer.update(grads, opt_state, dynamic)
            dynamic = optax.apply_updates(dynamic, updates)

        model = unfreeze_all(dynamic, static)
        y_trained = model(X, key=None, sample=False).squeeze(-1)
        mse_trained = jnp.mean((y - y_trained) ** 2)

        assert mse_trained < mse_rand, (
            f"Trained MSE {mse_trained:.4f} should be less than random MSE {mse_rand:.4f}"
        )
