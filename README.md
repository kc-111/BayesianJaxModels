# BayesianJaxModels

Modular Bayesian model framework in JAX, built on [Equinox](https://github.com/patrick-kidger/equinox).

Provides parameter types with reparameterized sampling, a `Module` base class with
parameter introspection, Bayesian layers, and utility functions for freeze/unfreeze,
sampling, and entropy computation. You own the training loop — this package provides
only the differentiable model and utilities.

## Installation

```bash
pip install git+https://github.com/kc-111/BayesianJaxModels.git
```

With test dependencies:

```bash
pip install "BayesianJaxModels[test] @ git+https://github.com/kc-111/BayesianJaxModels.git"
```

To list as a dependency in another project's `pyproject.toml`:

```toml
dependencies = [
    "BayesianJaxModels @ git+https://github.com/kc-111/BayesianJaxModels.git",
]
```

## Dependencies

- `jax`
- `equinox >= 0.13.0`
- `optax` (for training)

## Quick start

```python
import jax
import jax.numpy as jnp
from jax import random
from BayesianJaxModels import (
    BayesianLinear, Module, make_parameter,
    freeze_stdvs, freeze_means, unfreeze_all,
    sample_all_parameters, gaussian_entropy,
)
```

### Defining a model

Subclass `Module` and compose with `BayesianLinear` or `make_parameter`:

```python
class MLP(Module):
    layers: list

    def __init__(self, dims: list[int], *, key: jax.Array):
        keys = random.split(key, len(dims) - 1)
        self.layers = [
            BayesianLinear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x, *, key, sample=True):
        keys = random.split(key, len(self.layers)) if sample else [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x, key=keys[i], sample=sample)
            if i < len(self.layers) - 1:
                x = jax.nn.relu(x)
        return x

model = MLP([4, 32, 1], key=random.key(0))
```

### Forward pass

```python
# Stochastic (samples from variational distribution):
y = model(x, key=random.key(1), sample=True)

# Deterministic (uses means only):
y = model(x, key=None, sample=False)
```

### Custom parameters with `make_parameter`

Wrap any tensor as a Bayesian or deterministic parameter:

```python
from BayesianJaxModels import AbstractParameter, LaplacianParameter

class MyModel(Module):
    A: AbstractParameter
    scale: AbstractParameter

    def __init__(self, *, key):
        self.A = make_parameter(jnp.zeros((3, 4)))                          # Gaussian by default
        self.scale = make_parameter(jnp.array(1.0), bayesian=False)         # deterministic
        # self.B = make_parameter(jnp.eye(3), param_type=LaplacianParameter)  # Laplace
```

### Two-stage VI training

**Stage 1 — MAP (optimize means, freeze stdvs):**

```python
import optax

dynamic, static = freeze_stdvs(model)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(dynamic)

@jax.jit
def map_step(dynamic, opt_state, key):
    def loss_fn(dynamic):
        m = unfreeze_all(dynamic, static)
        y_pred = m(x_train, key=key, sample=True)
        return jnp.mean((y_train - y_pred) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(dynamic)
    updates, opt_state_ = optimizer.update(grads, opt_state, dynamic)
    return optax.apply_updates(dynamic, updates), opt_state_, loss

for i in range(1000):
    key, subkey = random.split(key)
    dynamic, opt_state, loss = map_step(dynamic, opt_state, subkey)

model = unfreeze_all(dynamic, static)
```

**Stage 2 — VI (optimize stdvs, freeze means):**

```python
dynamic, static = freeze_means(model)
optimizer2 = optax.adam(1e-4)
opt_state2 = optimizer2.init(dynamic)

@jax.jit
def vi_step(dynamic, opt_state, key):
    def loss_fn(dynamic):
        m = unfreeze_all(dynamic, static)
        y_pred = m(x_train, key=key, sample=True)
        nll = 0.5 * beta * jnp.sum((y_train - y_pred) ** 2)
        return nll - gaussian_entropy(m)

    loss, grads = jax.value_and_grad(loss_fn)(dynamic)
    updates, opt_state_ = optimizer2.update(grads, opt_state, dynamic)
    return optax.apply_updates(dynamic, updates), opt_state_, loss
```

Full example:
```python
import optax
import jax
import jax.numpy as jnp
import equinox as eqx
from tqdm import tqdm
from BayesianJaxModels import (
    freeze_stdvs,
    freeze_means,
    sample_all_parameters,
    gaussian_entropy,
    laplacian_entropy,
)


def train_map(
    model, X, Y, *,
    key, num_epochs, batch_size,
    learning_rate, prior_weight,
):
    """
    Train the model using the Maximum a posteriori (MAP) estimation.
    Objective: -log p(y|x, z) + λ * -log p(z)
    we assume -log p(y|x, z) is mean squared error
    and -log p(z) is ||z||_2^2.

    Args:
        model: The model to train.
        X: The input data (samples, features).
        Y: The output data (samples, targets).
        key: The random key.
        num_epochs: The number of epochs to train for.
        batch_size: The batch size.
        learning_rate: The learning rate for the optimizer.
        prior_weight: The weight of the prior.

    Returns:
        The trained model.
    """
    # Freeze variational parameters — only optimize means
    dynamic, static = freeze_stdvs(model)

    # Define optimizer adabelief with no weight decay
    optimizer = optax.adabelief(learning_rate)
    opt_state = optimizer.init(dynamic)

    # Define jitted loss function for map
    @eqx.filter_jit
    def step(dynamic, static, opt_state, X_batch, Y_batch):
        def loss_fn(dynamic):
            model = eqx.combine(dynamic, static)
            pred = model(X_batch, key=jax.random.key(0), sample=False)
            mse = jnp.mean((pred - Y_batch) ** 2)
            l2 = model.compute_prior()
            return mse + prior_weight * l2

        loss, grads = eqx.filter_value_and_grad(loss_fn)(dynamic)
        updates, opt_state = optimizer.update(grads, opt_state, dynamic)
        dynamic = eqx.apply_updates(dynamic, updates)
        return dynamic, opt_state, loss

    # Run training loop for num_epochs and minibatches
    n = X.shape[0]
    pbar = tqdm(range(num_epochs), desc="MAP")
    for epoch in pbar:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n)
        for i in range(n // batch_size):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            dynamic, opt_state, loss = step(
                dynamic, static, opt_state, X[idx], Y[idx]
            )
        pbar.set_postfix(loss=f"{float(loss):.4f}")

    # Return the trained model
    return eqx.combine(dynamic, static)


def train_vi(
    model, X, Y, *,
    key, num_epochs, batch_size,
    learning_rate, prior_weight,
    num_samples=1,
):
    """
    Train the model using the Variational Inference (VI) algorithm.
    p(z) = 1/Z * exp(-g(z)), so log p(z) = -g(z) - log Z
    KL(q || p) = -E_q[log q(z)] + E_q[log p(z)] ∝ -H(q) + E_q[-g(z)]
    Tempered ELBO = E_q[log p(y|x, z)] - λ * KL(q || p)
    = E_q[log p(y|x, z)] - λ * (-H(q) + E_q[g(z)])
    = E_q[log p(y|x, z)] + λ * (H(q) - E_q[g(z)])
    Objective: -E_q[log p(y|x, z)] + λ * (E_q[g(z)] - H(q))

    log p(y|x, z) is mean squared error.
    g(z) = ||z||_2^2.
    H(q) is the entropy of the variational distribution.

    Means are frozen (set via MAP); only variational widths are optimized.
    Both likelihood and prior expectations are estimated via Monte Carlo
    with independent samples from q.

    Args:
        model: The model to train (typically MAP-pretrained).
        X: The input data (samples, features).
        Y: The output data (samples, targets).
        key: The random key.
        num_epochs: The number of epochs to train for.
        batch_size: The batch size.
        learning_rate: The learning rate for the optimizer.
        prior_weight: The weight of the prior.
        num_samples: Number of MC samples for likelihood and prior
            expectations (independent draws for each).

    Returns:
        The trained model.
    """
    # Freeze means — only optimize variational widths (raw_stdv / raw_scale)
    dynamic, static = freeze_means(model)

    # Define optimizer adabelief with no weight decay
    optimizer = optax.adabelief(learning_rate)
    opt_state = optimizer.init(dynamic)

    # Define jitted loss function for vi
    @eqx.filter_jit
    def step(dynamic, static, opt_state, X_batch, Y_batch, key):
        def loss_fn(dynamic):
            model = eqx.combine(dynamic, static)
            k_lik, k_prior = jax.random.split(key)
            k_liks = jax.random.split(k_lik, num_samples)
            k_priors = jax.random.split(k_prior, num_samples)

            # Likelihood: E_q[-log p(y|x,z)] ≈ (1/K) Σ MSE(z_k)
            def lik_sample(k):
                sampled = sample_all_parameters(model, k)
                pred = sampled(X_batch, key=jax.random.key(0), sample=False)
                return jnp.mean((pred - Y_batch) ** 2)

            mse = jnp.mean(jax.vmap(lik_sample)(k_liks))

            # Prior: E_q[g(z)] ≈ (1/K) Σ g(z_k) on effective params
            def prior_sample(k):
                sampled = sample_all_parameters(model, k)
                return sampled.compute_prior()

            prior = jnp.mean(jax.vmap(prior_sample)(k_priors))

            # Entropy: H(q) (analytical)
            entropy = gaussian_entropy(model) + laplacian_entropy(model)

            return mse + prior_weight * (prior - entropy)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(dynamic)
        updates, opt_state = optimizer.update(grads, opt_state, dynamic)
        dynamic = eqx.apply_updates(dynamic, updates)
        return dynamic, opt_state, loss

    # Run training loop for num_epochs and minibatches
    n = X.shape[0]
    pbar = tqdm(range(num_epochs), desc="VI")
    for epoch in pbar:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n)
        for i in range(n // batch_size):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            key, subkey = jax.random.split(key)
            dynamic, opt_state, loss = step(
                dynamic, static, opt_state, X[idx], Y[idx], subkey
            )
        pbar.set_postfix(loss=f"{float(loss):.4f}")

    # Return the trained model
    return eqx.combine(dynamic, static)
```

### ODE integration pattern

For models where parameters must be fixed across time steps (ODEs, RNNs),
sample once then integrate deterministically. Gradients flow back through
the reparameterization to both `mean` and `raw_stdv`:

```python
import diffrax

def loss_fn(model, rng):
    sampled = sample_all_parameters(model, rng)   # sample weights once

    def vector_field(t, y, args):
        return sampled(y, key=None, sample=False)  # fixed weights

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field), diffrax.Tsit5(),
        t0=0.0, t1=1.0, dt0=0.01, y0=y0,
    )
    return jnp.mean((sol.ys - target) ** 2)

loss, grads = jax.value_and_grad(loss_fn)(model, random.key(0))
# grads contain derivatives w.r.t. both mean AND raw_stdv
```

Posterior trajectory samples via `vmap`:

```python
def sample_trajectory(key):
    sampled = sample_all_parameters(model, key)
    return integrate(sampled, y0)

keys = random.split(random.key(0), 100)
trajectories = jax.vmap(sample_trajectory)(keys)  # (100, T, state_dim)
```

### Introspection

```python
params = model.get_parameters()      # {"layers[0].W": GaussianParameter, ...}
means  = model.flatten_means()       # 1-D array of all means
stdvs  = model.flatten_stdvs()       # 1-D array of all stdvs (0 for deterministic)
counts = get_parameter_count(model)  # {"total": N, "bayesian": M, "deterministic": K}
groups = get_parameter_groups(model) # {"gaussian": [...], "laplacian": [...], "deterministic": [...]}
```

## Parameter types

| Type | Distribution | Variational field | Positivity transform |
|------|-------------|-------------------|---------------------|
| `GaussianParameter` | N(mean, stdv) | `raw_stdv` | `stdv = raw_stdv**2 + 1e-4` |
| `LaplacianParameter` | Laplace(mean, scale) | `raw_scale` | `scale = raw_scale**2 + 1e-4` |
| `DeterministicParameter` | Point mass | (none) | N/A |

## BayesianLinear

```python
BayesianLinear(
    in_features, out_features, *,
    use_bias=True,         # include bias term
    bayesian=True,         # Bayesian weights (False -> DeterministicParameter)
    bayesian_bias=True,    # Bayesian bias
    param_type=GaussianParameter,  # or LaplacianParameter
    init_raw_stdv=0.01,    # initial raw stdv/scale value
    key=...,               # PRNG key (required, used for Xavier init)
)
```

## Utility functions

| Function | Description |
|----------|-------------|
| `freeze_stdvs(model)` | Partition model: stdvs static, means dynamic |
| `freeze_means(model)` | Partition model: means static, stdvs dynamic |
| `freeze_params(model, names)` | Freeze specific named parameters |
| `unfreeze_all(dynamic, static)` | Recombine partitions (`eqx.combine`) |
| `sample_all_parameters(model, rng)` | Differentiable: replace means with reparameterized samples |
| `gaussian_entropy(model)` | `sum(log(stdv))` over Gaussian parameters |
| `laplacian_entropy(model)` | `sum(log(scale))` over Laplacian parameters |
| `flatten_means(model)` | All means as flat 1-D array |
| `flatten_stdvs(model)` | All stdvs as flat 1-D array |
| `get_parameter_count(model)` | Count total/bayesian/deterministic parameters |
| `get_parameter_groups(model)` | Group parameter names by distribution type |

## Performance notes

The recursive Python traversal in `get_parameters()`, `sample_all_parameters()`, and
entropy functions runs **only once during JAX tracing**. When called inside
`jax.jit` or `jax.value_and_grad`, JAX traces the Python code to build an XLA computation
graph, then caches and reuses the compiled program on subsequent calls with zero Python
overhead. The actual sampling and arithmetic are compiled into fused XLA kernels.

## Device placement

JAX is device-agnostic. All arrays are allocated on the default device (GPU if available,
otherwise CPU). Since equinox modules are pytrees of JAX arrays, the entire model
automatically lives on the correct device. No code changes are needed to switch between
CPU, GPU, or TPU.

## Running tests

```bash
pip install bayesian-jax-models[test]
pytest tests/ -v
```

## License

MIT
