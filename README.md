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
