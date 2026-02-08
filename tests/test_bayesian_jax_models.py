"""Comprehensive unit tests for BayesianJaxModels."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from BayesianJaxModels import (
    AbstractParameter,
    BayesianLinear,
    DeterministicParameter,
    GaussianParameter,
    LaplacianParameter,
    Module,
    flatten_means,
    flatten_stdvs,
    freeze_means,
    freeze_stdvs,
    gaussian_entropy,
    get_parameter_count,
    get_parameter_groups,
    laplacian_entropy,
    make_parameter,
    sample_all_parameters,
    unfreeze_all,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleMLP(Module):
    """Two-layer MLP for testing nested module introspection."""

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


class MixedModel(Module):
    """Model mixing Gaussian, Laplacian, and Deterministic parameters."""

    gauss_layer: BayesianLinear
    laplace_layer: BayesianLinear
    det_layer: BayesianLinear

    def __init__(self, *, key):
        k1, k2, k3 = random.split(key, 3)
        self.gauss_layer = BayesianLinear(4, 3, param_type=GaussianParameter, key=k1)
        self.laplace_layer = BayesianLinear(3, 2, param_type=LaplacianParameter, key=k2)
        self.det_layer = BayesianLinear(2, 1, bayesian=False, key=k3)

    def __call__(self, x, *, key, sample=True):
        if sample:
            k1, k2, k3 = random.split(key, 3)
        else:
            k1 = k2 = k3 = None
        x = jax.nn.relu(self.gauss_layer(x, key=k1, sample=sample))
        x = jax.nn.relu(self.laplace_layer(x, key=k2, sample=sample))
        x = self.det_layer(x, key=k3, sample=sample)
        return x


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------


class TestParameters:
    def test_deterministic_sample_returns_mean(self):
        p = DeterministicParameter(mean=jnp.array([1.0, 2.0]))
        assert jnp.allclose(p.sample(random.key(0)), p.mean)

    def test_gaussian_sample_shape(self):
        p = GaussianParameter(
            mean=jnp.zeros((3, 4)), raw_stdv=jnp.ones((3, 4)) * 0.1
        )
        s = p.sample(random.key(42))
        assert s.shape == (3, 4)

    def test_gaussian_stdv_positive(self):
        p = GaussianParameter(
            mean=jnp.zeros(5), raw_stdv=jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        )
        assert jnp.all(p.stdv > 0)

    def test_gaussian_sample_distribution(self):
        """Many samples should have mean ~ param.mean and std ~ param.stdv."""
        mean = jnp.array([1.0, -2.0])
        raw_stdv = jnp.array([0.5, 1.0])
        p = GaussianParameter(mean=mean, raw_stdv=raw_stdv)
        keys = random.split(random.key(0), 10000)
        samples = jax.vmap(p.sample)(keys)
        assert jnp.allclose(samples.mean(axis=0), mean, atol=0.05)
        assert jnp.allclose(samples.std(axis=0), p.stdv, atol=0.05)

    def test_laplacian_sample_shape(self):
        p = LaplacianParameter(
            mean=jnp.zeros(5), raw_scale=jnp.ones(5) * 0.1
        )
        s = p.sample(random.key(0))
        assert s.shape == (5,)

    def test_laplacian_scale_positive(self):
        p = LaplacianParameter(
            mean=jnp.zeros(3), raw_scale=jnp.array([-1.0, 0.0, 1.0])
        )
        assert jnp.all(p.scale > 0)

    def test_laplacian_sample_distribution(self):
        """Laplacian samples should have mean ~ param.mean."""
        mean = jnp.array([3.0])
        p = LaplacianParameter(mean=mean, raw_scale=jnp.array([0.5]))
        keys = random.split(random.key(0), 10000)
        samples = jax.vmap(p.sample)(keys)
        assert jnp.allclose(samples.mean(axis=0), mean, atol=0.1)

    def test_parameter_shape_property(self):
        p = GaussianParameter(mean=jnp.zeros((2, 3)), raw_stdv=jnp.ones((2, 3)))
        assert p.shape == (2, 3)


# ---------------------------------------------------------------------------
# BayesianLinear tests
# ---------------------------------------------------------------------------


class TestBayesianLinear:
    def test_forward_sample_true(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        x = jnp.ones((2, 4))
        y = layer(x, key=random.key(1), sample=True)
        assert y.shape == (2, 3)

    def test_forward_sample_false(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        x = jnp.ones((2, 4))
        y = layer(x, key=None, sample=False)
        assert y.shape == (2, 3)

    def test_deterministic_mode(self):
        """sample=False should give identical outputs each call."""
        layer = BayesianLinear(4, 3, key=random.key(0))
        x = jnp.ones((2, 4))
        y1 = layer(x, key=None, sample=False)
        y2 = layer(x, key=None, sample=False)
        assert jnp.allclose(y1, y2)

    def test_stochastic_mode_varies(self):
        """Different keys should give different outputs."""
        layer = BayesianLinear(4, 3, key=random.key(0))
        x = jnp.ones((2, 4))
        y1 = layer(x, key=random.key(1), sample=True)
        y2 = layer(x, key=random.key(2), sample=True)
        assert not jnp.allclose(y1, y2)

    def test_no_bias(self):
        layer = BayesianLinear(4, 3, use_bias=False, key=random.key(0))
        assert layer.b is None
        x = jnp.ones((2, 4))
        y = layer(x, key=random.key(1), sample=True)
        assert y.shape == (2, 3)

    def test_bayesian_bias_default(self):
        """bayesian_bias=True is the default."""
        layer = BayesianLinear(4, 3, key=random.key(0))
        assert isinstance(layer.b, GaussianParameter)

    def test_non_bayesian(self):
        layer = BayesianLinear(4, 3, bayesian=False, bayesian_bias=False, key=random.key(0))
        assert isinstance(layer.W, DeterministicParameter)
        assert isinstance(layer.b, DeterministicParameter)

    def test_laplacian_param_type(self):
        layer = BayesianLinear(4, 3, param_type=LaplacianParameter, key=random.key(0))
        assert isinstance(layer.W, LaplacianParameter)

    def test_key_none_sample_true_raises(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        x = jnp.ones((2, 4))
        with pytest.raises(ValueError, match="key is required"):
            layer(x, key=None, sample=True)


# ---------------------------------------------------------------------------
# Module introspection tests
# ---------------------------------------------------------------------------


class TestModuleIntrospection:
    def test_get_parameters_single_layer(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        params = layer.get_parameters()
        assert "W" in params
        assert "b" in params
        assert isinstance(params["W"], GaussianParameter)
        assert isinstance(params["b"], GaussianParameter)  # bayesian_bias=True by default

    def test_get_parameters_deterministic_bias(self):
        layer = BayesianLinear(4, 3, bayesian_bias=False, key=random.key(0))
        params = layer.get_parameters()
        assert isinstance(params["b"], DeterministicParameter)

    def test_get_parameters_mlp(self):
        mlp = SimpleMLP([4, 8, 3], key=random.key(0))
        params = mlp.get_parameters()
        assert "layers[0].W" in params
        assert "layers[0].b" in params
        assert "layers[1].W" in params
        assert "layers[1].b" in params
        assert len(params) == 4

    def test_flatten_means_shape(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        means = layer.flatten_means()
        # W: 3*4 = 12, b: 3 => total 15
        assert means.shape == (15,)

    def test_flatten_stdvs_shape(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        stdvs = layer.flatten_stdvs()
        # W: 12 stdvs, b: 3 stdvs (both bayesian by default) => 15
        assert stdvs.shape == (15,)
        assert jnp.all(stdvs > 0)  # all bayesian => all positive

    def test_flatten_stdvs_deterministic_bias(self):
        layer = BayesianLinear(4, 3, bayesian_bias=False, key=random.key(0))
        stdvs = layer.flatten_stdvs()
        assert stdvs.shape == (15,)
        assert jnp.all(stdvs[:12] > 0)
        assert jnp.allclose(stdvs[12:], 0.0)

    def test_flatten_raw_stdvs_shape(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        raw = layer.flatten_raw_stdvs()
        assert raw.shape == (15,)

    def test_mlp_forward(self):
        mlp = SimpleMLP([4, 8, 3], key=random.key(0))
        x = jnp.ones((2, 4))
        y = mlp(x, key=random.key(1), sample=True)
        assert y.shape == (2, 3)

    def test_mlp_forward_deterministic(self):
        mlp = SimpleMLP([4, 8, 3], key=random.key(0))
        x = jnp.ones((2, 4))
        y = mlp(x, key=random.key(1), sample=False)
        assert y.shape == (2, 3)


# ---------------------------------------------------------------------------
# Freeze / unfreeze tests
# ---------------------------------------------------------------------------


class TestFreezeUnfreeze:
    def test_freeze_stdvs_grad_on_means_only(self):
        """After freezing stdvs, gradients should only flow through means."""
        layer = BayesianLinear(4, 3, key=random.key(0))
        dynamic, static = freeze_stdvs(layer)

        def loss_fn(dynamic):
            model = unfreeze_all(dynamic, static)
            x = jnp.ones((1, 4))
            return jnp.sum(model(x, key=random.key(42), sample=True))

        grads = jax.grad(loss_fn)(dynamic)

        # Check grads exist on means
        flat_grads = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in flat_grads if g is not None)
        assert has_nonzero

        # Verify static contains raw_stdv (frozen, so not in dynamic)
        dynamic_flat, _ = jax.tree.flatten_with_path(dynamic)
        for path, val in dynamic_flat:
            path_str = str(path)
            if "raw_stdv" in path_str:
                assert val is None, f"raw_stdv should be frozen but found in dynamic: {path_str}"

    def test_freeze_means_grad_on_stdvs_only(self):
        """After freezing means, gradients should only flow through stdvs."""
        layer = BayesianLinear(4, 3, bayesian_bias=True, key=random.key(0))
        dynamic, static = freeze_means(layer)

        def loss_fn(dynamic):
            model = unfreeze_all(dynamic, static)
            x = jnp.ones((1, 4))
            return jnp.sum(model(x, key=random.key(42), sample=True))

        grads = jax.grad(loss_fn)(dynamic)

        # Check that means are frozen in dynamic (should be None)
        dynamic_flat, _ = jax.tree.flatten_with_path(dynamic)
        for path, val in dynamic_flat:
            path_str = str(path)
            if path_str.endswith("mean')")  or ".mean" in path_str:
                if "mean" in path_str and "raw" not in path_str:
                    assert val is None, f"mean should be frozen: {path_str}"

    def test_unfreeze_all_roundtrip(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        x = jnp.ones((2, 4))
        y_orig = layer(x, key=None, sample=False)

        dynamic, static = freeze_stdvs(layer)
        restored = unfreeze_all(dynamic, static)
        y_restored = restored(x, key=None, sample=False)

        assert jnp.allclose(y_orig, y_restored)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestUtils:
    def test_sample_all_parameters(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        sampled = sample_all_parameters(layer, random.key(1))

        # Means should differ from original (they are samples now)
        assert not jnp.allclose(sampled.W.mean, layer.W.mean)

        # Parameter types are preserved (not converted to DeterministicParameter)
        assert isinstance(sampled.W, GaussianParameter)

        # Forward pass should work with sample=False (uses the sampled means)
        x = jnp.ones((2, 4))
        y = sampled(x, key=None, sample=False)
        assert y.shape == (2, 3)

    def test_sample_all_parameters_varies(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        s1 = sample_all_parameters(layer, random.key(1))
        s2 = sample_all_parameters(layer, random.key(2))
        # Different keys should give different sampled models
        assert not jnp.allclose(s1.W.mean, s2.W.mean)

    def test_gaussian_entropy_scalar(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        h = gaussian_entropy(layer)
        assert h.shape == ()
        assert jnp.isfinite(h)

    def test_gaussian_entropy_deterministic_zero(self):
        """A fully deterministic model should have zero entropy."""
        layer = BayesianLinear(4, 3, bayesian=False, bayesian_bias=False, key=random.key(0))
        h = gaussian_entropy(layer)
        assert jnp.allclose(h, 0.0)

    def test_gaussian_entropy_value(self):
        """Check gaussian_entropy matches manual computation."""
        p = GaussianParameter(mean=jnp.zeros(3), raw_stdv=jnp.array([0.1, 0.5, 1.0]))

        class SingleParam(Module):
            w: GaussianParameter

        m = SingleParam(w=p)
        h = gaussian_entropy(m)
        expected = jnp.sum(jnp.log(p.stdv))
        assert jnp.allclose(h, expected)

    def test_laplacian_entropy_scalar(self):
        layer = BayesianLinear(4, 3, param_type=LaplacianParameter, key=random.key(0))
        h = laplacian_entropy(layer)
        assert h.shape == ()
        assert jnp.isfinite(h)

    def test_laplacian_entropy_deterministic_zero(self):
        """A fully deterministic model should have zero Laplacian entropy."""
        layer = BayesianLinear(4, 3, bayesian=False, bayesian_bias=False, key=random.key(0))
        h = laplacian_entropy(layer)
        assert jnp.allclose(h, 0.0)

    def test_laplacian_entropy_value(self):
        """Check laplacian_entropy matches manual computation."""
        p = LaplacianParameter(mean=jnp.zeros(4), raw_scale=jnp.array([0.1, 0.3, 0.7, 1.0]))

        class SingleParam(Module):
            w: LaplacianParameter

        m = SingleParam(w=p)
        h = laplacian_entropy(m)
        expected = jnp.sum(jnp.log(p.scale))
        assert jnp.allclose(h, expected)

    def test_laplacian_entropy_ignores_gaussian(self):
        """laplacian_entropy should only sum over Laplacian parameters."""
        model = MixedModel(key=random.key(0))
        h = laplacian_entropy(model)
        # Only laplace_layer contributes; gauss_layer and det_layer do not
        params = model.get_parameters()
        expected = jnp.array(0.0)
        for name, p in params.items():
            if isinstance(p, LaplacianParameter):
                expected = expected + jnp.sum(jnp.log(p.scale))
        assert jnp.allclose(h, expected)

    def test_gaussian_entropy_ignores_laplacian(self):
        """gaussian_entropy should only sum over Gaussian parameters."""
        model = MixedModel(key=random.key(0))
        h = gaussian_entropy(model)
        params = model.get_parameters()
        expected = jnp.array(0.0)
        for name, p in params.items():
            if isinstance(p, GaussianParameter):
                expected = expected + jnp.sum(jnp.log(p.stdv))
        assert jnp.allclose(h, expected)

    def test_get_parameter_count(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        counts = get_parameter_count(layer)
        assert counts["total"] == 15  # 12 + 3
        assert counts["bayesian"] == 15  # both W and b are bayesian by default
        assert counts["deterministic"] == 0

    def test_get_parameter_groups(self):
        model = MixedModel(key=random.key(0))
        groups = get_parameter_groups(model)
        assert len(groups["gaussian"]) > 0
        assert len(groups["laplacian"]) > 0
        assert len(groups["deterministic"]) > 0

    def test_flatten_means_module_level(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        m1 = flatten_means(layer)
        m2 = layer.flatten_means()
        assert jnp.allclose(m1, m2)

    def test_flatten_stdvs_module_level(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        s1 = flatten_stdvs(layer)
        s2 = layer.flatten_stdvs()
        assert jnp.allclose(s1, s2)


# ---------------------------------------------------------------------------
# Mixed parameter model tests
# ---------------------------------------------------------------------------


class TestMixedModel:
    def test_forward_sample(self):
        model = MixedModel(key=random.key(0))
        x = jnp.ones((2, 4))
        y = model(x, key=random.key(1), sample=True)
        assert y.shape == (2, 1)

    def test_forward_deterministic(self):
        model = MixedModel(key=random.key(0))
        x = jnp.ones((2, 4))
        y = model(x, key=None, sample=False)
        assert y.shape == (2, 1)

    def test_parameter_types_correct(self):
        model = MixedModel(key=random.key(0))
        params = model.get_parameters()
        assert isinstance(params["gauss_layer.W"], GaussianParameter)
        assert isinstance(params["laplace_layer.W"], LaplacianParameter)
        assert isinstance(params["det_layer.W"], DeterministicParameter)

    def test_sample_all_on_mixed(self):
        model = MixedModel(key=random.key(0))
        sampled = sample_all_parameters(model, random.key(1))

        # Parameter types are preserved
        params = sampled.get_parameters()
        assert isinstance(params["gauss_layer.W"], GaussianParameter)
        assert isinstance(params["laplace_layer.W"], LaplacianParameter)
        assert isinstance(params["det_layer.W"], DeterministicParameter)

        # Bayesian means should differ from original
        assert not jnp.allclose(
            sampled.gauss_layer.W.mean, model.gauss_layer.W.mean
        )

        x = jnp.ones((2, 4))
        y = sampled(x, key=None, sample=False)
        assert y.shape == (2, 1)


# ---------------------------------------------------------------------------
# JIT compatibility tests
# ---------------------------------------------------------------------------


class TestJIT:
    def test_jit_forward(self):
        layer = BayesianLinear(4, 3, key=random.key(0))

        @jax.jit
        def forward(model, x, key):
            return model(x, key=key, sample=True)

        x = jnp.ones((2, 4))
        y = forward(layer, x, random.key(1))
        assert y.shape == (2, 3)

    def test_jit_grad(self):
        layer = BayesianLinear(4, 3, key=random.key(0))
        dynamic, static = freeze_stdvs(layer)

        @jax.jit
        @jax.grad
        def grad_fn(dynamic):
            model = unfreeze_all(dynamic, static)
            x = jnp.ones((1, 4))
            return jnp.sum(model(x, key=random.key(42), sample=True))

        grads = grad_fn(dynamic)
        flat = jax.tree.leaves(grads)
        assert any(jnp.any(g != 0) for g in flat if g is not None)

    def test_vmap_sample(self):
        """vmap over multiple keys should produce different samples."""
        layer = BayesianLinear(4, 3, key=random.key(0))
        x = jnp.ones((4,))

        @jax.vmap
        def multi_sample(key):
            return layer(x, key=key, sample=True)

        keys = random.split(random.key(0), 8)
        ys = multi_sample(keys)
        assert ys.shape == (8, 3)
        # Not all the same
        assert not jnp.allclose(ys[0], ys[1])


# ---------------------------------------------------------------------------
# make_parameter tests
# ---------------------------------------------------------------------------


class TestMakeParameter:
    def test_gaussian_default(self):
        p = make_parameter(jnp.zeros((3, 4)))
        assert isinstance(p, GaussianParameter)
        assert p.shape == (3, 4)

    def test_scalar(self):
        p = make_parameter(jnp.array(1.0))
        assert isinstance(p, GaussianParameter)
        assert p.shape == ()

    def test_deterministic(self):
        p = make_parameter(jnp.ones(5), bayesian=False)
        assert isinstance(p, DeterministicParameter)
        assert jnp.allclose(p.mean, 1.0)

    def test_laplacian(self):
        p = make_parameter(jnp.eye(3), param_type=LaplacianParameter)
        assert isinstance(p, LaplacianParameter)
        assert p.shape == (3, 3)

    def test_init_raw_stdv(self):
        p = make_parameter(jnp.zeros(2), init_raw_stdv=0.5)
        assert jnp.allclose(p.raw_stdv, 0.5)

    def test_in_custom_model(self):
        """make_parameter works inside a user-defined Module."""

        class MyModel(Module):
            A: AbstractParameter
            scale: AbstractParameter

            def __init__(self, *, key):
                self.A = make_parameter(jnp.zeros((3, 4)))
                self.scale = make_parameter(jnp.array(1.0), bayesian=False)

            def __call__(self, x, *, key=None, sample=False):
                if sample:
                    A = self.A.sample(key)
                else:
                    A = self.A.mean
                return x @ A.T * self.scale.mean

        m = MyModel(key=random.key(0))
        params = m.get_parameters()
        assert "A" in params
        assert "scale" in params
        assert isinstance(params["A"], GaussianParameter)
        assert isinstance(params["scale"], DeterministicParameter)

        x = jnp.ones((2, 4))
        y = m(x, key=random.key(1), sample=True)
        assert y.shape == (2, 3)


# ---------------------------------------------------------------------------
# sample_all_parameters immutability test
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_sample_all_does_not_modify_original(self):
        """Equinox modules are frozen dataclasses, JAX arrays are immutable.
        sample_all_parameters must never modify the original model."""
        model = BayesianLinear(4, 3, key=random.key(0))
        original_W_mean = model.W.mean.copy()
        original_b_mean = model.b.mean.copy()

        for i in range(50):
            sampled = sample_all_parameters(model, random.key(i))
            assert not jnp.allclose(sampled.W.mean, model.W.mean)

        assert jnp.array_equal(model.W.mean, original_W_mean)
        assert jnp.array_equal(model.b.mean, original_b_mean)
