"""
Test suite for RSDR cross-validation and bootstrap alpha selection.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pyrsdr import RSDR, rsdr_cv, rsdr_bootstrap
from pyrsdr.cv import _validation_cost


# ============================================================================
# Data generators
# ============================================================================

def make_linear(n=200, p=6, seed=42):
    """y = X1 - X2 + noise.  True subspace is 1-D."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, 0] - X[:, 1] + 0.3 * rng.standard_normal(n)
    return X, y


def make_quadratic(n=300, p=6, seed=42):
    """y = X1*(1 + X2) + noise.  True subspace is 2-D."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, 0] * (1 + X[:, 1]) + 0.3 * rng.standard_normal(n)
    return X, y


# ============================================================================
# Core tests
# ============================================================================

class TestValidationCost:
    """_validation_cost agrees with training objective on same data."""

    def test_consistency(self):
        X, y = make_linear(n=200, p=6)
        model = RSDR(X, y, alpha=0.5)
        model.fit(d=1, maxiter=500, seed=42)

        cost_val = _validation_cost(model, X, y)
        diff = abs(cost_val - model.f_value)
        assert diff < 1e-6, f"diff={diff:.2e}"


class TestCVStructure:
    """rsdr_cv returns correctly structured results."""

    def test_keys_and_lengths(self):
        X, y = make_linear(n=150, p=4)
        alpha_grid = [0.5, 1.0]
        result = rsdr_cv(X, y, d=1, alpha_grid=alpha_grid, K=3,
                         maxiter=300, seed=42)

        assert "alpha_star" in result
        assert "summary" in result
        assert result["alpha_star"] in alpha_grid
        assert len(result["summary"]) == len(alpha_grid)
        assert all(len(r["fold_costs"]) == 3 for r in result["summary"])
        assert all(np.isfinite(r["mean_cost"]) for r in result["summary"])


class TestCVAlphaConsistency:
    """Fixed eval_alpha produces comparable-scale costs."""

    def test_same_scale(self):
        X, y = make_linear(n=200, p=6)
        result = rsdr_cv(X, y, d=1, alpha_grid=[0.25, 1.5], K=3,
                         maxiter=300, seed=42, eval_alpha=0.5)

        costs = [r["mean_cost"] for r in result["summary"]]
        assert abs(costs[0] - costs[1]) > 1e-6, "costs should differ"
        assert abs(costs[1] / costs[0]) < 100, "costs not same order"
        assert result.get("eval_alpha") == 0.5


class TestCVReproducibility:
    """Same seed produces identical results."""

    def test_reproducible(self):
        X, y = make_linear(n=150, p=4)
        kwargs = dict(d=1, alpha_grid=[0.5, 1.0], K=3, maxiter=200, seed=99)

        r1 = rsdr_cv(X, y, **kwargs)
        r2 = rsdr_cv(X, y, **kwargs)

        assert r1["alpha_star"] == r2["alpha_star"]
        for i in range(len(r1["summary"])):
            np.testing.assert_allclose(
                r1["summary"][i]["fold_costs"],
                r2["summary"][i]["fold_costs"]
            )


class TestBootstrapStructure:
    """rsdr_bootstrap returns correctly structured results."""

    def test_keys_and_lengths(self):
        X, y = make_linear(n=100, p=4)
        result = rsdr_bootstrap(X, y, d=1, alpha_grid=[0.5, 1.0],
                                B=5, maxiter=200, seed=42)

        assert "alpha_star" in result
        assert "summary" in result
        assert result["alpha_star"] in [0.5, 1.0]
        assert all("rep_costs" in r for r in result["summary"])
        assert all(np.isfinite(r["mean_cost"]) for r in result["summary"])


class TestMultivariateY:
    """CV works with multivariate Y (n x q)."""

    def test_finite_results(self):
        rng = np.random.default_rng(42)
        n, p, q = 150, 4, 2
        X = rng.standard_normal((n, p))
        Y = np.column_stack([
            X[:, 0] + 0.3 * rng.standard_normal(n),
            X[:, 1] ** 2 + 0.3 * rng.standard_normal(n),
        ])

        result = rsdr_cv(X, Y, d=2, alpha_grid=[0.5, 1.0], K=3,
                         maxiter=200, seed=42)

        assert all(np.isfinite(r["mean_cost"]) for r in result["summary"])
        assert result["alpha_star"] in [0.5, 1.0]


# ============================================================================
# NaN handling tests
# ============================================================================

class TestNaNFilteringRSDR:
    """RSDR auto-filters NaN rows in Y."""

    def test_nan_drop(self):
        rng = np.random.default_rng(42)
        n, p = 200, 6
        X = rng.standard_normal((n, p))
        y = X[:, 0] + 0.3 * rng.standard_normal(n)

        nan_idx = rng.choice(n, size=40, replace=False)
        y_nan = y.copy()
        y_nan[nan_idx] = np.nan

        model = RSDR(X, y_nan, alpha=0.5)
        model.fit(d=1, maxiter=300, seed=42)

        assert model.n_dropped == 40
        assert model.keep_mask.shape == (n,)
        assert model.keep_mask.sum() == n - 40
        assert model.n == 160
        assert model.projected_data.shape == (160, 1)

    def test_matches_manual_filter(self):
        rng = np.random.default_rng(42)
        n, p = 200, 6
        X = rng.standard_normal((n, p))
        y = X[:, 0] + 0.3 * rng.standard_normal(n)

        nan_idx = rng.choice(n, size=40, replace=False)
        y_nan = y.copy()
        y_nan[nan_idx] = np.nan

        model = RSDR(X, y_nan, alpha=0.5)
        model.fit(d=1, maxiter=300, seed=42)

        X_clean = X[~np.isin(np.arange(n), nan_idx)]
        y_clean = y[~np.isin(np.arange(n), nan_idx)]
        model_ref = RSDR(X_clean, y_clean, alpha=0.5)
        model_ref.fit(d=1, maxiter=300, seed=42)

        assert abs(model.f_value - model_ref.f_value) < 1e-10

    def test_no_nan_no_drop(self):
        X = np.random.default_rng(0).standard_normal((50, 4))
        y = X[:, 0]
        model = RSDR(X, y)
        assert model.n_dropped == 0


class TestCVNaN:
    """rsdr_cv handles NaN in Y."""

    def test_nan_y(self):
        rng = np.random.default_rng(42)
        n, p = 200, 4
        X = rng.standard_normal((n, p))
        y = X[:, 0] + 0.3 * rng.standard_normal(n)

        y_nan = y.copy()
        y_nan[rng.choice(n, size=60, replace=False)] = np.nan

        result = rsdr_cv(X, y_nan, d=1, alpha_grid=[0.5, 1.0], K=3,
                         maxiter=200, seed=42)

        assert result["alpha_star"] in [0.5, 1.0]
        assert all(np.isfinite(r["mean_cost"]) for r in result["summary"])


class TestNaNMultivariate:
    """NaN in one column of multivariate Y drops the row."""

    def test_multivariate_nan(self):
        rng = np.random.default_rng(42)
        n, p = 100, 4
        X = rng.standard_normal((n, p))
        Y = np.column_stack([
            X[:, 0] + 0.1 * rng.standard_normal(n),
            X[:, 1] + 0.1 * rng.standard_normal(n),
        ])

        Y[0:10, 0] = np.nan
        Y[10:15, 1] = np.nan

        model = RSDR(X, Y, alpha=0.5)
        assert model.n_dropped == 15
        assert model.n == 85


# ============================================================================
# Input validation tests
# ============================================================================

class TestCVValidation:
    def test_empty_alpha_grid(self):
        X, y = make_linear(n=50, p=4)
        with pytest.raises(ValueError, match="alpha_grid must not be empty"):
            rsdr_cv(X, y, d=1, alpha_grid=[], K=3)

    def test_k_less_than_2(self):
        X, y = make_linear(n=50, p=4)
        with pytest.raises(ValueError, match="K must be >= 2"):
            rsdr_cv(X, y, d=1, alpha_grid=[0.5], K=1)

    def test_d_zero(self):
        X, y = make_linear(n=50, p=4)
        with pytest.raises(ValueError, match="d must be >= 1"):
            rsdr_cv(X, y, d=0, alpha_grid=[0.5])

    def test_x_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            rsdr_cv(np.ones(10), np.ones(10), d=1, alpha_grid=[0.5])

    def test_row_mismatch(self):
        with pytest.raises(ValueError, match="same number of rows"):
            rsdr_cv(np.ones((10, 3)), np.ones(5), d=1, alpha_grid=[0.5])

    def test_bootstrap_empty_grid(self):
        X, y = make_linear(n=50, p=4)
        with pytest.raises(ValueError, match="alpha_grid must not be empty"):
            rsdr_bootstrap(X, y, d=1, alpha_grid=[], B=5)

    def test_bootstrap_b_zero(self):
        X, y = make_linear(n=50, p=4)
        with pytest.raises(ValueError, match="B must be >= 1"):
            rsdr_bootstrap(X, y, d=1, alpha_grid=[0.5], B=0)
