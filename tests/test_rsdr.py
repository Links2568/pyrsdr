"""
RSDR test suite — gradient check, linear/quadratic recovery,
R reference comparison, input validation.
"""
import csv
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pyrsdr import RSDR, canonical_angles
from tests.conftest import TESTDATA_DIR


# ============================================================================
# Helpers
# ============================================================================

def gendat_quadratic(n, p, rng):
    """y = (X1 - X2) * (1 + X3 + X4) + noise."""
    X = rng.standard_normal((n, p))
    X = X - X.mean(axis=0)
    lp1 = X[:, 0] - X[:, 1]
    lp2 = X[:, 2] + X[:, 3]
    y = lp1 * (1 + lp2) + 0.5 * rng.standard_normal(n)
    ii = np.argsort(y)
    return X[ii], y[ii]


# ============================================================================
# Core tests
# ============================================================================

class TestGradient:
    """Autodiff gradient matches finite differences."""

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    def test_gradient(self, alpha):
        rng = np.random.default_rng(123)
        n, p, q, d = 200, 4, 2, 2

        X = rng.standard_normal((n, p))
        Y = rng.standard_normal((n, q))
        Y[:, 0] += X[:, 0] * (1 + X[:, 1]) ** 2
        Y[:, 1] += X[:, 1] * (1 + X[:, 0]) ** 2

        m = RSDR(X, Y, alpha=alpha)

        for _ in range(3):
            C_raw = rng.standard_normal((p, d))
            C, _ = np.linalg.qr(C_raw)
            C = C[:, :d]
            C_jax = jnp.array(C)

            _, egrad = RSDR._val_and_egrad(C_jax, m.Z, m.B, m.alpha, m.eta)

            eps = 1e-5
            grad_numerical = np.zeros_like(C)
            for i in range(p):
                for j in range(d):
                    Cp = C.copy(); Cp[i, j] += eps
                    Cm = C.copy(); Cm[i, j] -= eps
                    fp = float(RSDR.objective(jnp.array(Cp), m.Z, m.B, m.alpha, m.eta))
                    fm = float(RSDR.objective(jnp.array(Cm), m.Z, m.B, m.alpha, m.eta))
                    grad_numerical[i, j] = (fp - fm) / (2 * eps)

            egrad_np = np.array(egrad)
            rel_err = np.max(np.abs(egrad_np - grad_numerical)) / (
                np.max(np.abs(grad_numerical)) + 1e-15
            )
            assert rel_err < 1e-3, f"Gradient rel_err={rel_err:.2e}"


class TestFitLinear:
    """RSDR recovers 1-D linear subspace."""

    def test_angle(self, linear_data):
        X, y = linear_data
        p = X.shape[1]
        m = RSDR(X, y, alpha=0.5)
        m.fit(d=1, maxiter=2000, seed=42)

        G = np.zeros((p, 1))
        G[0, 0] = 1; G[1, 0] = -1
        G[:, 0] /= np.linalg.norm(G[:, 0])

        angles = canonical_angles(G, np.array(m.dirs))
        assert angles[0] < 0.15, f"angle={np.degrees(angles[0]):.2f} deg"


class TestFitQuadratic:
    """RSDR recovers 2-D quadratic subspace."""

    def test_angle(self):
        rng = np.random.default_rng(42)
        n, p = 500, 6
        X, y = gendat_quadratic(n, p, rng)

        m = RSDR(X, y, alpha=0.5)
        m.fit(d=2, maxiter=2000, seed=42)

        G = np.zeros((p, 2))
        G[0, 0] = 1; G[1, 0] = -1
        G[2, 1] = 1; G[3, 1] = 1
        G[:, 0] /= np.linalg.norm(G[:, 0])
        G[:, 1] /= np.linalg.norm(G[:, 1])

        angles = canonical_angles(G, np.array(m.dirs))
        assert np.all(angles < 0.3), (
            f"angles={np.degrees(angles).round(2)} deg"
        )


# ============================================================================
# R reference test
# ============================================================================

@pytest.mark.reference
class TestIonosphere:
    """Compare with R rSDR reference on Ionosphere data."""

    @pytest.fixture(autouse=True)
    def _load_data(self):
        testdir = TESTDATA_DIR
        try:
            self.X = np.loadtxt(f"{testdir}/ionosphere_X.csv", delimiter=",", skiprows=1)
            self.Y = np.loadtxt(f"{testdir}/ionosphere_Y.csv", delimiter=",", skiprows=1)
            self.beta_r = np.loadtxt(f"{testdir}/ionosphere_beta_r.csv", delimiter=",", skiprows=1)
            meta = {}
            with open(f"{testdir}/ionosphere_meta.csv") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    meta[row[0]] = row[1]
            self.d = int(float(meta["d"]))
            self.alpha_r = float(meta["alpha"])
            self.f_value_r = float(meta["f_value_cost"])
        except FileNotFoundError:
            pytest.skip("ionosphere test data not found")

    def test_objective_match(self):
        m = RSDR(self.X, self.Y, alpha=self.alpha_r)
        C_from_r = m.beta_to_C(self.beta_r)
        f_at_r = float(RSDR.objective(C_from_r, m.Z, m.B, m.alpha, m.eta))
        assert abs(f_at_r - self.f_value_r) < 0.005, (
            f"diff={abs(f_at_r - self.f_value_r):.2e}"
        )

    def test_fit_angles(self):
        m = RSDR(self.X, self.Y, alpha=self.alpha_r)
        m.fit(d=self.d, maxiter=2000, seed=42)
        angles = canonical_angles(self.beta_r, np.array(m.dirs))
        assert np.all(angles < 0.5), (
            f"angles={np.degrees(angles).round(2)} deg"
        )


# ============================================================================
# Input validation tests
# ============================================================================

class TestValidation:
    def test_x_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            RSDR(np.ones(10), np.ones(10))

    def test_row_mismatch(self):
        with pytest.raises(ValueError, match="same number of rows"):
            RSDR(np.ones((10, 3)), np.ones(5))

    def test_alpha_nonpositive(self):
        with pytest.raises(ValueError, match="alpha"):
            RSDR(np.ones((10, 3)), np.ones(10), alpha=0)

    def test_eta_negative(self):
        with pytest.raises(ValueError, match="eta"):
            RSDR(np.ones((10, 3)), np.ones(10), eta=-1)

    def test_d_zero(self):
        m = RSDR(np.random.randn(20, 4), np.random.randn(20))
        with pytest.raises(ValueError, match="d must be >= 1"):
            m.fit(d=0)

    def test_d_exceeds_p(self):
        m = RSDR(np.random.randn(20, 4), np.random.randn(20))
        with pytest.raises(ValueError, match="d must be <= p"):
            m.fit(d=5)

    def test_c_init_shape(self):
        m = RSDR(np.random.randn(20, 4), np.random.randn(20))
        with pytest.raises(ValueError, match="C_init shape"):
            m.fit(d=2, C_init=np.eye(4, 3))
