"""
SIR test suite — true direction recovery, R reference comparison,
whitening check, input validation.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pyrsdr import SIR, canonical_angles
from tests.conftest import TESTDATA_DIR


# ============================================================================
# Core tests (no external data needed)
# ============================================================================

class TestSIRLinear:
    """SIR recovers true direction for y = X1 + noise."""

    def test_direction_recovery(self):
        rng = np.random.default_rng(42)
        n, p = 500, 5
        X = rng.standard_normal((n, p))
        y = X[:, 0] + 0.1 * rng.standard_normal(n)

        sir = SIR(nslice=10)
        sir.fit(X, y, ndir=1)

        true_dir = np.zeros((p, 1))
        true_dir[0, 0] = 1.0
        angles = canonical_angles(true_dir, np.array(sir.dirs[:, :1]))
        assert angles[0] < 0.15, f"angle={np.degrees(angles[0]):.2f} deg"


class TestSIRWhitening:
    """Internal consistency: whitened data has identity covariance."""

    def test_cov_identity(self):
        rng = np.random.default_rng(42)
        n, p = 500, 5
        X = rng.standard_normal((n, p))
        y = X[:, 0] + 0.1 * rng.standard_normal(n)

        sir = SIR(nslice=10)
        sir.fit(X, y, ndir=2)

        sort_idx = np.argsort(y)
        X_sorted = X[sort_idx]
        Xc = X_sorted - X_sorted.mean(axis=0)
        Z = Xc @ sir.trans
        cov_Z = Z.T @ Z / len(Z)
        max_off = np.max(np.abs(cov_Z - np.eye(p)))
        assert max_off < 0.01, f"max |Cov(Z) - I| = {max_off:.2e}"

    def test_dirs_equal_T_eigvecs(self):
        rng = np.random.default_rng(42)
        n, p = 500, 5
        X = rng.standard_normal((n, p))
        y = X[:, 0] + 0.1 * rng.standard_normal(n)

        sir = SIR(nslice=10)
        sir.fit(X, y, ndir=2)

        raw = sir.trans @ sir.eigv[:, :2]
        norms = np.linalg.norm(raw, axis=0, keepdims=True)
        reconstructed = raw / norms
        diff = np.max(np.abs(np.abs(sir.dirs) - np.abs(reconstructed)))
        assert diff < 1e-10, f"diff={diff:.2e}"


class TestSIRDimensionTest:
    """dimension_test produces sensible p-values."""

    def test_first_significant(self):
        rng = np.random.default_rng(42)
        n, p = 500, 5
        X = rng.standard_normal((n, p))
        y = X[:, 0] + 0.1 * rng.standard_normal(n)

        sir = SIR(nslice=10)
        sir.fit(X, y, ndir=p)
        dt = sir.dimension_test(n)

        assert dt["p_values"][0] < 0.05


# ============================================================================
# R reference tests (require test_data/)
# ============================================================================

@pytest.mark.reference
class TestSIRReferenceLinear:
    """Compare SIR eigenvalues with R dr::sir on linear test data."""

    @pytest.fixture(autouse=True)
    def _load(self):
        try:
            self.X = np.loadtxt(f"{TESTDATA_DIR}/sir_test1_X.csv",
                                delimiter=",", skiprows=1)
            self.Y = np.loadtxt(f"{TESTDATA_DIR}/sir_test1_Y.csv",
                                delimiter=",", skiprows=1)
            self.r_eigs = np.loadtxt(f"{TESTDATA_DIR}/sir_r_test1_eigs.csv",
                                     delimiter=",", skiprows=1)
            self.r_dirs = np.loadtxt(f"{TESTDATA_DIR}/sir_r_test1_dirs.csv",
                                     delimiter=",", skiprows=1)
        except FileNotFoundError:
            pytest.skip("SIR linear reference data not found")

    def test_eigenvalues(self):
        p = self.X.shape[1]
        sir = SIR(nslice=10)
        sir.fit(self.X, self.Y, ndir=p)
        diff = np.max(np.abs(np.array(sir.eigs[:p]) - self.r_eigs[:p]))
        assert diff < 0.01, f"max eigenvalue diff={diff:.2e}"

    def test_subspace_angles(self):
        sir = SIR(nslice=10)
        sir.fit(self.X, self.Y, ndir=2)
        angles = canonical_angles(self.r_dirs[:, :2],
                                  np.array(sir.dirs[:, :2]))
        assert np.all(angles < 0.15), (
            f"angles={np.degrees(angles).round(2)} deg"
        )


@pytest.mark.reference
class TestSIRReferenceQuadratic:
    """Compare SIR eigenvalues with R on quadratic test data."""

    @pytest.fixture(autouse=True)
    def _load(self):
        try:
            self.X = np.loadtxt(f"{TESTDATA_DIR}/sir_test2_X.csv",
                                delimiter=",", skiprows=1)
            self.Y = np.loadtxt(f"{TESTDATA_DIR}/sir_test2_Y.csv",
                                delimiter=",", skiprows=1)
            self.r_eigs = np.loadtxt(f"{TESTDATA_DIR}/sir_r_test2_eigs.csv",
                                     delimiter=",", skiprows=1)
        except FileNotFoundError:
            pytest.skip("SIR quadratic reference data not found")

    def test_eigenvalues(self):
        p = self.X.shape[1]
        sir = SIR(nslice=10)
        sir.fit(self.X, self.Y, ndir=p)
        diff = np.max(np.abs(np.array(sir.eigs[:p]) - self.r_eigs[:p]))
        assert diff < 0.01, f"max eigenvalue diff={diff:.2e}"


@pytest.mark.reference
class TestSIRReferenceIonosphere:
    """Compare SIR eigenvalues with R on Ionosphere data."""

    @pytest.fixture(autouse=True)
    def _load(self):
        try:
            self.X = np.loadtxt(f"{TESTDATA_DIR}/ionosphere_X.csv",
                                delimiter=",", skiprows=1)
            self.Y = np.loadtxt(f"{TESTDATA_DIR}/ionosphere_Y.csv",
                                delimiter=",", skiprows=1)
            self.r_eigs = np.loadtxt(f"{TESTDATA_DIR}/sir_r_test3_eigs.csv",
                                     delimiter=",", skiprows=1)
        except FileNotFoundError:
            pytest.skip("SIR ionosphere reference data not found")

    def test_eigenvalues(self):
        p = self.X.shape[1]
        sir = SIR(nslice=10)
        sir.fit(self.X, self.Y, ndir=min(5, p))
        diff = np.max(np.abs(np.array(sir.eigs[:p]) - self.r_eigs[:p]))
        assert diff < 0.05, f"max eigenvalue diff={diff:.2e}"


# ============================================================================
# Input validation tests
# ============================================================================

class TestSIRValidation:
    def test_nslice_zero(self):
        with pytest.raises(ValueError, match="nslice"):
            SIR(nslice=0)

    def test_x_not_2d(self):
        sir = SIR(nslice=5)
        with pytest.raises(ValueError, match="2-D"):
            sir.fit(np.ones(10), np.ones(10))

    def test_row_mismatch(self):
        sir = SIR(nslice=5)
        with pytest.raises(ValueError, match="same number of rows"):
            sir.fit(np.ones((10, 3)), np.ones(5))

    def test_ndir_zero(self):
        sir = SIR(nslice=5)
        with pytest.raises(ValueError, match="ndir must be >= 1"):
            sir.fit(np.random.randn(20, 4), np.random.randn(20), ndir=0)

    def test_ndir_exceeds_p(self):
        sir = SIR(nslice=5)
        with pytest.raises(ValueError, match="ndir must be <= p"):
            sir.fit(np.random.randn(20, 4), np.random.randn(20), ndir=5)
