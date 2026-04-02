"""Tests for pyrsdr.utils — canonical_angles, procrustes_align."""
import numpy as np
import pytest

from pyrsdr.utils import canonical_angles, canonical_angles_deg, procrustes_align


# ── canonical_angles ────────────────────────────────────────────────────────

class TestCanonicalAngles:
    def test_identical_subspace(self):
        """Same subspace → zero angles."""
        A = np.eye(5, 2)
        angles = canonical_angles(A, A)
        np.testing.assert_allclose(angles, 0.0, atol=1e-12)

    def test_orthogonal_subspace(self):
        """Orthogonal subspaces → pi/2 angles."""
        A = np.eye(4, 2)
        B = np.eye(4, 2)[:, ::-1].copy()
        B[:2, :] = 0
        B[2, 0] = 1
        B[3, 1] = 1
        angles = canonical_angles(A, B)
        np.testing.assert_allclose(angles, np.pi / 2, atol=1e-12)

    def test_rotated_subspace(self):
        """Small rotation → small canonical angle."""
        A = np.eye(5, 1)
        theta = 0.1
        B = A.copy()
        B[0, 0] = np.cos(theta)
        B[1, 0] = np.sin(theta)
        angles = canonical_angles(A, B)
        np.testing.assert_allclose(angles[0], theta, atol=1e-6)

    def test_degrees(self):
        A = np.eye(3, 1)
        B = np.zeros((3, 1))
        B[1, 0] = 1.0
        deg = canonical_angles_deg(A, B)
        np.testing.assert_allclose(deg[0], 90.0, atol=1e-10)

    def test_validation_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            canonical_angles(np.ones(5), np.eye(5, 2))

    def test_validation_row_mismatch(self):
        with pytest.raises(ValueError, match="same number of rows"):
            canonical_angles(np.eye(4, 2), np.eye(5, 2))


# ── procrustes_align ────────────────────────────────────────────────────────

class TestProcrustesAlign:
    def test_identity(self):
        """Aligning identical matrices → same result."""
        Z = np.random.default_rng(0).standard_normal((20, 3))
        Z = Z - Z.mean(axis=0)
        aligned = procrustes_align(Z, Z)
        np.testing.assert_allclose(aligned, Z, atol=1e-12)

    def test_rotation(self):
        """Aligning a rotated version recovers original."""
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((30, 2))
        Z = Z - Z.mean(axis=0)
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
        Z_rot = Z @ R
        aligned = procrustes_align(Z, Z_rot)
        np.testing.assert_allclose(aligned, Z, atol=1e-10)

    def test_reflection(self):
        """Procrustes handles reflections."""
        rng = np.random.default_rng(2)
        Z = rng.standard_normal((20, 2))
        Z = Z - Z.mean(axis=0)
        Z_flip = Z.copy()
        Z_flip[:, 0] *= -1
        aligned = procrustes_align(Z, Z_flip)
        np.testing.assert_allclose(aligned, Z, atol=1e-10)

    def test_validation_not_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            procrustes_align(np.ones(5), np.ones((5, 1)))

    def test_validation_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            procrustes_align(np.ones((5, 2)), np.ones((5, 3)))
