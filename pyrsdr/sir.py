"""
SIR — Sliced Inverse Regression via JAX/NumPy.

Implements Li (1991) SIR with slicing logic matching Julia's
DimensionReductionRegression.jl and R's dr::sir.

Whitening convention (matching Julia/R dr::sir):
  Sigma = Xc'Xc / n,  T = Sigma^{-1/2},  Z = Xc @ T
  dirs  = T @ eigenvectors  (then unit-normalised)

Reference:
  Li, K-C. (1991) "Sliced Inverse Regression for Dimension Reduction"
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2

jax.config.update("jax_enable_x64", True)


class SIR:
    """Sliced Inverse Regression (SIR).

    Args:
        nslice: Number of slices.  If None, defaults to max(8, p+3).

    Raises:
        ValueError: If nslice is provided and is not positive.
    """

    def __init__(self, nslice: int | None = None) -> None:
        if nslice is not None and nslice < 1:
            raise ValueError(f"nslice must be > 0, got {nslice}")
        self.nslice = nslice
        self.dirs = None           # (p, ndir) estimated directions in X-space
        self.eigs = None           # eigenvalues (descending)
        self.eigv = None           # eigenvectors in whitened space
        self.slice_means = None    # (p, h) slice means of whitened data
        self.slice_props = None    # (h,) slice proportions
        self.trans = None          # Sigma^{-1/2} whitening transform
        self.center_mu = None      # column means of X
        self.n_actual_slices = None

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_logic(y, nslice):
        """Slicing logic matching Julia's slicer/slice1/slice2.

        Returns (slice_ids, n_actual_slices, boundary_array).
        """
        n = len(y)
        u = np.unique(y)

        if len(u) <= nslice:
            # One slice per distinct y value
            bds = []
            for v in u:
                bds.append(int(np.searchsorted(y, v)))
            bds.append(n)
            bds = np.unique(bds)
            if bds[0] != 0:
                bds = np.insert(bds, 0, 0)
            if bds[-1] != n:
                bds = np.append(bds, n)
        else:
            # Balanced slicing
            bds1 = []
            for v in u:
                bds1.append(int(np.searchsorted(y, v)))
            bds1.append(n)
            cty = np.cumsum(np.diff(bds1))

            m = np.floor(n / nslice)

            slice_boundary_indices = []
            jj = 0
            while jj < n - 2:
                jj += m
                s = int(np.searchsorted(cty, jj, side='left'))
                if s >= len(cty):
                    s = len(cty) - 1
                jj = cty[s]
                slice_boundary_indices.append(s)

            final_bds = [0]
            for s in slice_boundary_indices:
                final_bds.append(int(bds1[s + 1]))
            if final_bds[-1] != n:
                final_bds.append(n)
            bds = np.array(final_bds, dtype=int)

        slice_ids = np.zeros(n, dtype=int)
        for i in range(len(bds) - 1):
            slice_ids[bds[i]:bds[i + 1]] = i

        return slice_ids, len(bds) - 1, bds

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_core(X_np, slice_ids, n_slices, bds):
        """Core SIR: whiten -> slice means -> weighted covariance -> eigen.

        Returns (eig_vals, eig_vecs, T, mu, slice_means, fw).
        """
        n, p = X_np.shape

        # 1. Center
        mu = np.mean(X_np, axis=0)
        Xc = X_np - mu

        # 2. Whiten: Sigma = Xc'Xc/n  (n denominator, matching Julia/R dr::sir)
        Sigma = Xc.T @ Xc / n
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-10)
        inv_sqrt_eigs = 1.0 / np.sqrt(eigvals)
        T = eigvecs @ np.diag(inv_sqrt_eigs) @ eigvecs.T
        Z = Xc @ T

        # 3. Slice means
        sm = np.zeros((p, n_slices))
        ns = np.zeros(n_slices)
        for i in range(n_slices):
            mask = slice_ids == i
            ns[i] = np.sum(mask)
            if ns[i] > 0:
                sm[:, i] = np.mean(Z[mask], axis=0)

        fw = ns / n

        # 4. Weighted covariance M = sum(fw_h * sm_h sm_h')
        M = sm @ np.diag(fw) @ sm.T

        # 5. Eigendecompose (descending)
        eig_vals, eig_vecs = np.linalg.eigh(M)
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        return eig_vals, eig_vecs, T, mu, sm, fw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike, ndir: int = 2) -> "SIR":
        """Fit the SIR model.

        Args:
            X: (n, p) predictor matrix.
            y: (n,) response vector.
            ndir: Number of directions to estimate.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If X is not 2-D, X/y row counts differ,
                ndir < 1, or ndir > p.
        """
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64).ravel()

        if X_np.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X_np.ndim}-D")
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, "
                f"got {X_np.shape[0]} and {y_np.shape[0]}"
            )

        sort_idx = np.argsort(y_np)
        X_sorted = X_np[sort_idx]
        y_sorted = y_np[sort_idx]

        n, p = X_sorted.shape

        if ndir < 1:
            raise ValueError(f"ndir must be >= 1, got {ndir}")
        if ndir > p:
            raise ValueError(f"ndir must be <= p ({p}), got {ndir}")

        if self.nslice is None:
            self.nslice = max(8, p + 3)

        slice_ids, actual_n_slices, bds = self._slice_logic(y_sorted, self.nslice)

        eig_vals, eig_vecs, T, mu, sm, fw = self._fit_core(
            X_sorted, slice_ids, actual_n_slices, bds
        )

        # Direction recovery: beta = T @ eigvec = Sigma^{-1/2} @ eigvec
        raw_dirs = T @ eig_vecs[:, :ndir]
        norms = np.linalg.norm(raw_dirs, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        self.dirs = raw_dirs / norms

        self.eigs = eig_vals
        self.eigv = eig_vecs
        self.trans = T
        self.center_mu = mu
        self.slice_means = sm
        self.slice_props = fw
        self.n_actual_slices = actual_n_slices

        return self

    def dimension_test(self, n_samples: int) -> dict:
        """Chi-square sequential test for the structural dimension (Li 1991).

        Tests H0: dim(EDR) <= k  for k = 0, 1, ...

        Args:
            n_samples: Sample size (needed for the test statistic).

        Returns:
            dict with keys ``stats``, ``dofs``, ``p_values``.

        Raises:
            RuntimeError: If model has not been fitted yet.
            ValueError: If n_samples < 1.
        """
        if self.eigs is None:
            raise RuntimeError("Model not fitted")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        p = len(self.eigs)
        h = self.n_actual_slices

        eigs_asc = self.eigs[::-1]
        stat_asc = n_samples * np.cumsum(eigs_asc)
        stats = stat_asc[::-1]

        k_vals = np.arange(p)
        dofs = (p - k_vals) * (h - k_vals - 1)

        p_values = np.full(p, np.nan)
        for i in range(p):
            if dofs[i] > 0:
                p_values[i] = 1.0 - chi2.cdf(stats[i], df=dofs[i])

        return {"stats": stats, "dofs": dofs, "p_values": p_values}


# ============================================================================
# Quick demo
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    n, p = 500, 5
    X = np.random.randn(n, p)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n)

    sir = SIR(nslice=10)
    sir.fit(X, y, ndir=2)

    print("Estimated EDR directions (first 2):")
    print(sir.dirs)

    print("\nEigenvalues:")
    print(sir.eigs)

    print("\nDimension test:")
    dt = sir.dimension_test(n)
    for k in range(min(p, 5)):
        print(f"  Dim {k}: Stat={dt['stats'][k]:.2f}, "
              f"DoF={dt['dofs'][k]:.0f}, "
              f"P-val={dt['p_values'][k]:.4f}")
