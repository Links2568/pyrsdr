"""
RSDR — Distance-covariance Sufficient Dimension Reduction via JAX.

Minimises f(C) = -mean(D_{ZC}^alpha * B) on the Stiefel manifold St(p, d)
using Riemannian Conjugate Gradient (Polak-Ribiere+).

Matches the R package rSDR (rSDR::rSDR) in:
  - Whitening with n-1 denominator covariance
  - Double-centred distance kernel B = H D_Y^alpha H
  - ManifoldOptim RCG optimiser with Armijo line search

Reference:
  Sheng & Yin (2016), "Sufficient Dimension Reduction via Distance Covariance"
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import numpy as np
from numpy.typing import ArrayLike

jax.config.update("jax_enable_x64", True)

# ============================================================================
# Utility functions
# ============================================================================

@jit
def get_pairwise_sq_dist(X):
    """Pairwise squared Euclidean distances: |X_i - X_j|^2."""
    G = X @ X.T
    diag_G = jnp.diag(G)
    D2 = diag_G[:, None] + diag_G[None, :] - 2 * G
    return jnp.maximum(D2, 0.0)


@jit
def double_center(A):
    """Double-center a matrix: H A H  where H = I - 11'/n."""
    mu = jnp.mean(A)
    mu_row = jnp.mean(A, axis=1, keepdims=True)
    mu_col = jnp.mean(A, axis=0, keepdims=True)
    return A - mu_row - mu_col + mu


@jit
def compute_B_matrix(Y, alpha):
    """Double-centred distance kernel of Y: B = H (D_Y^alpha) H."""
    D2 = get_pairwise_sq_dist(Y)
    D = jnp.sqrt(D2)
    A = D ** alpha
    return double_center(A)


@jit(static_argnums=(1,))
def project_stiefel(M, d):
    """Retract onto St(p,d) via polar decomposition: M = U S V' -> U V'."""
    U, _, Vt = jnp.linalg.svd(M, full_matrices=False)
    return U[:, :d] @ Vt[:d, :]


# ============================================================================
# Stiefel manifold operations (tangent-space geometry for RCG)
# ============================================================================

@jit
def stiefel_proj(C, Z):
    """Project Z onto the tangent space T_C St(p,d).
    proj(Z) = Z - C * sym(C'Z), where sym(A) = (A+A')/2.
    """
    sym = (C.T @ Z + Z.T @ C) / 2
    return Z - C @ sym


@jit
def stiefel_inner(A, B):
    """Frobenius inner product on the tangent space."""
    return jnp.sum(A * B)


# ============================================================================
# RSDR class
# ============================================================================

class RSDR:
    """Distance-covariance Sufficient Dimension Reduction.

    Finds the projection C in St(p,d) that maximises the alpha-distance
    covariance between the projected predictors Z@C and the response Y.

    Rows where Y contains NaN are **automatically dropped** before whitening
    and fitting.  The number of dropped rows is stored in ``n_dropped``, and
    the boolean keep-mask in ``keep_mask`` (True = row was used).
    To project the full original X (including dropped rows), use
    ``X_full @ model.dirs`` after fitting.

    Args:
        X: (n, p) predictor matrix.
        Y: (n, q) response matrix (or 1-D vector).  Rows with any NaN are
           removed together with the corresponding X rows.
        alpha: Distance covariance exponent (default 0.5, matching R).
        eta: Smoothing parameter for sqrt(D^2 + eta) to avoid gradient
             singularity at zero (default 1e-10).

    Raises:
        ValueError: If X is not 2-D, X/Y row counts differ, or
            alpha/eta are out of range.
    """

    def __init__(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        alpha: float = 0.5,
        eta: float = 1e-10,
    ) -> None:
        X_np = np.asarray(X, dtype=np.float64)
        Y_np = np.asarray(Y, dtype=np.float64)

        if X_np.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X_np.ndim}-D")
        if Y_np.ndim == 1:
            Y_np = Y_np[:, None]
        if X_np.shape[0] != Y_np.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of rows, "
                f"got {X_np.shape[0]} and {Y_np.shape[0]}"
            )
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if eta < 0:
            raise ValueError(f"eta must be >= 0, got {eta}")

        # ── Auto-filter NaN rows in Y ──
        nan_rows = np.any(np.isnan(Y_np), axis=1)
        n_total = X_np.shape[0]
        self.n_dropped = int(nan_rows.sum())
        if self.n_dropped > 0:
            keep = ~nan_rows
            X_np = X_np[keep]
            Y_np = Y_np[keep]
            self.keep_mask = keep
            print(f"RSDR: dropped {self.n_dropped} NaN-Y rows "
                  f"({X_np.shape[0]} of {n_total} retained)")
        else:
            self.keep_mask = np.ones(n_total, dtype=bool)

        self.X_orig = jnp.array(X_np, dtype=jnp.float64)
        self.Y = jnp.array(Y_np, dtype=jnp.float64)

        self.alpha = alpha
        self.eta = eta
        self.n, self.p = self.X_orig.shape

        # Whitening: Sigma = cov(X) with n-1 denominator (matching R)
        X_centered = self.X_orig - jnp.mean(self.X_orig, axis=0)
        Sigma = X_centered.T @ X_centered / (self.n - 1)  # n-1: matches R
        eigvals, eigvecs = jnp.linalg.eigh(Sigma)
        eigvals = jnp.maximum(eigvals, 1e-10)

        self.Nhalf_inv = eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals)) @ eigvecs.T
        self.Nhalf = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T
        self.Z = self.X_orig @ self.Nhalf_inv  # whitened data

        # Double-centred distance kernel of Y
        self.B = compute_B_matrix(self.Y, self.alpha)

        self.dirs = None          # beta in original X-space (p x d)
        self.C_value = None       # C on St(p,d) in whitened space
        self.f_value = None       # final objective value
        self.projected_data = None
        self.fitted = False

    def beta_to_C(self, beta: ArrayLike) -> jnp.ndarray:
        """Convert original-space directions beta to Stiefel matrix C.
        beta = Nhalf_inv @ C  =>  C = Nhalf @ beta, then re-orthogonalise.
        """
        C = self.Nhalf @ jnp.array(beta, dtype=jnp.float64)
        U, _, Vt = jnp.linalg.svd(C, full_matrices=False)
        return U @ Vt

    # ------------------------------------------------------------------
    # Objective & gradient (static so JAX can trace/JIT them)
    # ------------------------------------------------------------------

    @staticmethod
    @jit
    def objective(C, Z, B, alpha, eta):
        """f(C) = -mean(D_{ZC}^alpha * B)  with smoothed distance."""
        ZC = Z @ C
        D2 = get_pairwise_sq_dist(ZC)
        D = jnp.sqrt(D2 + eta)
        return -jnp.mean((D ** alpha) * B)

    @staticmethod
    @jit
    def _val_and_egrad(C, Z, B, alpha, eta):
        """Objective value + Euclidean gradient via autodiff."""
        return value_and_grad(RSDR.objective)(C, Z, B, alpha, eta)

    # ------------------------------------------------------------------
    # Riemannian Conjugate Gradient optimiser
    # ------------------------------------------------------------------

    def fit(
        self,
        d: int = 2,
        maxiter: int = 1000,
        tol: float = 1e-7,
        verbose: bool = False,
        seed: int = 42,
        C_init: ArrayLike | None = None,
    ) -> "RSDR":
        """Fit RSDR via Riemannian CG on St(p, d).

        Args:
            d: Target dimensionality.
            maxiter: Maximum iterations (default 1000, matching R).
            tol: Gradient-norm convergence threshold (default 1e-7).
            verbose: Print progress every 10 iterations.
            seed: Random seed for Stiefel initialisation.
            C_init: Optional (p, d) initial point on St(p, d).
                    Use ``beta_to_C`` to convert from original-space directions.

        Raises:
            ValueError: If d < 1, d > p, or C_init has wrong shape.
        """
        n, p = self.n, self.p

        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if d > p:
            raise ValueError(f"d must be <= p ({p}), got {d}")

        Z, B, alpha, eta = self.Z, self.B, self.alpha, self.eta

        # --- Initialisation ---
        if C_init is not None:
            C = jnp.array(C_init, dtype=jnp.float64)
            if C.shape != (p, d):
                raise ValueError(f"C_init shape {C.shape} != ({p}, {d})")
            U, _, Vt = jnp.linalg.svd(C, full_matrices=False)
            C = U @ Vt
        else:
            key = jax.random.PRNGKey(seed)
            M = jax.random.normal(key, (p, d), dtype=jnp.float64)
            C, _ = jnp.linalg.qr(M)
            C = C[:, :d]

        # --- Initial gradient ---
        f_val, eg = self._val_and_egrad(C, Z, B, alpha, eta)
        rg = stiefel_proj(C, eg)
        rg_norm = jnp.linalg.norm(rg)
        direction = -rg

        history = [{"iter": 0, "f": float(f_val), "grad_norm": float(rg_norm)}]

        if verbose:
            print(f"Iter {0:4d} | f: {f_val:.8f} | Grad Norm: {rg_norm:.8e}")

        # --- RCG main loop ---
        for k in range(maxiter):
            if rg_norm < tol:
                if verbose:
                    print(f"Converged at iter {k} (grad norm {rg_norm:.8e} < {tol})")
                break

            # Armijo backtracking line search
            step = 1.0
            c_armijo = 1e-4
            slope = float(stiefel_inner(rg, direction))

            # If CG direction is not descent, fall back to steepest descent
            if slope >= 0:
                direction = -rg
                slope = float(stiefel_inner(rg, direction))

            ls_success = False
            for _ in range(30):
                C_trial = project_stiefel(C + step * direction, d)
                f_trial = self.objective(C_trial, Z, B, alpha, eta)
                if float(f_trial) <= float(f_val) + c_armijo * step * slope:
                    ls_success = True
                    break
                step *= 0.5

            if not ls_success:
                if verbose:
                    print(f"Line search failed at iter {k+1}")
                break

            C_new = C_trial
            f_new = f_trial

            _, eg_new = self._val_and_egrad(C_new, Z, B, alpha, eta)
            rg_new = stiefel_proj(C_new, eg_new)

            # Polak-Ribiere+ with vector transport
            rg_transported = stiefel_proj(C_new, rg)
            dir_transported = stiefel_proj(C_new, direction)

            diff = rg_new - rg_transported
            denom = float(stiefel_inner(rg, rg))
            if denom > 1e-30:
                beta_pr = max(0.0, float(stiefel_inner(rg_new, diff)) / denom)
            else:
                beta_pr = 0.0

            direction = -rg_new + beta_pr * dir_transported

            # Reset to steepest descent if not a descent direction
            if float(stiefel_inner(rg_new, direction)) > 0:
                direction = -rg_new

            C = C_new
            f_val = f_new
            rg = rg_new
            rg_norm = jnp.linalg.norm(rg)

            history.append({"iter": k + 1, "f": float(f_val),
                            "grad_norm": float(rg_norm)})

            if verbose and (k + 1) % 10 == 0:
                print(f"Iter {k+1:4d} | f: {f_val:.8f} | Grad Norm: {rg_norm:.8e}")

        # --- Store results ---
        self.C_value = C
        self.f_value = float(f_val)
        self.dirs = self.Nhalf_inv @ C           # beta = Sigma^{-1/2} @ C
        self.projected_data = self.X_orig @ self.dirs
        self.history = history
        self.fitted = True
        return self


# ============================================================================
# Quick demo
# ============================================================================

if __name__ == "__main__":
    import time

    print("Generating data...")
    np.random.seed(42)
    n, p = 200, 10
    X_data = np.random.randn(n, p)
    Y_data = np.sin(X_data[:, 0]) + (X_data[:, 1]) ** 2 + 0.1 * np.random.randn(n)

    print("Fitting RSDR...")
    model = RSDR(X_data, Y_data, alpha=0.5)

    t0 = time.time()
    model.fit(d=2, verbose=True)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.4f}s,  f = {model.f_value:.8f}")
    print("Estimated beta (first 5 rows):")
    print(np.round(np.array(model.dirs[:5, :]), 4))
