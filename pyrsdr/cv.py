"""
Cross-validation and bootstrap selection of alpha for RSDR.

Training uses each candidate alpha for optimisation, but validation always
evaluates the learned subspace C with a **fixed** ``eval_alpha`` (default 0.5).
This keeps the validation metric on the same scale across candidates so that
they can be compared fairly.

Typical usage::

    from pyrsdr.cv import rsdr_cv

    result = rsdr_cv(X, Y, d=2,
                     alpha_grid=[0.25, 0.5, 0.75, 1.0],
                     K=5, seed=42, verbose=True)
    print(result["alpha_star"])
    print(result["summary"])
"""
from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from pyrsdr.rsdr import RSDR, compute_B_matrix


# ── Default evaluation alpha ────────────────────────────────────────────────

EVAL_ALPHA = 0.5


# ── Validation cost helper ───────────────────────────────────────────────────

def _validation_cost(
    model: RSDR,
    X_test: ArrayLike,
    Y_test: ArrayLike,
    eval_alpha: float = EVAL_ALPHA,
) -> float:
    """Evaluate the RSDR objective on held-out data.

    Uses the **training** whitening transform (model.Nhalf_inv) to project
    X_test, but computes B_test and the distance kernel with ``eval_alpha``
    (not the alpha used for training).  This keeps validation costs on the
    same scale across different training alphas.

    Rows where Y_test contains NaN are automatically dropped.

    Parameters
    ----------
    model : RSDR
        A fitted RSDR object (must have .C_value, .Nhalf_inv, .eta).
    X_test : array-like, shape (n_test, p)
        Held-out predictor data.
    Y_test : array-like, shape (n_test,) or (n_test, q)
        Held-out response data.
    eval_alpha : float
        Alpha exponent used to compute the validation objective (default 0.5).

    Returns
    -------
    float
        Objective value (negative distance covariance — lower is better).
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    Y_test = np.asarray(Y_test, dtype=np.float64)
    if Y_test.ndim == 1:
        Y_test = Y_test[:, None]

    # Drop NaN rows in Y_test
    keep = ~np.any(np.isnan(Y_test), axis=1)
    if not keep.all():
        X_test = X_test[keep]
        Y_test = Y_test[keep]

    Z_test = jnp.asarray(X_test) @ model.Nhalf_inv
    B_test = compute_B_matrix(jnp.asarray(Y_test), eval_alpha)
    return float(RSDR.objective(model.C_value, Z_test, B_test,
                                eval_alpha, model.eta))


# ── K-fold cross-validation ─────────────────────────────────────────────────

def rsdr_cv(
    X: ArrayLike,
    Y: ArrayLike,
    d: int,
    alpha_grid: Sequence[float],
    K: int = 5,
    maxiter: int = 1000,
    tol: float = 1e-7,
    eta: float = 1e-10,
    eval_alpha: float = EVAL_ALPHA,
    seed: int = 42,
    verbose: bool = False,
    init: str = "random",
) -> dict:
    """K-fold cross-validation for selecting alpha in RSDR.

    For every candidate alpha and every fold the procedure:

    1. Trains RSDR on the training split with that alpha.
    2. Evaluates the learned subspace C on the validation split using a
       **fixed** ``eval_alpha`` (default 0.5) so that costs are on the same
       scale and comparable across candidates.
    3. Picks the alpha with the lowest (most negative) mean validation cost.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Predictor matrix.
    Y : array-like, shape (n,) or (n, q)
        Response vector / matrix.
    d : int
        Target dimensionality.
    alpha_grid : sequence of float
        Candidate alpha values (e.g. ``[0.25, 0.5, 0.75, 1.0]``).
    K : int
        Number of folds (default 5).
    maxiter : int
        Maximum RCG iterations per fit.
    tol : float
        Gradient-norm convergence threshold.
    eta : float
        Smoothing parameter (sqrt(D² + eta)).
    eval_alpha : float
        Alpha used for computing validation cost (default 0.5).
    seed : int
        Controls both fold assignment and RSDR initialisation.
    verbose : bool
        Print per-fold and per-alpha summaries.
    init : str
        Initialisation strategy for each RSDR fit — ``"random"`` (default)
        or ``"sir"`` (SIR warm start).

    Returns
    -------
    dict
        ``alpha_star``  – best alpha (float).
        ``eval_alpha``  – the alpha used for validation scoring.
        ``summary``     – list of dicts, one per alpha, each with keys
                          ``alpha``, ``mean_cost``, ``sd_cost``, ``fold_costs``.

    Raises
    ------
    ValueError
        If alpha_grid is empty, K < 2, or d < 1.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]

    alpha_grid = list(alpha_grid)
    if len(alpha_grid) == 0:
        raise ValueError("alpha_grid must not be empty")
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.ndim}-D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of rows, "
            f"got {X.shape[0]} and {Y.shape[0]}"
        )

    # ── pre-filter NaN rows so folds only contain usable data ──
    keep = ~np.any(np.isnan(Y), axis=1)
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        X, Y = X[keep], Y[keep]
        if verbose:
            print(f"rsdr_cv: dropped {n_dropped} NaN-Y rows "
                  f"({X.shape[0]} retained)")

    n = X.shape[0]

    if verbose:
        print(f"rsdr_cv: eval_alpha={eval_alpha} (fixed for all validation)")

    # ── create K folds (shuffled, deterministic) ──
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, K)

    summary = []

    for alpha in alpha_grid:
        fold_costs = []

        for k in range(K):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(K) if j != k])

            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            # Train with candidate alpha
            model = RSDR(X_train, Y_train, alpha=alpha, eta=eta)
            model.fit(d=d, maxiter=maxiter, tol=tol,
                      seed=seed + k, verbose=False, init=init)

            # Validate with fixed eval_alpha
            cost_k = _validation_cost(model, X_test, Y_test,
                                      eval_alpha=eval_alpha)
            fold_costs.append(cost_k)

            if verbose:
                print(f"  alpha={alpha:.4f}  fold {k+1}/{K}  cost={cost_k:.6f}")

        mean_cost = float(np.mean(fold_costs))
        sd_cost = float(np.std(fold_costs, ddof=1)) if K > 1 else 0.0

        summary.append({
            "alpha": alpha,
            "mean_cost": mean_cost,
            "sd_cost": sd_cost,
            "fold_costs": fold_costs,
        })

        if verbose:
            print(f"  alpha={alpha:.4f}  =>  mean={mean_cost:.6f} ± {sd_cost:.6f}\n")

    # ── pick best alpha (lowest mean cost = most negative objective) ──
    best = min(summary, key=lambda r: r["mean_cost"])
    alpha_star = best["alpha"]

    if verbose:
        print(f"Best alpha: {alpha_star:.4f}  "
              f"(mean_cost={best['mean_cost']:.6f})")

    return {"alpha_star": alpha_star, "eval_alpha": eval_alpha, "summary": summary}


# ── Bootstrap selection (optional) ───────────────────────────────────────────

def rsdr_bootstrap(
    X: ArrayLike,
    Y: ArrayLike,
    d: int,
    alpha_grid: Sequence[float],
    B: int = 50,
    maxiter: int = 1000,
    tol: float = 1e-7,
    eta: float = 1e-10,
    eval_alpha: float = EVAL_ALPHA,
    seed: int = 42,
    verbose: bool = False,
    init: str = "random",
) -> dict:
    """Out-of-bag bootstrap selection of alpha for RSDR.

    For every candidate alpha and every bootstrap replicate:

    1. Draw a bootstrap sample (with replacement, size n) as training set.
    2. Use the out-of-bag (OOB) indices as the validation set.
    3. Train RSDR on the bootstrap sample with that alpha.
    4. Evaluate the objective on OOB data with the fixed ``eval_alpha``.

    Picks the alpha with the lowest mean OOB cost.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Predictor matrix.
    Y : array-like, shape (n,) or (n, q)
        Response vector / matrix.
    d : int
        Target dimensionality.
    alpha_grid : sequence of float
        Candidate alpha values.
    B : int
        Number of bootstrap replicates (default 50).
    eval_alpha : float
        Alpha used for computing validation cost (default 0.5).
    maxiter, tol, eta, seed, verbose, init
        Passed through to RSDR (see :func:`rsdr_cv`).

    Returns
    -------
    dict
        ``alpha_star``  – best alpha (float).
        ``eval_alpha``  – the alpha used for validation scoring.
        ``summary``     – list of dicts, one per alpha, each with keys
                          ``alpha``, ``mean_cost``, ``sd_cost``, ``rep_costs``.

    Raises
    ------
    ValueError
        If alpha_grid is empty, B < 1, or d < 1.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]

    alpha_grid = list(alpha_grid)
    if len(alpha_grid) == 0:
        raise ValueError("alpha_grid must not be empty")
    if B < 1:
        raise ValueError(f"B must be >= 1, got {B}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    # ── pre-filter NaN rows ──
    keep = ~np.any(np.isnan(Y), axis=1)
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        X, Y = X[keep], Y[keep]
        if verbose:
            print(f"rsdr_bootstrap: dropped {n_dropped} NaN-Y rows "
                  f"({X.shape[0]} retained)")

    n = X.shape[0]
    all_idx = np.arange(n)
    rng = np.random.RandomState(seed)

    summary = []

    for alpha in alpha_grid:
        rep_costs = []

        for b in range(B):
            train_idx = rng.choice(n, size=n, replace=True)
            oob_idx = np.setdiff1d(all_idx, train_idx)

            if len(oob_idx) < 2:
                continue  # degenerate replicate, skip

            X_train, Y_train = X[train_idx], Y[train_idx]
            X_oob, Y_oob = X[oob_idx], Y[oob_idx]

            model = RSDR(X_train, Y_train, alpha=alpha, eta=eta)
            model.fit(d=d, maxiter=maxiter, tol=tol,
                      seed=seed + b, verbose=False, init=init)

            cost_b = _validation_cost(model, X_oob, Y_oob,
                                      eval_alpha=eval_alpha)
            rep_costs.append(cost_b)

            if verbose:
                print(f"  alpha={alpha:.4f}  rep {b+1}/{B}  "
                      f"OOB size={len(oob_idx)}  cost={cost_b:.6f}")

        mean_cost = float(np.mean(rep_costs)) if rep_costs else float("nan")
        sd_cost = (float(np.std(rep_costs, ddof=1))
                   if len(rep_costs) > 1 else 0.0)

        summary.append({
            "alpha": alpha,
            "mean_cost": mean_cost,
            "sd_cost": sd_cost,
            "rep_costs": rep_costs,
        })

        if verbose:
            print(f"  alpha={alpha:.4f}  =>  mean={mean_cost:.6f} ± {sd_cost:.6f}"
                  f"  ({len(rep_costs)} reps)\n")

    best = min(summary, key=lambda r: r["mean_cost"])
    alpha_star = best["alpha"]

    if verbose:
        print(f"Best alpha: {alpha_star:.4f}  "
              f"(mean_cost={best['mean_cost']:.6f})")

    return {"alpha_star": alpha_star, "eval_alpha": eval_alpha, "summary": summary}
