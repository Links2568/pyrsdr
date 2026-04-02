"""
Shared utilities for dimension reduction methods.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import svd
from numpy.typing import ArrayLike, NDArray


def canonical_angles(A: ArrayLike, B: ArrayLike) -> NDArray[np.floating]:
    """Canonical (principal) angles between two subspaces (in radians).

    Args:
        A: (p, d1) matrix whose columns span the first subspace.
        B: (p, d2) matrix whose columns span the second subspace.

    Returns:
        Array of min(d1, d2) canonical angles in radians (sorted ascending).

    Raises:
        ValueError: If A or B is not 2-D, or row counts differ.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got {A.ndim}-D")
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D, got {B.ndim}-D")
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"A and B must have the same number of rows, "
            f"got {A.shape[0]} and {B.shape[0]}"
        )
    U_A, _, _ = np.linalg.svd(A, full_matrices=False)
    U_B, _, _ = np.linalg.svd(B, full_matrices=False)
    _, s, _ = np.linalg.svd(U_A.T @ U_B, full_matrices=False)
    s = np.clip(s, -1, 1)
    return np.arccos(s)


def canonical_angles_deg(A: ArrayLike, B: ArrayLike) -> NDArray[np.floating]:
    """Canonical angles between two subspaces in degrees.

    Args:
        A: (p, d1) matrix whose columns span the first subspace.
        B: (p, d2) matrix whose columns span the second subspace.

    Returns:
        Array of min(d1, d2) canonical angles in degrees.
    """
    return np.degrees(canonical_angles(A, B))


def procrustes_align(
    Z_ref: ArrayLike, Z_target: ArrayLike
) -> NDArray[np.floating]:
    """Align Z_target to Z_ref via orthogonal Procrustes (rotation + reflection).

    Args:
        Z_ref:    (n, d) reference coordinates.
        Z_target: (n, d) target coordinates to align.

    Returns:
        (n, d) aligned coordinates.

    Raises:
        ValueError: If inputs are not 2-D or shapes don't match.
    """
    Z_ref = np.asarray(Z_ref, dtype=np.float64)
    Z_target = np.asarray(Z_target, dtype=np.float64)
    if Z_ref.ndim != 2:
        raise ValueError(f"Z_ref must be 2-D, got {Z_ref.ndim}-D")
    if Z_target.ndim != 2:
        raise ValueError(f"Z_target must be 2-D, got {Z_target.ndim}-D")
    if Z_ref.shape != Z_target.shape:
        raise ValueError(
            f"Z_ref and Z_target must have the same shape, "
            f"got {Z_ref.shape} and {Z_target.shape}"
        )
    Z_ref_c = Z_ref - Z_ref.mean(axis=0)
    Z_tgt_c = Z_target - Z_target.mean(axis=0)
    U, _, Vt = svd(Z_ref_c.T @ Z_tgt_c)
    R = (U @ Vt).T  # optimal rotation
    return Z_tgt_c @ R
