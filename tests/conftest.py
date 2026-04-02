"""Shared fixtures for pyrsdr test suite."""
import os

import numpy as np
import pytest

TESTDATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "test_data"
)


@pytest.fixture
def rng():
    """Deterministic NumPy random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def linear_data(rng):
    """y = X1 - X2 + noise, 200 x 6."""
    n, p = 200, 6
    X = rng.standard_normal((n, p))
    X = X - X.mean(axis=0)
    y = X[:, 0] - X[:, 1] + 0.5 * rng.standard_normal(n)
    return X, y
