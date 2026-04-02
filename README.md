# pyrsdr

JAX-accelerated **Sufficient Dimension Reduction** for Python.

| Method | Description |
|--------|-------------|
| **RSDR** | Distance-covariance SDR via Riemannian CG on the Stiefel manifold (Sheng & Yin 2016) |
| **SIR** | Sliced Inverse Regression (Li 1991) |

## Installation

```bash
pip install pyrsdr
```

## Quick Start

```python
from pyrsdr import RSDR, SIR

# RSDR — distance-covariance SDR
model = RSDR(X, Y, alpha=0.5)
model.fit(d=2)
projected = model.projected_data   # (n, d)
directions = model.dirs            # (p, d)

# SIR — sliced inverse regression
sir = SIR(nslice=10)
sir.fit(X, y, ndir=2)
directions = sir.dirs
```

## Cross-validation

```python
from pyrsdr import rsdr_cv

result = rsdr_cv(X, Y, d=2,
                 alpha_grid=[0.25, 0.5, 0.75, 1.0],
                 K=5, seed=42)
print(result["alpha_star"])
```

## References

- Sheng, W. & Yin, X. (2016). Sufficient Dimension Reduction via Distance Covariance. *Journal of Computational and Graphical Statistics*.
- Li, K-C. (1991). Sliced Inverse Regression for Dimension Reduction. *JASA*.

## License

MIT
