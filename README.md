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

### Sufficient Dimension Reduction — Foundations

- Li, K.-C. (1991). Sliced inverse regression for dimension reduction. *Journal of the American Statistical Association*, 86(414), 316–327.
- Cook, R. D. (1998). *Regression Graphics*. New York: Wiley.
- Li, B. (2018). *Sufficient Dimension Reduction*. Chapman & Hall/CRC.

### Distance-Covariance SDR (RSDR)

- Sheng, W. & Yin, X. (2016). Sufficient Dimension Reduction via Distance Covariance. *Journal of Computational and Graphical Statistics*, 25(3), 684–708.

### Multivariate Response & Related SDR Work

- Li, K.-C., Aragon, Y., Shedden, K., & Thomas Agnan, C. (2003). Dimension Reduction for Multivariate Response Data. *JASA*, 98(461), 99–109.
- Shedden, K. & Li, K.-C. (2003). Dimension reduction and spatiotemporal regression: applications to neuroimaging. *Computing in Science & Engineering*, 5(5), 30–36.
- Huang, S.-H., Shedden, K., & Chang, H.-W. (2023). Inference for the dimension of a regression relationship using pseudo-covariates. *Biometrics*, 79(3), 2394–2403.

### Stiefel Manifold Optimization

- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
- Edelman, A., Arias, T. A., & Smith, S. T. (1998). The geometry of algorithms with orthogonality constraints. *SIAM J. Matrix Anal. Appl.*, 20(2), 303–353.

### R Packages (Baselines)

- Weisberg, S. (2002). [dr: Methods for Dimension Reduction for Regression](https://cran.r-project.org/package=dr). CRAN.
- Adragni, K. P. & Cook, R. D. (2014). ldr: An R Software Package for Likelihood-Based Sufficient Dimension Reduction. *Journal of Statistical Software*, 61(3).
- Huang, Y., Yu, Z. & Zhang, J. (2024). [rSDR: Robust Sufficient Dimension Reduction](https://cran.r-project.org/package=rSDR). CRAN.

### Related Software

- Shedden, K. — [github.com/kshedden](https://github.com/kshedden): Go/Python/Julia packages for statistical modeling, multivariate analysis, and dimension reduction.

## License

MIT
