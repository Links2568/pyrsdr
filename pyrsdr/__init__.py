"""
pyrsdr — JAX-based Dimension Reduction Regression

Implements sufficient dimension reduction methods using JAX for GPU acceleration:
  - RSDR: Distance-covariance based SDR via Riemannian CG on the Stiefel manifold
  - SIR:  Sliced Inverse Regression (Li 1991)

Example usage:
    from pyrsdr import RSDR, SIR

    # RSDR
    model = RSDR(X, Y, alpha=0.5)
    model.fit(d=2)
    projected = model.projected_data

    # SIR
    sir = SIR(nslice=10)
    sir.fit(X, Y, ndir=2)
    dirs = sir.dirs
"""

from pyrsdr.rsdr import RSDR
from pyrsdr.sir import SIR
from pyrsdr.utils import canonical_angles, canonical_angles_deg, procrustes_align
from pyrsdr.cv import rsdr_cv, rsdr_bootstrap

__all__ = [
    "RSDR", "SIR",
    "canonical_angles", "canonical_angles_deg", "procrustes_align",
    "rsdr_cv", "rsdr_bootstrap",
]
__version__ = "0.1.2"
