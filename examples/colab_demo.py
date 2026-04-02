"""
pyrsdr Colab Demo — copy-paste into a Colab cell.
"""

# ── Cell 1: Install ──
# !pip install pyrsdr

# ── Cell 2: Demo ──
import numpy as np
from pyrsdr import RSDR, SIR, canonical_angles_deg, rsdr_cv

# ---------- Generate fake data ----------
# True model: y = sin(X1 + X2) + (X3)^2 + noise
# True subspace is 2-D: [1,1,0,...,0] and [0,0,1,0,...,0]
rng = np.random.default_rng(42)
n, p = 500, 8

X = rng.standard_normal((n, p))
y = np.sin(X[:, 0] + X[:, 1]) + X[:, 2] ** 2 + 0.3 * rng.standard_normal(n)

# True directions (for comparison)
G = np.zeros((p, 2))
G[0, 0] = 1; G[1, 0] = 1    # direction 1
G[2, 1] = 1                   # direction 2
G[:, 0] /= np.linalg.norm(G[:, 0])
G[:, 1] /= np.linalg.norm(G[:, 1])

# ---------- RSDR (random init) ----------
model = RSDR(X, y, alpha=0.5)
model.fit(d=2, maxiter=2000, verbose=True)

angles = canonical_angles_deg(G, np.array(model.dirs))
print(f"\n[RSDR random] objective = {model.f_value:.6f}")
print(f"  Canonical angles to truth: {angles.round(2)} deg")

# ---------- RSDR (SIR init) ----------
model_sir = RSDR(X, y, alpha=0.5)
model_sir.fit(d=2, maxiter=2000, init="sir", verbose=True)

angles_sir = canonical_angles_deg(G, np.array(model_sir.dirs))
print(f"\n[RSDR sir] objective = {model_sir.f_value:.6f}")
print(f"  Canonical angles to truth: {angles_sir.round(2)} deg")

# ---------- SIR (for reference) ----------
sir = SIR(nslice=10)
sir.fit(X, y, ndir=2)

angles_sir_only = canonical_angles_deg(G, sir.dirs[:, :2])
print(f"\n[SIR only] angles to truth: {angles_sir_only.round(2)} deg")

# ---------- CV select alpha ----------
result = rsdr_cv(X, y, d=2,
                 alpha_grid=[0.25, 0.5, 0.75, 1.0],
                 K=5, init="sir", verbose=True)
print(f"\nBest alpha: {result['alpha_star']}")

# ---------- Scatter plot ----------
import matplotlib.pyplot as plt

proj = X @ np.array(model_sir.dirs)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, ax in enumerate(axes):
    sc = ax.scatter(proj[:, i], y, c=y, cmap="viridis", s=8, alpha=0.7)
    ax.set_xlabel(f"RSDR direction {i+1}")
    ax.set_ylabel("y")
    ax.set_title(f"Dir {i+1} (angle={angles_sir[i]:.1f}°)")
plt.colorbar(sc, ax=axes, label="y")
plt.tight_layout()
plt.savefig("rsdr_demo.png", dpi=150)
plt.show()
print("Plot saved to rsdr_demo.png")
