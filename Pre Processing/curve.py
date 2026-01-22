import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- USER PARAMETERS ----------
csv_path = "/content/E_mean_ellipse.csv"   # averaged ellipse CSV (rho vs theta)
a = 2.5           # ellipse semi-major axis (physical units or relative)
b = 1.0           # ellipse semi-minor axis
theta0_deg = 0.0  # rotation offset (degrees): how theta=0 maps to ellipse major axis
K = 5             # angular Fourier modes  (0..K)
N = 6             # radial polynomial order (0..N)
out_coeffs_json = "/content/E_ellipse_coeffs.json"
# -------------------------------------

# 1) Load averaged CSV (rows: rho, cols: theta)
df = pd.read_csv(csv_path, index_col=0)
rho = df.index.astype(float).values           # normalized rho ∈ [0,1]
theta_deg = df.columns.astype(float).values   # degrees
theta_rad = np.deg2rad(theta_deg)             # radians
Z = df.values                                 # shape (Nrho, Ntheta)

Nrho, Ntheta = Z.shape
print("Loaded:", csv_path, "shape:", Z.shape)

# 2) Flatten to fit; keep only finite samples
Rgrid, Tgrid = np.meshgrid(rho, theta_rad, indexing='ij')  # Rgrid.shape = (Nrho,Ntheta)
R = Rgrid.flatten()
T = Tgrid.flatten()
Zf = Z.flatten()

mask = np.isfinite(Zf)
R = R[mask]
T = T[mask]
Zf = Zf[mask]
print("Fitting using", len(Zf), "valid samples.")

# 3) Build design matrix A with basis: rho^n * cos(k*theta), and rho^n * sin(k*theta) (k>0)
terms = []
term_labels = []
for k in range(K+1):
    for n in range(N+1):
        terms.append((R**n) * np.cos(k*T))
        term_labels.append(f"cos{k}_r^{n}")
        if k > 0:
            terms.append((R**n) * np.sin(k*T))
            term_labels.append(f"sin{k}_r^{n}")

A = np.column_stack(terms)
print("Design matrix shape:", A.shape, "num_terms:", A.shape[1])

# 4) Least squares fit
coeffs, *_ = lstsq(A, Zf, rcond=None)
print("Fitted coefficients:", coeffs.shape[0])

# 5) Build evaluator based on coeffs (vectorized)
def E_norm_rho_theta(rho_arr, theta_arr):
    """
    Evaluate fitted field on arrays (rho_arr, theta_arr) with same shape or broadcastable.
    rho_arr: normalized rho in [0,1]
    theta_arr: radians
    returns: same-shaped array of fitted values (NaN if rho out of range)
    """
    rho_a = np.array(rho_arr, dtype=float)
    theta_a = np.array(theta_arr, dtype=float)
    # broadcast to 1D arrays then reshape later
    flat_r = rho_a.flatten()
    flat_t = theta_a.flatten()
    val = np.zeros(flat_r.shape, dtype=float)
    idx = 0
    term_idx = 0
    for k in range(K+1):
        for n in range(N+1):
            val += coeffs[term_idx] * (flat_r**n) * np.cos(k*flat_t)
            term_idx += 1
            if k > 0:
                val += coeffs[term_idx] * (flat_r**n) * np.sin(k*flat_t)
                term_idx += 1
    val = val.reshape(rho_a.shape)
    # Mask rho outside [0,1]
    val = np.where((rho_a>=0) & (rho_a<=1), val, np.nan)
    return val

# 6) Provide simulator-friendly wrapper: accepts physical r and theta (deg)
def r_ellipse(theta_rad_local):
    """Ellipse boundary radius at angle theta (rotated by theta0)."""
    theta_rot = theta_rad_local - np.deg2rad(theta0_deg)
    # r_ellipse = (a*b) / sqrt((b*cos)^2 + (a*sin)^2)
    num = a * b
    den = np.sqrt((b * np.cos(theta_rot))**2 + (a * np.sin(theta_rot))**2)
    return num / den

def E_fit_phys(r_phys, theta_deg_local):
    """
    Evaluate fitted z = f(r_phys, theta_deg_local).
    r_phys: scalar or array (same shape as theta)
    theta_deg_local: degrees (scalar or array)
    returns: z (same shape), NaN if r_phys outside ellipse radius for that theta
    """
    theta_r = np.deg2rad(theta_deg_local)
    r_ell = r_ellipse(theta_r)
    rho = np.array(r_phys, dtype=float) / r_ell
    return E_norm_rho_theta(rho, theta_r)

# 7) Compute RMSE on training data (sanity)
# Recreate fitted values at original sample points
Z_pred_at_samples = E_norm_rho_theta(Rgrid, Tgrid).flatten()[mask]  # uses normalized rho grid
rmse = np.sqrt(np.nanmean((Z_pred_at_samples - Zf)**2))
print(f"RMSE on fitted samples: {rmse:.6g}")

# 8) Print human readable equation (compact)
print("\nEquation terms (nonzero-ish):")
term_idx = 0
for k in range(K+1):
    for n in range(N+1):
        c1 = coeffs[term_idx]
        print(f"{c1:+.6g} * rho^{n} * cos({k}*theta)", end="")
        term_idx += 1
        if k > 0:
            c2 = coeffs[term_idx]
            print(f"  {c2:+.6g} * rho^{n} * sin({k}*theta)", end="")
            term_idx += 1
        print()

# 9) Save coefficients + meta
meta = {
    "a": float(a), "b": float(b), "theta0_deg": float(theta0_deg),
    "K": int(K), "N": int(N),
    "rho_grid_len": int(Nrho), "theta_grid_len": int(Ntheta)
}
payload = {"meta": meta, "term_labels": term_labels, "coeffs": coeffs.tolist()}
with open(out_coeffs_json, "w") as fh:
    json.dump(payload, fh, indent=2)
print("Saved coeffs JSON ->", out_coeffs_json)

# 10) Visualize fitted surface on normalized grid and physical ellipse
# Normalized (rho, theta) visualization - same as circular case
rho_plot = np.linspace(0, 1, 200)
theta_plot = np.linspace(0, 2*np.pi, 360)
Rplot, Tplot = np.meshgrid(rho_plot, theta_plot, indexing="ij")
Zfit_norm = E_norm_rho_theta(Rplot, Tplot)

plt.figure(figsize=(8,4))
plt.imshow(Zfit_norm, extent=[0,360,1,0], aspect="auto", cmap="viridis")
plt.xlabel("θ (deg)")
plt.ylabel("ρ")
plt.colorbar(label="Fitted value")
plt.title("Fitted field (normalized rho vs theta)")
plt.show()

# Physical ellipse mapping: compute r_phys grid and plot on Cartesian
# r_phys(ρ,θ) = ρ * r_ellipse(θ)
r_ell_theta = r_ellipse(theta_plot)  # shape (Ntheta,)
Rphys = Rplot * r_ell_theta[None, :]  # (Nrho, Ntheta)
X = Rphys * np.cos(Tplot)
Y = Rphys * np.sin(Tplot)
Zfit_phys = Zfit_norm  # same values, now mapped to X,Y

# Quick 2D ellipse image (interpolate to Cartesian grid)
from scipy.interpolate import griddata
pts = np.vstack([X.flatten(), Y.flatten()]).T
vals = Zfit_phys.flatten()
valid = np.isfinite(vals)
xi = np.linspace(X.min(), X.max(), 400)
yi = np.linspace(Y.min(), Y.max(), 400)
Xi, Yi = np.meshgrid(xi, yi)
gridZ = griddata(pts[valid], vals[valid], (Xi, Yi), method='linear', fill_value=np.nan)

plt.figure(figsize=(6,6))
plt.imshow(gridZ, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap='viridis')
plt.gca().set_aspect('equal')
plt.axis('off')
plt.title("Fitted field mapped to ellipse (physical coords)")
plt.colorbar(label='Fitted value')
plt.show()

# 11) Example: evaluate E at some physical points
# scalar example
theta_test_deg = 10.0
r_test_phys = r_ellipse(np.deg2rad(theta_test_deg)) * 0.5  # half-way to ellipse boundary
print("E_fit_phys(scalar):", E_fit_phys(r_test_phys, theta_test_deg))

# vector example
thetas_vec = np.linspace(0, 360, 36, endpoint=False)
r_phys_vec = r_ellipse(np.deg2rad(thetas_vec)) * 0.7
Evals = E_fit_phys(r_phys_vec, thetas_vec)
print("E_fit_phys(vector) shape:", Evals.shape)
