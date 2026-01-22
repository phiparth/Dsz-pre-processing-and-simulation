import numpy as np
import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
try:
    from google.colab import files as colab_files
except Exception:
    colab_files = None

# ---------- user params ----------
pattern = "/content/Normalized E3 Cell *.csv"   # adjust to match your files
out_path = "/content/E_mean_ellipse.csv"        # averaged CSV path
mask_threshold = 0.5    # fraction of cells that must have signal at (rho,theta) to include that radius
apply_combined_ellipse_mask = True   # if True, clip mean outside the combined ellipse

a = 2.5              # semi-major axis scale
b = 1.0              # semi-minor axis scale (a:b = 2.5:1)
theta0_deg = 0.0     # rotate angles so theta=0 aligns to major axis (degrees)
cart_grid_res = 400  # interpolation resolution for final PNG/TIFF
png_path = "/content/E3_final_average_ellipse.png"
tiff_path = "/content/E3_final_average_ellipse.tif"
# ----------------------------------

files = sorted(glob(pattern))
if len(files) == 0:
    raise SystemExit("No input files found for pattern: " + pattern)

# Read theta and rho from the first file (assumes same grids for all files)
df0 = pd.read_csv(files[0], header=None)
theta = pd.to_numeric(df0.iloc[0, 1:].values, errors='coerce').astype(float)   # (Nθ,)
rho   = pd.to_numeric(df0.iloc[1:, 0].values, errors='coerce').astype(float)   # (Nρ,)

all_cells = []
for f in files:
    df = pd.read_csv(f, header=None)
    I = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce').values  # NaN where missing
    if I.shape != (len(rho), len(theta)):
        raise SystemExit(f"Grid mismatch in {f}: expected {(len(rho),len(theta))}, got {I.shape}")
    all_cells.append(I)

all_cells = np.stack(all_cells)  # shape (Ncells, N_rho, N_theta)
Ncells, N_rho, N_theta = all_cells.shape
print(f"Loaded {Ncells} files. grid shape = (rho: {N_rho}, theta: {N_theta})")

# Compute mean across files ignoring NaNs
I_mean = np.nanmean(all_cells, axis=0)   # shape (N_rho, N_theta)

# Optionally compute a combined-ellipse mask and clip the mean
if apply_combined_ellipse_mask:
    presence = ~np.isnan(all_cells)     # True where that cell had a value
    presence_frac = np.sum(presence, axis=0) / float(Ncells)   # shape (N_rho, N_theta)
    # For each theta, find the maximum rho-index that meets the threshold.
    combined_mask = np.zeros_like(I_mean, dtype=bool)
    for a_idx in range(N_theta):
        col = presence_frac[:, a_idx]
        valid_idx = np.where(col >= mask_threshold)[0]
        if valid_idx.size > 0:
            max_idx = valid_idx.max()
            combined_mask[:max_idx+1, a_idx] = True
    I_mean_masked = np.array(I_mean, copy=True)
    I_mean_masked[~combined_mask] = np.nan
    I_out = I_mean_masked
else:
    I_out = I_mean

# Build DataFrame with same layout as input CSVs:
mean_df = pd.DataFrame(I_out, index=rho, columns=theta)
mean_df.index.name = ""      # keep first cell blank-like
mean_df.columns.name = ""

# Save averaged CSV
mean_df.to_csv(out_path, float_format="%.6g", na_rep="")
print(f"Saved averaged ellipse CSV to: {out_path}")
if colab_files:
    try:
        colab_files.download(out_path)
    except Exception:
        pass

# ----------------------------
# Build ellipse-mapped Cartesian image from I_out (rho x theta)
# ----------------------------
rho_target = np.asarray(rho)            # (Nrho,)
theta_target = np.asarray(theta)        # (Ntheta,) degrees
I_rho = np.asarray(I_out)               # (Nrho, Ntheta)

# Build param grids
Theta = np.deg2rad(theta_target)                          # (Ntheta,)
Rho, Theta_grid = np.meshgrid(rho_target, Theta, indexing='ij')  # (Nrho,Ntheta)

# rotate theta so theta=0 aligns with major axis
theta0 = np.deg2rad(theta0_deg)
Theta_rot = Theta_grid - theta0

# compute ellipse boundary radius r_ellipse(theta)
# r_ellipse(theta) = (a*b) / sqrt((b*cos(theta))^2 + (a*sin(theta))^2)
theta_rot_vec = Theta - theta0   # shape (Ntheta,)
num = a * b
den = np.sqrt((b * np.cos(theta_rot_vec))**2 + (a * np.sin(theta_rot_vec))**2)
r_ellipse_theta = num / den    # (Ntheta,) physical radial distance at rho=1 for each theta

# physical radius grid
r_phys_grid = Rho * r_ellipse_theta[None, :]   # (Nrho,Ntheta)

# Cartesian coordinates (rotated)
X = r_phys_grid * np.cos(Theta_rot)
Y = r_phys_grid * np.sin(Theta_rot)
Z = I_rho

# Interpolate scattered (X,Y,Z) -> regular Cartesian grid for image
pts = np.vstack((X.flatten(), Y.flatten())).T
vals = Z.flatten()
valid = np.isfinite(vals)

if valid.sum() < 10:
    raise RuntimeError("Too few valid intensity points to interpolate an image.")

xi = np.linspace(np.nanmin(X), np.nanmax(X), cart_grid_res)
yi = np.linspace(np.nanmin(Y), np.nanmax(Y), cart_grid_res)
Xi, Yi = np.meshgrid(xi, yi)

gridZ = griddata(pts[valid], vals[valid], (Xi, Yi), method='linear', fill_value=np.nan)
if np.isnan(gridZ).all():
    gridZ = griddata(pts[valid], vals[valid], (Xi, Yi), method='nearest')

# Display final image
plt.figure(figsize=(6,6))
plt.imshow(gridZ, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap='viridis')
plt.gca().set_aspect('equal')
plt.axis('off')
plt.title(f"Final averaged ellipse (a:b = {a}:{b})")
plt.colorbar(label='Intensity')
plt.show()

# Save PNG
plt.imsave(png_path, gridZ, origin='lower', cmap='viridis', dpi=300)
print("Saved PNG ->", png_path)
if colab_files:
    try:
        colab_files.download(png_path)
    except Exception:
        pass

# Save TIFF (lossless) if tifffile is available
try:
    import tifffile as tiff
    tiff.imwrite(tiff_path, np.nan_to_num(gridZ, nan=0.0).astype(np.float32))
    print("Saved TIFF ->", tiff_path)
    if colab_files:
        try:
            colab_files.download(tiff_path)
        except Exception:
            pass
except Exception:
    print("tifffile not available; TIFF not saved.")
