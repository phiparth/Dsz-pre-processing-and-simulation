import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from google.colab import files


# ----------------------------
a = 2.5      # semi-major axis
b = 1.0      # semi-minor axis  -> a:b = 2.5:1
theta0_deg = 0.0  
rho_bins = 50
df = pd.read_csv("/content/final.csv", header=None)

theta = df.iloc[0, 1:].values.astype(float)   # (Ntheta,) in deg
r = df.iloc[1:, 0].values.astype(float)       # (Nr,) in px
I = df.iloc[1:, 1:].values.astype(float)      # (Nr, Ntheta)

r_max = np.array([ r[np.isfinite(I[:, j])].max() if np.isfinite(I[:, j]).any() else np.nan
                   for j in range(I.shape[1]) ])

# rho grid (normalized radius)
rho_target = np.linspace(0, 1, rho_bins)
# theta grid: keep original theta ordering
theta_target = theta  # degrees from CSV
theta_rad = np.deg2rad(theta_target)

# interpolate each θ along rho (same as before)
rho = r[:, None] / r_max[None, :]   # (Nr, Ntheta)

I_rho = np.full((len(rho_target), len(theta_target)), np.nan)
for j in range(len(theta_target)):
    valid = np.isfinite(I[:, j]) & np.isfinite(rho[:, j])
    if valid.sum() < 2:
        continue
    order = np.argsort(rho[valid, j])
    rv = rho[valid, j][order]
    Iv = I[valid, j][order]
    f = interp1d(rv, Iv, bounds_error=False, fill_value=np.nan)
    I_rho[:, j] = f(rho_target)

# Save normalized polar CSV (same convention you used earlier)
out_df = pd.DataFrame(I_rho, index=rho_target, columns=theta_target)
out_df.index.name = "rho"
out_df.columns.name = "theta_deg"
csv_path = "/content/cell_0_normalized_polar.csv"
out_df.to_csv(csv_path)
files.download(csv_path)
print("Saved normalized polar CSV ->", csv_path)

# ----------------------------
# Map normalized polar grid onto an ELLIPSE (correct radial variation)
# ----------------------------
# Build mesh (rho x theta)
Rho, Theta = np.meshgrid(rho_target, np.deg2rad(theta_target), indexing='ij')  # shape (Nrho, Ntheta)
# Rotate theta so theta=0 aligns with major axis if desired
theta0 = np.deg2rad(theta0_deg)
Theta_rot = Theta - theta0

# compute ellipse radial boundary r_ellipse(theta) for axis-aligned ellipse (semi-axes a,b)
# formula: r_ellipse(theta) = 1 / sqrt(cos^2(theta)/a^2 + sin^2(theta)/b^2)
# equivalent and numerically stable: (a*b) / sqrt((b*cos)^2 + (a*sin)^2)
theta_rot_vec = np.deg2rad(theta_target) - theta0  # shape (Ntheta,)
num = a * b
den = np.sqrt((b * np.cos(theta_rot_vec))**2 + (a * np.sin(theta_rot_vec))**2)
r_ellipse_theta = num / den    # shape (Ntheta,) physical radial distance at rho=1 for each theta

# Expand to grid: physical radius at each (rho,theta) = rho * r_ellipse_theta
r_phys_grid = Rho * r_ellipse_theta[None, :]   # shape (Nrho, Ntheta)

# Now compute Cartesian coordinates using rotated theta (so direction matches r_ellipse)
X = r_phys_grid * np.cos(Theta_rot)
Y = r_phys_grid * np.sin(Theta_rot)
Z = I_rho  # intensities mapped to these coords

# Optional: compute radial physical distance (should equal r_phys_grid)
R_phys = np.sqrt(X**2 + Y**2)

# Flatten and save x,y,r_phys,theta_deg,intensity CSV for simulation
flat = pd.DataFrame({
    "x": X.flatten(),
    "y": Y.flatten(),
    "r_phys": R_phys.flatten(),
    "rho": Rho.flatten(),
    "theta_deg": np.rad2deg(Theta.flatten()),
    "intensity": Z.flatten()
})
flat_path = "/content/cell_0_ellipse_xyz_phys.csv"
flat.to_csv(flat_path, index=False)
files.download(flat_path)
print("Saved ellipse-mapped (phys) x,y,rho,theta,intensity CSV ->", flat_path)

# ----------------------------
# Visualizations (quick checks)
# ----------------------------
# 1) Rectangular normalized polar view
plt.figure(figsize=(10,4))
plt.imshow(I_rho, extent=[theta_target.min(), theta_target.max(), 1, 0],
           aspect='auto', cmap='viridis')
plt.xlabel("θ (deg)")
plt.ylabel("ρ (normalized radius)")
plt.colorbar(label="Intensity")
plt.title("Normalized polar intensity (ρ, θ)")
plt.show()

# 2) Ellipse-mapped scatter view (shows physical radial variation)
plt.figure(figsize=(6,6))
mask = np.isfinite(Z)
plt.scatter(X[mask], Y[mask], c=Z[mask], s=12, cmap='viridis', marker='s')
plt.colorbar(label='Intensity')
plt.gca().set_aspect('equal')
plt.title(f"Intensity mapped to ellipse (a:b = {a}:{b}), rotated by {theta0_deg}°")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 3) Smooth Cartesian interpolation for nicer image
xi = np.linspace(X.min(), X.max(), 400)
yi = np.linspace(Y.min(), Y.max(), 400)
Xi, Yi = np.meshgrid(xi, yi)
points = np.vstack([X.flatten(), Y.flatten()]).T
values = Z.flatten()
valid = np.isfinite(values)
gridZ = griddata(points[valid], values[valid], (Xi, Yi), method='linear', fill_value=np.nan)

plt.figure(figsize=(6,6))
plt.imshow(gridZ, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.gca().set_aspect('equal')
plt.title("Smoothed ellipse view (interpolated to Cartesian grid)")
plt.show()
