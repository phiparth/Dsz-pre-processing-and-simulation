import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import json
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from google.colab import files

# ---------- USER PARAMETERS ----------
csv_path = "/content/Normalized_Cut_07.csv"  # The file generated previously
out_coeffs_json = "/content/cell_0_ellipse_coeffs.json"

# Ellipse geometry (Must match the mapping used to generate the CSV)
a = 2.5           # semi-major axis
b = 1.0           # semi-minor axis
theta0_deg = 0.0  # The CSV is already aligned to the major axis

# Fitting Complexity
K = 6             # Angular Fourier modes (0..K) -> Controls angular detail
N = 6             # Radial polynomial order (0..N) -> Controls radial profile detail
# -------------------------------------

# 1) Load Data
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

# Extract columns (The CSV is already "tidy"/flattened)
rho_data = df['rho_norm'].values
theta_deg_data = df['theta_aligned_deg'].values
intensity_data = df['intensity'].values

# Filter invalid data
mask = np.isfinite(rho_data) & np.isfinite(theta_deg_data) & np.isfinite(intensity_data)
rho = rho_data[mask]
theta_rad = np.deg2rad(theta_deg_data[mask])
Z = intensity_data[mask]

print(f"Fitting using {len(Z)} valid sample points.")

# 2) Build Design Matrix A
# Basis: rho^n * cos(k*theta) AND rho^n * sin(k*theta)
# Order of coeffs:
#   For k=0: n=0..N (cos terms only)
#   For k=1..K:
#       n=0..N (cos terms)
#       n=0..N (sin terms)

terms_list = []
term_labels = []

# k=0 case (DC component and radial dependence, no sine)
for n in range(N + 1):
    terms_list.append((rho**n) * np.cos(0 * theta_rad))
    term_labels.append(f"k0_cos_n{n}")

# k > 0 cases
for k in range(1, K + 1):
    # Cosine terms
    for n in range(N + 1):
        terms_list.append((rho**n) * np.cos(k * theta_rad))
        term_labels.append(f"k{k}_cos_n{n}")
    # Sine terms
    for n in range(N + 1):
        terms_list.append((rho**n) * np.sin(k * theta_rad))
        term_labels.append(f"k{k}_sin_n{n}")

A = np.column_stack(terms_list)
print(f"Design matrix shape: {A.shape} (Samples x Coefficients)")

# 3) Least Squares Fit
print("Solving least squares...")
coeffs, residuals, rank, s = lstsq(A, Z, rcond=None)
print(f"Fit complete. Coefficients found: {len(coeffs)}")

# Calculate RMSE
Z_pred = A @ coeffs
rmse = np.sqrt(np.mean((Z - Z_pred)**2))
print(f"RMSE on training data: {rmse:.4f}")

# 4) Save Coefficients to JSON
meta_data = {
    "a": a,
    "b": b,
    "theta0_deg": theta0_deg,
    "K": K,
    "N": N,
    "rmse": rmse,
    "equation_format": "sum_k sum_n rho^n * (A_kn cos(k theta) + B_kn sin(k theta))",
    "term_order": term_labels
}

output_payload = {
    "meta": meta_data,
    "coeffs": coeffs.tolist()
}

with open(out_coeffs_json, "w") as f:
    json.dump(output_payload, f, indent=2)

print(f"Coefficients saved to: {out_coeffs_json}")
files.download(out_coeffs_json)

# ---------------------------------------------------------
# 5) Visualization & Validation
# ---------------------------------------------------------

def evaluate_model(rho_grid, theta_grid_rad, coefficients):
    """Reconstructs the field from coefficients."""
    result = np.zeros_like(rho_grid)
    idx = 0

    # k=0
    for n in range(N + 1):
        result += coefficients[idx] * (rho_grid**n)
        idx += 1

    # k>0
    for k in range(1, K + 1):
        for n in range(N + 1):
            result += coefficients[idx] * (rho_grid**n) * np.cos(k * theta_grid_rad)
            idx += 1
        for n in range(N + 1):
            result += coefficients[idx] * (rho_grid**n) * np.sin(k * theta_grid_rad)
            idx += 1
    return result

# Generate a dense grid for plotting the smooth reconstruction
grid_rho = np.linspace(0, 1, 100)
grid_theta = np.linspace(0, 2*np.pi, 180)
R_plot, T_plot = np.meshgrid(grid_rho, grid_theta)

# Evaluate the fitted function
Z_reconst = evaluate_model(R_plot, T_plot, coeffs)

# Map to Cartesian coordinates for Ellipse visualization
# r_boundary(theta) = ab / sqrt((b cos)^2 + (a sin)^2)
# r_phys = rho * r_boundary
denom = np.sqrt((b * np.cos(T_plot))**2 + (a * np.sin(T_plot))**2)
r_boundary = (a * b) / denom
R_phys_plot = R_plot * r_boundary

X_plot = R_phys_plot * np.cos(T_plot)
Y_plot = R_phys_plot * np.sin(T_plot)

# Plot
plt.figure(figsize=(12, 5))

# Subplot 1: Original Data Scatter
plt.subplot(1, 2, 1)
plt.scatter(df['x_aligned'], df['y_aligned'], c=df['intensity'], s=2, cmap='viridis')
plt.colorbar(label='Intensity')
plt.title("Original Data Points")
plt.axis('equal')

# Subplot 2: Smooth Reconstruction
plt.subplot(1, 2, 2)
plt.pcolormesh(X_plot, Y_plot, Z_reconst, shading='auto', cmap='viridis')
plt.colorbar(label='Fitted Intensity')
plt.title(f"Fitted Analytical Model\n(K={K}, N={N}, RMSE={rmse:.1f})")
plt.axis('equal')

plt.tight_layout()
plt.show()

print("\nExample Evaluation (Scalar):")
test_rho = 0.5
test_theta_deg = 45.0
val = evaluate_model(np.array([test_rho]), np.deg2rad(np.array([test_theta_deg])), coeffs)
print(f"Intensity at rho={test_rho}, theta={test_theta_deg}° : {val[0]:.2f}")

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

# --- 1. USER PARAMETERS ---
csv_path = "/content/Normalized_Cut_07.csv"
out_sparse_json = "Sparse_coeffs.json"

a, b = 2.5, 1.0
theta0_deg = 0.0
K = 6  # Start with max complexity, let Lasso prune it
N = 6

# --- 2. LOAD DATA ---
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)
mask = np.isfinite(df['rho_norm']) & np.isfinite(df['theta_aligned_deg']) & np.isfinite(df['intensity'])
rho = df['rho_norm'].values[mask]
theta_rad = np.deg2rad(df['theta_aligned_deg'].values[mask])
Z = df['intensity'].values[mask]

# --- 3. BUILD MATRIX ---
terms_list = []
term_labels = []

for n in range(N + 1):
    terms_list.append((rho**n) * np.cos(0 * theta_rad))
    term_labels.append(f"k0_cos_n{n}")

for k in range(1, K + 1):
    for n in range(N + 1):
        terms_list.append((rho**n) * np.cos(k * theta_rad))
        term_labels.append(f"k{k}_cos_n{n}")
        terms_list.append((rho**n) * np.sin(k * theta_rad))
        term_labels.append(f"k{k}_sin_n{n}")

A = np.column_stack(terms_list)

# Standardization: Lasso requires inputs to be on the same scale
# otherwise it unfairly penalizes high-order rho terms
scale_factors = np.max(np.abs(A), axis=0)
scale_factors[scale_factors == 0] = 1.0
A_scaled = A / scale_factors

# --- 4. LASSO REGRESSION ---
print("\nSolving using Lasso Regression (L1 Penalty)...")
# LassoCV automatically finds the optimal penalty strength (alpha)
lasso_model = LassoCV(cv=5, fit_intercept=False, max_iter=100000)
lasso_model.fit(A_scaled, Z)

# De-scale the coefficients so they work with your raw rho/theta formulas
coeffs_unscaled = lasso_model.coef_ / scale_factors

# Calculate Training RMSE
Z_pred = A @ coeffs_unscaled
rmse = np.sqrt(np.mean((Z - Z_pred)**2))

# --- 5. EXTRACT SPARSE TERMS ---
# Filter out coefficients that are effectively zero
active_indices = np.where(np.abs(coeffs_unscaled) > 1e-5)[0]

sparse_labels = [term_labels[i] for i in active_indices]
sparse_coeffs = [float(coeffs_unscaled[i]) for i in active_indices]

print(f"\n--- Model Reduction Results ---")
print(f"Original terms: {len(term_labels)}")
print(f"Terms forced to zero: {len(term_labels) - len(active_indices)}")
print(f"Active terms kept: {len(active_indices)}")
print(f"RMSE: {rmse:.2f}")

# --- 6. SAVE COMPACT JSON ---
meta_data = {
    "a": a,
    "b": b,
    "theta0_deg": theta0_deg,
    "original_K": K,
    "original_N": N,
    "rmse": rmse,
    "model_type": "Lasso Sparse Regression",
    "optimal_alpha": lasso_model.alpha_,
    "active_terms_count": len(active_indices),
    "term_order": sparse_labels
}

output_payload = {
    "meta": meta_data,
    "coeffs": sparse_coeffs
}

with open(out_sparse_json, "w") as f:
    json.dump(output_payload, f, indent=2)

print(f"\nSparse model saved to: {out_sparse_json}")

# Optional Plotting to visualize the remaining terms
plt.figure(figsize=(10, 5))
plt.bar(range(len(sparse_coeffs)), sparse_coeffs)
plt.xticks(range(len(sparse_coeffs)), sparse_labels, rotation=45, ha='right')
plt.title(f"Lasso: The {len(active_indices)} Remaining Active Coefficients")
plt.ylabel("Coefficient Magnitude")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# --- 1. USER INPUTS ---
csv_path = "/content/Normalized_Cut_07.csv"
json_path = "Sparse_coeffs.json"  # Pointing to the Lasso-generated file

# --- 2. LOAD FILES ---
print(f"Loading {json_path}...")
try:
    with open(json_path, 'r') as f:
        model_data = json.load(f)
except FileNotFoundError:
    print(f"Error: {json_path} not found. Please ensure you ran the Lasso script first.")
    exit()

meta = model_data['meta']
old_coeffs = np.array(model_data['coeffs'])
term_labels = meta['term_order']  # This is now just the ~20 surviving terms

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)
mask = np.isfinite(df['rho_norm']) & np.isfinite(df['theta_aligned_deg']) & np.isfinite(df['intensity'])
rho = df['rho_norm'].values[mask]
theta_rad = np.deg2rad(df['theta_aligned_deg'].values[mask])
Z = df['intensity'].values[mask]

# --- 3. DYNAMICALLY BUILD SPARSE MATRIX ---
# We read the label (e.g., "k2_sin_n4") and generate exactly that mathematical term
print(f"\nBuilding sparse design matrix with {len(term_labels)} terms...")
col_vectors = []

for label in term_labels:
    # Parse string like 'k2_sin_n4'
    parts = label.split('_')
    k = int(parts[0].replace('k', ''))
    func = parts[1] # 'cos' or 'sin'
    n = int(parts[2].replace('n', ''))

    if func == 'cos':
        col = (rho**n) * np.cos(k * theta_rad)
    elif func == 'sin':
        col = (rho**n) * np.sin(k * theta_rad)

    col_vectors.append(col)

A_sparse = np.column_stack(col_vectors)

# --- 4. SENSITIVITY ANALYSIS ---
n_iters = 100
noise_level = 0.05 * np.std(Z) # 5% noise

print("\n--- Diagnostics ---")
# If Lasso did its job, this Condition Number should be massively lower than 31,000!
print(f"Condition Number of Sparse Matrix: {np.linalg.cond(A_sparse):.2f}")
print(f"Running {n_iters} iterations with +/- {noise_level:.2f} intensity noise...\n")

mc_coeffs = []
for _ in range(n_iters):
    Z_noisy = Z + np.random.normal(0, noise_level, size=Z.shape)

    # Because the matrix is now sparse and well-conditioned, standard OLS is safe
    coeffs_noisy, _, _, _ = lstsq(A_sparse, Z_noisy, rcond=None)
    mc_coeffs.append(coeffs_noisy)

mc_coeffs = np.array(mc_coeffs)
coeff_variances = np.var(mc_coeffs, axis=0)

print("Top 5 Most Volatile Coefficients in Sparse Model:")
top_5_idx = np.argsort(coeff_variances)[-5:][::-1]
# Safe check in case there are fewer than 5 terms
num_to_print = min(5, len(term_labels))
for idx in top_5_idx[:num_to_print]:
    print(f"  {term_labels[idx]}: Variance = {coeff_variances[idx]:.2f}")

# --- 5. PLOT RESULTS ---
# Plot all surviving terms (or up to 20 if more survived)
num_to_plot = min(20, len(term_labels))
top_volatile_idx = np.argsort(coeff_variances)[-num_to_plot:]

plt.figure(figsize=(10, max(5, num_to_plot * 0.3))) # Scale height based on term count
plt.boxplot(mc_coeffs[:, top_volatile_idx], vert=False, positions=np.arange(num_to_plot))
plt.scatter(old_coeffs[top_volatile_idx], np.arange(num_to_plot), color='red', zorder=10, label='Original Sparse Coeffs')

plt.yticks(np.arange(num_to_plot), [term_labels[i] for i in top_volatile_idx])
plt.xlabel('Coefficient Value Range')
plt.title(f'Sensitivity Analysis: Sparse Model ({len(term_labels)} terms) with 5% Noise')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

