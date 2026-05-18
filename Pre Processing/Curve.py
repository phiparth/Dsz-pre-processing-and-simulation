import json
import numpy as np
import plotly.graph_objects as go

# 1. Load the JSON data
file_path = '/content/cell_0_ellipse_coeffs.json'
with open(file_path, 'r') as f:
    data = json.load(f)

meta = data['meta']
coeffs = data['coeffs']
term_order = meta['term_order']

a = meta['a']
b = meta['b']
theta0_rad = np.deg2rad(meta.get('theta0_deg', 0.0))

# 2. Create a grid for normalized radius rho (0 to 1) and theta (0 to 2*pi)
rho = np.linspace(0, 1, 100)
theta = np.linspace(0, 2 * np.pi, 100)
RHO, THETA = np.meshgrid(rho, theta)

# 3. Map to X and Y coordinates accounting for the ellipse parameters (a, b) and rotation
X = RHO * a * np.cos(THETA + theta0_rad)
Y = RHO * b * np.sin(THETA + theta0_rad)

# 4. Calculate Intensity (Z) based on the provided equation format
Z = np.zeros_like(RHO)

for term, coeff in zip(term_order, coeffs):
    parts = term.split('_')
    k = int(parts[0][1:])
    trig_func = parts[1]
    n = int(parts[2][1:])

    if trig_func == 'cos':
        Z += coeff * (RHO**n) * np.cos(k * THETA)
    elif trig_func == 'sin':
        Z += coeff * (RHO**n) * np.sin(k * THETA)

# Define global min and max of Z to unify the colorscale across all parts
z_min = np.min(Z)
z_max = np.max(Z)

# 5. Build the Solid Shape (Top, Base, and Walls)
# A: The Top Surface
top_surface = go.Surface(
    z=Z, x=X, y=Y,
    colorscale='Viridis',
    cmin=z_min, cmax=z_max,
    showscale=True
)

# B: The Bottom Base (Flat plane at z_min)
Z_bottom = np.full_like(Z, z_min)
bottom_surface = go.Surface(
    z=Z_bottom, x=X, y=Y,
    colorscale='Viridis',
    cmin=z_min, cmax=z_max,
    showscale=False, hoverinfo='skip'
)

# C: The Side Walls (Connecting outer edge rho=1 down to z_min)
X_edge = X[:, -1]
Y_edge = Y[:, -1]
Z_edge_top = Z[:, -1]
Z_edge_bottom = np.full_like(Z_edge_top, z_min)

# Create a mini 2xN grid connecting the bottom edge to the top edge
X_side = np.array([X_edge, X_edge])
Y_side = np.array([Y_edge, Y_edge])
Z_side = np.array([Z_edge_bottom, Z_edge_top])

side_surface = go.Surface(
    z=Z_side, x=X_side, y=Y_side,
    colorscale='Viridis',
    cmin=z_min, cmax=z_max,
    showscale=False, hoverinfo='skip'
)

# 6. Plot the fully enclosed solid using all 3 surfaces
fig = go.Figure(data=[top_surface, bottom_surface, side_surface])

# Determine the longest physical dimension to normalize the visual aspect ratio
max_dim = max(a, b)

fig.update_layout(
    title='3D Solid Intensity Graph (True Elliptical Proportion)',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Intensity (Z)',
        # DYNAMIC ASPECT RATIO:
        # Instead of hardcoding x=1, y=1, we scale the visual axes relative to `a` and `b`
        # This guarantees the base perfectly matches the physical ellipse size.
        aspectratio=dict(
            x = a / max_dim,
            y = b / max_dim,
            z = 0.6  # Tweak this up or down to make the vertical height visually pleasing
        )
    ),
    width=900,
    height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.show()

import json
import numpy as np
import plotly.graph_objects as go

# 1. Load the JSON data
file_path = '/content/Sparse_coeffs.json'
with open(file_path, 'r') as f:
    data = json.load(f)

meta = data['meta']
coeffs = data['coeffs']
term_order = meta['term_order']

a = meta['a']
b = meta['b']
theta0_rad = np.deg2rad(meta.get('theta0_deg', 0.0))

# 2. Create a grid for normalized radius rho (0 to 1) and theta (0 to 2*pi)
rho = np.linspace(0, 1, 100)
theta = np.linspace(0, 2 * np.pi, 100)
RHO, THETA = np.meshgrid(rho, theta)

# 3. Map to X and Y coordinates accounting for the ellipse parameters (a, b) and rotation
X = RHO * a * np.cos(THETA + theta0_rad)
Y = RHO * b * np.sin(THETA + theta0_rad)

# 4. Calculate Intensity (Z) based on the provided equation format
Z = np.zeros_like(RHO)

for term, coeff in zip(term_order, coeffs):
    parts = term.split('_')
    k = int(parts[0][1:])
    trig_func = parts[1]
    n = int(parts[2][1:])

    if trig_func == 'cos':
        Z += coeff * (RHO**n) * np.cos(k * THETA)
    elif trig_func == 'sin':
        Z += coeff * (RHO**n) * np.sin(k * THETA)

# Define global min and max of Z to unify the colorscale across all parts
z_min = np.min(Z)
z_max = np.max(Z)

# 5. Build the Solid Shape (Top, Base, and Walls)
# A: The Top Surface
top_surface = go.Surface(
    z=Z, x=X, y=Y,
    colorscale='Viridis',
    cmin=z_min, cmax=z_max,
    showscale=True
)

# B: The Bottom Base (Flat plane at z_min)
Z_bottom = np.full_like(Z, z_min)
bottom_surface = go.Surface(
    z=Z_bottom, x=X, y=Y,
    colorscale='Viridis',
    cmin=z_min, cmax=z_max,
    showscale=False, hoverinfo='skip'
)

# C: The Side Walls (Connecting outer edge rho=1 down to z_min)
X_edge = X[:, -1]
Y_edge = Y[:, -1]
Z_edge_top = Z[:, -1]
Z_edge_bottom = np.full_like(Z_edge_top, z_min)

# Create a mini 2xN grid connecting the bottom edge to the top edge
X_side = np.array([X_edge, X_edge])
Y_side = np.array([Y_edge, Y_edge])
Z_side = np.array([Z_edge_bottom, Z_edge_top])

side_surface = go.Surface(
    z=Z_side, x=X_side, y=Y_side,
    colorscale='Viridis',
    cmin=z_min, cmax=z_max,
    showscale=False, hoverinfo='skip'
)

# 6. Plot the fully enclosed solid using all 3 surfaces
fig = go.Figure(data=[top_surface, bottom_surface, side_surface])

# Determine the longest physical dimension to normalize the visual aspect ratio
max_dim = max(a, b)

fig.update_layout(
    title='3D Solid Intensity Graph (True Elliptical Proportion)',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Intensity (Z)',
        # DYNAMIC ASPECT RATIO:
        # Instead of hardcoding x=1, y=1, we scale the visual axes relative to `a` and `b`
        # This guarantees the base perfectly matches the physical ellipse size.
        aspectratio=dict(
            x = a / max_dim,
            y = b / max_dim,
            z = 0.6  # Tweak this up or down to make the vertical height visually pleasing
        )
    ),
    width=900,
    height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

fig.show()