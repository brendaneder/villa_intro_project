import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import time
start = time.time()

plot = False 


# -----------------------------
# Config (same semantics as yours)
# -----------------------------
np.random.seed(42)  # used only if you keep legacy RNG elsewhere

uniform = True

B  = 1.0
Cp = 1.0
c  = 1.5  # mm/us

t_min = 0.0
t_max = 40
sample_rate = 40.0    # Hz, times per us  (== samples/us)
dt = 1.0 / sample_rate

detectors_center = np.array([0.0, 0.0 , 0.0], dtype=np.float32)

# -----------------------------
# Time axis
# -----------------------------
t_vals = np.arange(t_min, t_max, dt, dtype=np.float32)
t_npoints = t_vals.size

# -----------------------------
# Spheres
# -----------------------------
def make_spheres(n_spheres=1,
                 x_bounds=(-15.0, 15.0),
                 y_bounds=(-15.0, 15.0),
                 z_bounds=(-35.0,  -5.0),
                 r_bounds=(  0.5,   2.0)):
    rng = np.random.default_rng(42)
    x = rng.uniform(*x_bounds, size=n_spheres).astype(np.float32)
    y = rng.uniform(*y_bounds, size=n_spheres).astype(np.float32)
    z = rng.uniform(*z_bounds, size=n_spheres).astype(np.float32)
    r = rng.uniform(*r_bounds, size=n_spheres).astype(np.float32)
    return np.column_stack([x, y, z, r]).astype(np.float32)

if uniform:
    spheres = np.zeros((5, 4), dtype=np.float32)
    for i in range(5):
        spheres[i] = np.array([0.0, 0.0, -30.0 + 5.0 * i, 1.0], dtype=np.float32)
else:
    spheres = make_spheres(n_spheres=6)

# -----------------------------
# Detector grid (vectorized)
# -----------------------------
x_npoints_detector, y_npoints_detector = 128, 1   # number of points
x_spacing_detector, y_spacing_detector = 0.2, 0.2 # mm

ix = np.arange(x_npoints_detector, dtype=np.float32)
jy = np.arange(y_npoints_detector, dtype=np.float32)
X, Y = np.meshgrid(ix * x_spacing_detector, jy * y_spacing_detector, indexing='ij')

detectors = np.column_stack((
    X.ravel(),
    Y.ravel(),
    np.zeros(X.size, dtype=np.float32)
)).astype(np.float32)

detectors -= detectors.mean(axis=0).astype(np.float32)
detectors += detectors_center.astype(np.float32)

Nd = detectors.shape[0]
Ns = spheres.shape[0]
Nt = t_npoints


# -----------------------------
# PRESSURE SYNTHESIS (all detectors × spheres at once)
# p(t) = K * (d - c t) on interval |c t - d| <= R; 0 otherwise
# We assemble piecewise-linear segments via two prefix sums.
# -----------------------------
# Distances D (Nd, Ns)

D = np.linalg.norm(detectors[:, None, :] - spheres[None, :, :3], axis=2).astype(np.float32)  # (Nd, Ns)
R = spheres[None, :, 3].astype(np.float32)                                                   # (1, Ns) -> (Nd, Ns)
K = (B * (c**2)) / (2.0 * Cp) / D                                                            # (Nd, Ns)

# Active time window per (det, sphere)
t0 = (D - R) / c
t1 = (D + R) / c

# Convert to [i0, i1] indices in [0..Nt]; allow i1==Nt by using Nt+1 diff arrays
i0 = np.maximum(0, np.ceil((t0 - t_vals[0]) / dt)).astype(np.int32)             # (Nd, Ns)
i1 = np.minimum(Nt, np.floor((t1 - t_vals[0]) / dt + 1.0)).astype(np.int32)     # (Nd, Ns)

# Linear coefficients over active slice:
# p(t) = A - B_* t, where A = K*D and B_* = K*c
A   = (K * D).astype(np.float32)   # (Nd, Ns)
B_  = (K * c).astype(np.float32)   # (Nd, Ns)

# Build difference arrays for prefix sums (Nd, Nt+1)
addA = np.zeros((Nd, Nt + 1), dtype=np.float32)
addB = np.zeros((Nd, Nt + 1), dtype=np.float32)

rows = np.repeat(np.arange(Nd, dtype=np.int32)[:, None], Ns, axis=1)  # (Nd, Ns)

# Scatter-add starts and ends (vectorized)
np.add.at(addA, (rows, i0),  A)
np.add.at(addB, (rows, i0),  B_)
np.add.at(addA, (rows, i1), -A)
np.add.at(addB, (rows, i1), -B_)

# Prefix-sum along time
A_run = np.cumsum(addA, axis=1)[:, :Nt]  # (Nd, Nt)
B_run = np.cumsum(addB, axis=1)[:, :Nt]  # (Nd, Nt)

# Evaluate p(t) = A_run - B_run * t
pressure_composit = (A_run - B_run * t_vals[None, :]).astype(np.float32)  # (Nd, Nt)

# -----------------------------
# DAS (vectorized interpolation and sum over detectors)
# -----------------------------
das_x_values = detectors[:, 0].astype(np.float32)  # reuse detector x-positions as lateral pixels
z_resolution = (1.0 / sample_rate) * c            # mm
das_z_values = np.arange(-35.0, 0.0, z_resolution, dtype=np.float32)

# Pixel grid (Nx, Nz)
Xpix, Zpix = np.meshgrid(das_x_values, das_z_values, indexing='ij')
Nx, Nz = Xpix.shape
Npix = Nx * Nz

pixels = np.column_stack((
    Xpix.ravel(),
    np.zeros(Npix, dtype=np.float32),
    Zpix.ravel()
)).astype(np.float32)  # (Npix, 3)

# Distances from all pixels to all detectors: (Npix, Nd)
diff = pixels[:, None, :] - detectors[None, :, :]          # (Npix, Nd, 3)
dist = np.linalg.norm(diff, axis=2).astype(np.float32)     # (Npix, Nd)

# Convert distances to fractional sample indices in pressure traces
f = (dist / c - t_min) / dt
# Keep i1 in range by clipping slightly below Nt
f = np.clip(f, 0.0, Nt - 1.000001).astype(np.float32)

i0 = np.floor(f).astype(np.int32)   # (Npix, Nd)
i1 = i0 + 1
w  = (f - i0).astype(np.float32)

# Gather samples from pressure_composit: rows=detector, cols=time
# Build detector row index of shape (Npix, Nd)
det_idx = np.broadcast_to(np.arange(Nd, dtype=np.int32)[None, :], (Npix, Nd))



P0 = pressure_composit[det_idx, i0]   # (Npix, Nd)
P1 = pressure_composit[det_idx, i1]   # (Npix, Nd)

vals = (1.0 - w) * P0 + w * P1        # (Npix, Nd)
das_flat = vals.sum(axis=1, dtype=np.float32)  # (Npix,)
das_img  = das_flat.reshape(Nx, Nz)            # (Nx, Nz)



end = time.time()
print(f"Execution time: {end - start:.4f} seconds")



# -----------------------------
# Visualization (with requested fixes)
# -----------------------------
depth_vals = (c * t_vals).astype(np.float32)   # mm

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# --- Shared colormap normalization ---
# Force white at 0, blue <0, red >0
norm = mcolors.TwoSlopeNorm(vmin=pressure_composit.min(),
                            vcenter=0,
                            vmax=pressure_composit.max())

# --- Left: pressure traces ---
im0 = axes[0].imshow(
    pressure_composit.T,
    extent=[0, pressure_composit.shape[0], depth_vals.max(), depth_vals.min()],
    aspect='auto',
    cmap='seismic',
    norm=norm,              # <-- enforce white at 0
    origin='upper'
)
axes[0].set_xlabel("Detector index")
axes[0].set_ylabel("Depth (mm)")
axes[0].set_title("Pressure traces (detector vs depth)")
fig.colorbar(im0, ax=axes[0], label="Pressure")

# --- Right: DAS image ---
im1 = axes[1].imshow(
    np.flipud(das_img.T),   # flipped vertically
    extent=[das_x_values.min(), das_x_values.max(),
            -35, 0],        # axis labels flipped
    aspect='auto',
    cmap='seismic',
    norm=mcolors.TwoSlopeNorm(vmin=das_img.min(),
                              vcenter=0,
                              vmax=das_img.max()),  # <-- white at 0
    origin='upper'
)
axes[1].set_xlabel("x (mm)")
axes[1].set_ylabel("Depth (mm)")
axes[1].set_title("DAS image")
fig.colorbar(im1, ax=axes[1], label="DAS amplitude")


if plot:
    plt.show()


np.savez("output_data.npz", t_vals=t_vals, pressure_composit=pressure_composit, das_img=das_img, dt=dt)
print("Saved: output_data.npz")
