import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- NEW: import CuPy and pick it as the array module
try:
    import cupy as cp
    xp = cp
    GPU = True
except Exception:
    xp = np
    GPU = False

start = time.time()

# -----------------------------
# Config
# -----------------------------
np.random.seed(42)  # keep NumPy seed for any CPU-side RNG use

uniform = False

B  = 1.0
Cp = 1.0
c  = 1.5  # mm/us

t_min = 0.0
t_max = 400
sample_rate = 60.0    # Hz, times per us  (== samples/us)
dt = 1.0 / sample_rate

# keep as NumPy for convenience, then move to GPU once as needed
detectors_center = np.array([0.0, 0.0 , 0.0], dtype=np.float32)

# -----------------------------
# Time axis (GPU)
# -----------------------------
t_vals = xp.arange(t_min, t_max, dt, dtype=xp.float32)
t_npoints = t_vals.size

# -----------------------------
# Spheres (build on CPU, move to GPU)
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
    spheres_np = np.zeros((5, 4), dtype=np.float32)
    for i in range(5):
        spheres_np[i] = np.array([0.0, 0.0, -30.0 + 5.0 * i, 1.0], dtype=np.float32)
else:
    spheres_np = make_spheres(n_spheres=6)

spheres = xp.asarray(spheres_np)

# -----------------------------
# Detector grid (vectorized) on GPU
# -----------------------------
x_npoints_detector, y_npoints_detector = 1000, 1   # number of points
x_spacing_detector, y_spacing_detector = 0.2, 0.2 # mm

ix = xp.arange(x_npoints_detector, dtype=xp.float32)
jy = xp.arange(y_npoints_detector, dtype=xp.float32)
X, Y = xp.meshgrid(ix * x_spacing_detector, jy * y_spacing_detector, indexing='ij')

detectors = xp.column_stack((
    X.ravel(),
    Y.ravel(),
    xp.zeros(X.size, dtype=xp.float32)
)).astype(xp.float32)

detectors -= detectors.mean(axis=0).astype(xp.float32)
detectors += xp.asarray(detectors_center, dtype=xp.float32)

Nd = detectors.shape[0]
Ns = spheres.shape[0]
Nt = t_npoints

# -----------------------------
# PRESSURE SYNTHESIS (GPU)
# -----------------------------
# Distances D (Nd, Ns)
# cupy.linalg.norm mirrors numpy
D = xp.linalg.norm(detectors[:, None, :] - spheres[None, :, :3], axis=2).astype(xp.float32)  # (Nd, Ns)
R = spheres[None, :, 3].astype(xp.float32)                                                   # (Nd, Ns)
K = (B * (c**2)) / (2.0 * Cp) / D                                                            # (Nd, Ns)

# Active time window per (det, sphere)
t0 = (D - R) / c
t1 = (D + R) / c

# Convert to [i0, i1] indices in [0..Nt]
i0 = xp.maximum(0, xp.ceil((t0 - t_vals[0]) / dt)).astype(xp.int32)             # (Nd, Ns)
i1 = xp.minimum(Nt, xp.floor((t1 - t_vals[0]) / dt + 1.0)).astype(xp.int32)     # (Nd, Ns)

# Linear coefficients over active slice:
# p(t) = A - B_* t, where A = K*D and B_* = K*c
A   = (K * D).astype(xp.float32)   # (Nd, Ns)
B_  = (K * c).astype(xp.float32)   # (Nd, Ns)

# Build difference arrays for prefix sums (Nd, Nt+1)
addA = xp.zeros((Nd, Nt + 1), dtype=xp.float32)
addB = xp.zeros((Nd, Nt + 1), dtype=xp.float32)

rows = xp.repeat(xp.arange(Nd, dtype=xp.int32)[:, None], Ns, axis=1)  # (Nd, Ns)

# Scatter-add starts and ends (vectorized)
# CuPy supports add.at
xp.add.at(addA, (rows, i0),  A)
xp.add.at(addB, (rows, i0),  B_)
xp.add.at(addA, (rows, i1), -A)
xp.add.at(addB, (rows, i1), -B_)

# Prefix-sum along time
A_run = xp.cumsum(addA, axis=1)[:, :Nt]  # (Nd, Nt)
B_run = xp.cumsum(addB, axis=1)[:, :Nt]  # (Nd, Nt)

# Evaluate p(t) = A_run - B_run * t
pressure_composit = (A_run - B_run * t_vals[None, :]).astype(xp.float32)  # (Nd, Nt)

# -----------------------------
# DAS (GPU)
# -----------------------------
das_x_values = detectors[:, 0].astype(xp.float32)  # lateral pixels
z_resolution = (1.0 / sample_rate) * c            # mm
das_z_values = xp.arange(-35.0, 0.0, z_resolution, dtype=xp.float32)

# Pixel grid (Nx, Nz)
Xpix, Zpix = xp.meshgrid(das_x_values, das_z_values, indexing='ij')
Nx, Nz = Xpix.shape
Npix = Nx * Nz

pixels = xp.column_stack((
    Xpix.ravel(),
    xp.zeros(Npix, dtype=xp.float32),
    Zpix.ravel()
)).astype(xp.float32)  # (Npix, 3)

# Distances from all pixels to all detectors: (Npix, Nd)
diff = pixels[:, None, :] - detectors[None, :, :]          # (Npix, Nd, 3)
dist = xp.linalg.norm(diff, axis=2).astype(xp.float32)     # (Npix, Nd)

# Convert distances to fractional sample indices in pressure traces
f = (dist / c - t_min) / dt
f = xp.clip(f, 0.0, Nt - 1.000001).astype(xp.float32)

i0 = xp.floor(f).astype(xp.int32)   # (Npix, Nd)
i1 = i0 + 1
w  = (f - i0).astype(xp.float32)

# Build detector row index of shape (Npix, Nd)
det_idx = xp.broadcast_to(xp.arange(Nd, dtype=xp.int32)[None, :], (Npix, Nd))

P0 = pressure_composit[det_idx, i0]   # (Npix, Nd)
P1 = pressure_composit[det_idx, i1]   # (Npix, Nd)

vals = (1.0 - w) * P0 + w * P1        # (Npix, Nd)
das_flat = vals.sum(axis=1, dtype=xp.float32)  # (Npix,)
das_img  = das_flat.reshape(Nx, Nz)            # (Nx, Nz)

# --- ensure all GPU kernels finished before timing ---
if GPU:
    cp.cuda.Stream.null.synchronize()

end = time.time()
print(f"Execution time (GPU={GPU}): {end - start:.4f} seconds")

# -----------------------------
# Visualization (CPU; convert arrays)
# -----------------------------
depth_vals = (c * (cp.asnumpy(t_vals) if GPU else t_vals)).astype(np.float32)

if GPU:
    pressure_cpu = cp.asnumpy(pressure_composit)
    das_img_cpu  = cp.asnumpy(das_img)
    das_x_cpu    = cp.asnumpy(das_x_values)
else:
    pressure_cpu = pressure_composit
    das_img_cpu  = das_img
    das_x_cpu    = das_x_values

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

norm = mcolors.TwoSlopeNorm(vmin=pressure_cpu.min(),
                            vcenter=0,
                            vmax=pressure_cpu.max())

im0 = axes[0].imshow(
    pressure_cpu.T,
    extent=[0, pressure_cpu.shape[0], depth_vals.max(), depth_vals.min()],
    aspect='auto',
    cmap='seismic',
    norm=norm,
    origin='upper'
)
axes[0].set_xlabel("Detector index")
axes[0].set_ylabel("Depth (mm)")
axes[0].set_title("Pressure traces (detector vs depth)")
fig.colorbar(im0, ax=axes[0], label="Pressure")

im1 = axes[1].imshow(
    np.flipud(das_img_cpu.T),
    extent=[das_x_cpu.min(), das_x_cpu.max(), -35, 0],
    aspect='auto',
    cmap='seismic',
    norm=mcolors.TwoSlopeNorm(vmin=das_img_cpu.min(), vcenter=0, vmax=das_img_cpu.max()),
    origin='upper'
)
axes[1].set_xlabel("x (mm)")
axes[1].set_ylabel("Depth (mm)")
axes[1].set_title("DAS image")
fig.colorbar(im1, ax=axes[1], label="DAS amplitude")

plt.show()
