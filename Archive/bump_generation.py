import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

start = time.time()
plot = True

# -----------------------------
# Config
# -----------------------------
np.random.seed(42)  # reproducibility

uniform = 1
bump_func = 1   # <--- TOGGLE bump function here

B  = 1.0
Cp = 1.0
c  = 1.5  # mm/us

t_min = 0.0
t_max = 40
sample_rate = 40.0    # samples per us
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
        spheres[i] = np.array([0, 0+np.sqrt(i)*3, -10.0 - 5.0 * i, 1.0], dtype=np.float32)
else:
    spheres = make_spheres(n_spheres=6)

# -----------------------------
# Detector grid
# -----------------------------
x_npoints_detector, y_npoints_detector = 128, 1
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
# PRESSURE SYNTHESIS
# -----------------------------
D = np.linalg.norm(detectors[:, None, :] - spheres[None, :, :3], axis=2).astype(np.float32)  # (Nd, Ns)
R = spheres[None, :, 3].astype(np.float32)                                                   # (1, Ns)
K = (B * (c**2)) / (2.0 * Cp) / D                                                            # (Nd, Ns)

if bump_func:
    # --- N-wave × bump method ---
    ct = (c * t_vals)[None, :].astype(np.float32)  # (1, Nt)
    pressure_composit = np.zeros((Nd, Nt), dtype=np.float32)

    for s in range(Ns):
        D_s = D[:, s][:, None]            # (Nd, 1)
        R_s = float(R[0, s])
        K_s = K[:, s][:, None]

        pN = (K_s * (D_s - ct)).astype(np.float32)  # base N-wave

        tau  = (ct - D_s) / R_s
        bump = np.zeros_like(tau, dtype=np.float32)
        inside = (np.abs(tau) < 1.0)
        tau_in = tau[inside]
        bump[inside] = np.exp(-1.0 / (1.0 - (tau_in * tau_in))).astype(np.float32) * np.e

        pressure_composit += (pN * bump).astype(np.float32)

else:
    # --- Original prefix-sum linear N-wave method ---
    t0 = (D - R) / c
    t1 = (D + R) / c

    i0 = np.maximum(0, np.ceil((t0 - t_vals[0]) / dt)).astype(np.int32)
    i1 = np.minimum(Nt, np.floor((t1 - t_vals[0]) / dt + 1.0)).astype(np.int32)

    A   = (K * D).astype(np.float32)   # (Nd, Ns)
    B_  = (K * c).astype(np.float32)

    addA = np.zeros((Nd, Nt + 1), dtype=np.float32)
    addB = np.zeros((Nd, Nt + 1), dtype=np.float32)

    rows = np.repeat(np.arange(Nd, dtype=np.int32)[:, None], Ns, axis=1)

    np.add.at(addA, (rows, i0),  A)
    np.add.at(addB, (rows, i0),  B_)
    np.add.at(addA, (rows, i1), -A)
    np.add.at(addB, (rows, i1), -B_)

    A_run = np.cumsum(addA, axis=1)[:, :Nt]
    B_run = np.cumsum(addB, axis=1)[:, :Nt]

    pressure_composit = (A_run - B_run * t_vals[None, :]).astype(np.float32)

# -----------------------------
# DAS back-projection
# -----------------------------
das_x_values = detectors[:, 0].astype(np.float32)
z_resolution = (1.0 / sample_rate) * c
das_z_values = np.arange(-35.0, 0.0, z_resolution, dtype=np.float32)

Xpix, Zpix = np.meshgrid(das_x_values, das_z_values, indexing='ij')
Nx, Nz = Xpix.shape
Npix = Nx * Nz

pixels = np.column_stack((
    Xpix.ravel(),
    np.zeros(Npix, dtype=np.float32),
    Zpix.ravel()
)).astype(np.float32)

diff = pixels[:, None, :] - detectors[None, :, :]
dist = np.linalg.norm(diff, axis=2).astype(np.float32)

f = (dist / c - t_min) / dt
f = np.clip(f, 0.0, Nt - 1.000001).astype(np.float32)

i0 = np.floor(f).astype(np.int32)
i1 = i0 + 1
w  = (f - i0).astype(np.float32)

det_idx = np.broadcast_to(np.arange(Nd, dtype=np.int32)[None, :], (Npix, Nd))

P0 = pressure_composit[det_idx, i0]
P1 = pressure_composit[det_idx, i1]

vals = (1.0 - w) * P0 + w * P1
das_flat = vals.sum(axis=1, dtype=np.float32)
das_img  = das_flat.reshape(Nx, Nz)

end = time.time()
print(f"Execution time: {end - start:.4f} seconds")

# -----------------------------
# Visualization
# -----------------------------
depth_vals = (c * t_vals).astype(np.float32)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

norm = mcolors.TwoSlopeNorm(vmin=pressure_composit.min(),
                            vcenter=0,
                            vmax=pressure_composit.max())

im0 = axes[0].imshow(
    pressure_composit.T,
    extent=[0, pressure_composit.shape[0], depth_vals.max(), depth_vals.min()],
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
    np.flipud(das_img.T),
    extent=[das_x_values.min(), das_x_values.max(), -35, 0],
    aspect='auto',
    cmap='seismic',
    norm=mcolors.TwoSlopeNorm(vmin=das_img.min(),
                              vcenter=0,
                              vmax=das_img.max()),
    origin='upper'
)
axes[1].set_xlabel("x (mm)")
axes[1].set_ylabel("Depth (mm)")
axes[1].set_title("DAS image")
fig.colorbar(im1, ax=axes[1], label="DAS amplitude")

if plot:
    plt.show()
    plt.plot( t_vals , pressure_composit[64])
    plt.show()

# -----------------------------
# Save outputs
# -----------------------------
outname = "output_data_bump.npz" if bump_func else "output_data.npz"
np.savez(
    outname,
    t_vals=t_vals,
    pressure_composit=pressure_composit,
    das_img=das_img,
    dt=dt,
    detectors=detectors,
    spheres=spheres,
    c=c,
    B=B,
    Cp=Cp
)
print(f"Saved: {outname}")
