import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import time
start = time.time()

plot = True
bump_func = True

# -----------------------------
# Config (same semantics as yours)
# -----------------------------
np.random.seed(42)

uniform = 1

B  = 1.0
Cp = 1.0
c  = 1.5  # mm/us

t_min = 0.0
t_max = 40
sample_rate = 40.0    # samples/us
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
# Detector grid (vectorized)
# -----------------------------
x_npoints_detector, y_npoints_detector = 128, 1
x_spacing_detector, y_spacing_detector = 0.2, 0.2  # mm

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

# =========================================================
# NEW: Sphere pressure response — multiplies the N-wave
# =========================================================
def sphere_pressure_response(D, spheres, t_vals, c, bump_func=bump_func):
    """
    Returns a weight in [0,1] per (detector, sphere, time).
    Default: 1 when |c*t - D| <= R (inside the spherical shell), else 0.

    D:  (Nd, Ns)   distances detector<->sphere center
    R:  (Ns,)      radii
    t_vals: (Nt,)  time samples
    c:  scalar     speed
    """
    R = spheres[:, 3].astype(np.float32) 
    ct = c * t_vals[None, None, :]                  # (1,1,Nt)
    r  = np.abs(ct - D[:, :, None])                 # (Nd,Ns,Nt) where r = |c t - D|
    inside = (r < R[None, :, None])
      
    
    output = inside.astype(np.float32)

    if bump_func:
        output = np.where(output,
                            np.exp( 1 / ( -1 + (np.abs(r-R[None, :, None]) / R[None, :, None]) ** 2 ) ),
                            0.0
                        ).astype(np.float32)

      

        x = range(len(r[63,0,:]))
        y = r[63,0,:]
        z = np.abs(y-1)
        y1 = output[63,0,:]
        plt.plot(x,y)
        plt.plot(x,z)
        plt.plot(x,y1)
        plt.show()
        ''' x = range(len(r[63,0,:]))
        y = np.abs(r-R[None, :, None])[63,0,:]
        
        y = np.exp( 1 / ( -1 - (y / 1) ** 2 ) )
        y = np.where(inside[63,0,:], y, 0.0)
        plt.plot(x,y)
        plt.show()'''
        
    
    return output



# -----------------------------
# PRESSURE SYNTHESIS (preserve N-wave shape, then gate/attenuate)
# -----------------------------
# Distances D (Nd, Ns) and radii R (Ns,)
D = np.linalg.norm(detectors[:, None, :] - spheres[None, :, :3], axis=2).astype(np.float32)  # (Nd,Ns)
R = spheres[:, 3].astype(np.float32)                                                         # (Ns,)

# N-wave coefficients per (det, sphere)
K = (B * (c**2)) / (2.0 * Cp) / D                                                            # (Nd,Ns)
A = (K * D)[:, :, None].astype(np.float32)           # (Nd,Ns,1)
B_ = (K * c)[:, :, None].astype(np.float32)          # (Nd,Ns,1)

t3 = t_vals[None, None, :]                           # (1,1,Nt)

# Raw N-wave (NOT masked) per (det, sphere, time)
p_N = (A - B_ * t3).astype(np.float32)               # (Nd,Ns,Nt)

# Multiply by response (1 inside, 0 outside by default)
resp = sphere_pressure_response(D, spheres, t_vals, c, bump_func = bump_func)     # (Nd,Ns,Nt)
p_mod = (p_N * resp).astype(np.float32)              # (Nd,Ns,Nt)

# Sum contributions from all spheres -> (Nd, Nt)
pressure_composit = p_mod.sum(axis=1, dtype=np.float32)  # (Nd,Nt)

# -----------------------------
# DAS (vectorized interpolation and sum over detectors)
# -----------------------------
das_x_values = detectors[:, 0].astype(np.float32)
z_resolution = (1.0 / sample_rate) * c            # mm
das_z_values = np.arange(-35.0, 0.0, z_resolution, dtype=np.float32)

Xpix, Zpix = np.meshgrid(das_x_values, das_z_values, indexing='ij')
Nx, Nz = Xpix.shape
Npix = Nx * Nz

pixels = np.column_stack((
    Xpix.ravel(),
    np.zeros(Npix, dtype=np.float32),
    Zpix.ravel()
)).astype(np.float32)

diff = pixels[:, None, :] - detectors[None, :, :]          # (Npix, Nd, 3)
dist = np.linalg.norm(diff, axis=2).astype(np.float32)     # (Npix, Nd)

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
    norm=mcolors.TwoSlopeNorm(vmin=das_img.min(), vcenter=0, vmax=das_img.max()),
    origin='upper'
)
axes[1].set_xlabel("x (mm)")
axes[1].set_ylabel("Depth (mm)")
axes[1].set_title("DAS image")
fig.colorbar(im1, ax=axes[1], label="DAS amplitude")

if plot:
    plt.show()
    plt.plot(t_vals,pressure_composit[63])
    plt.show()

if bump_func:
    np.savez(
    "output_data_bump.npz",
    t_vals=t_vals,
    pressure_composit=pressure_composit,
    das_img=das_img,
    dt=dt,
    detectors=detectors,
    spheres=spheres,
    c=c,
    B=B,
    Cp=Cp,
    )
    
    print("Saved: output_data_bump.npz")

else:
    np.savez(
        "output_data.npz",
        t_vals=t_vals,
        pressure_composit=pressure_composit,
        das_img=das_img,
        dt=dt,
        detectors=detectors,
        spheres=spheres,
        c=c,
        B=B,
        Cp=Cp,
        )
    print("Saved: output_data.npz")


