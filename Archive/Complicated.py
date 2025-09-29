import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from dataclasses import dataclass

"""
Program 4: SIR + CG-based detection & mapping

This script:
 1) Loads the forward-simulated pressure traces from Program 1 (output_data.npz)
    and, if present, the EIR-filtered traces from Program 2 (output_data_eir.npz).
 2) Applies the SPATIAL IMPULSE RESPONSE (SIR) for a rectangular receive aperture
    *within* a steer-and-sample operator, so SIR is direction-dependent per pixel.
 3) For each (x, y) in a user grid, solves a small least-squares line problem over depth z
    via Conjugate Gradient for Normal Equations (CGNR): min ||S * alpha - y||^2,
    where S collects per-depth steering vectors (with SIR) evaluated on the post-EIR data.
 4) Reports per-(x,y):
      - theta := max |alpha(z)| (detection score)
      - z* at which |alpha| is maximized (estimated depth)
      - rough sphere radius using N-wave width measured from post-filtered data
 5) Renders a 2D map over x∈[-15,15] mm and y∈[-15,15] mm where pixels are black if
    no detection; else brightness encodes estimated depth using the requested
    scale: −35 → gray, −5 → bright white.

Assumptions:
 - Array: 128 rx elements along +X, centered; pitch = 0.2 mm.
 - Rectangular element size: a (lateral, along X) × b (elevation, along Y), with defaults
   a = 0.2 mm, b = 3.0 mm (adjustable via CLI params near the bottom).
 - Array plane is Z=0, elements normal along +Z.

Lightweight CGNR is used per (x, y) line over Nz depth bins. Each CG is tiny and fast.
"""

# -----------------------------
# Utilities & dataclasses
# -----------------------------

def sinc(x: np.ndarray) -> np.ndarray:
    # normalized sinc: sin(x)/x, well-defined at 0
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(x[nz]) / x[nz]
    return out

@dataclass
class ArrayGeom:
    centers: np.ndarray   # (Nd, 3)
    xhat: np.ndarray      # (3,) local X along element width a
    yhat: np.ndarray      # (3,) local Y along element height b
    nhat: np.ndarray      # (3,) element normal
    a: float              # mm (width along xhat)
    b: float              # mm (height along yhat)
    c: float              # mm/us speed of sound

# -----------------------------
# Data loading
# -----------------------------

base = np.load("output_data.npz")
t_vals = base["t_vals"].astype(np.float32)
pressure = base["pressure_composit"].astype(np.float32)  # shape (Nd, Nt)
dt = float(base["dt"])  # us per sample
Nt = pressure.shape[1]
Nd = pressure.shape[0]

# Optional EIR-filtered (preferred per user request):
try:
    post = np.load("output_data_eir.npz")
    pressure_post = post["data_eir_final"].astype(np.float32)
    t_vals_post = post["t_vals"].astype(np.float32)
    dt_post = float(post["dt"])  # should equal dt
    if pressure_post.shape == pressure.shape:
        pressure = pressure_post
        t_vals = t_vals_post
        dt = dt_post
        Nt = pressure.shape[1]
    else:
        print("Warning: EIR file present but shape mismatch; using base pressure.")
except FileNotFoundError:
    print("Note: output_data_eir.npz not found; proceeding with pre-EIR pressure.")

# -----------------------------
# Geometry (duplicated in case not saved by Program 1)
# -----------------------------

# Default assumptions matching existing code
x_npoints_detector = Nd
x_spacing_detector = 0.2  # mm pitch

y_npoints_detector = 1

iX = np.arange(x_npoints_detector, dtype=np.float32)
X = (iX * x_spacing_detector).reshape(-1, 1)
Y = np.zeros_like(X)
Z = np.zeros_like(X)
centers = np.concatenate([X, Y, Z], axis=1)
centers = centers - centers.mean(axis=0, keepdims=True)

# Local bases: array along global X; height along global Y; normal +Z
xhat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
yhat = np.array([0.0, 1.0, 0.0], dtype=np.float32)
nhat = np.array([0.0, 0.0, 1.0], dtype=np.float32)

# Element size (user adjustable)
a_width_mm = 0.2        # mm, lateral (along xhat)
b_height_mm = 3.0       # mm, elevation (along yhat)  <-- make variable

# Speed of sound consistent with Program 1
c_mm_us = 1.5           # mm/us

geom = ArrayGeom(centers=centers, xhat=xhat, yhat=yhat, nhat=nhat,
                 a=a_width_mm, b=b_height_mm, c=c_mm_us)

# -----------------------------
# Depth grid & (x,y) mapping grid
# -----------------------------

z_min, z_max = -35.0, -5.0
z_res = (1.0 / (40.0)) * c_mm_us  # mirror Program 1 z-resolution ~ c/sample_rate
z_vals = np.arange(z_min, z_max + 1e-6, z_res, dtype=np.float32)
Nz = z_vals.size

# User map: x,y in [-15, 15] mm; choose stride to keep runtime modest
x_map = np.linspace(-15.0, 15.0, 61, dtype=np.float32)  # 0.5 mm grid
y_map = np.linspace(-15.0, 15.0, 61, dtype=np.float32)
Nxmap, Nymap = x_map.size, y_map.size

# Output buffers
THETA = np.zeros((Nxmap, Nymap), dtype=np.float32)
ZSTAR = np.full((Nxmap, Nymap), np.nan, dtype=np.float32)
RADIUS = np.full((Nxmap, Nymap), np.nan, dtype=np.float32)

# -----------------------------
# Steering function with rectangular SIR (far-field), per voxel
# -----------------------------

def sir_weight_per_voxel(geom: ArrayGeom, vox: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute complex SIR weights (Nd, F) for a single voxel at position vox=(x,y,z).
    Far-field rectangular piston: (a*b)/(2*pi*r) * exp(-i 2pi f r/c) * sinc(pi f a X/(c r)) * sinc(pi f b Y/(c r))
    where (X,Y) are components of rvec in element-local coordinates.
    """
    d = vox[None, :] - geom.centers  # (Nd,3)
    r = np.linalg.norm(d, axis=1)    # (Nd,)
    # local coords
    Xloc = d @ geom.xhat
    Yloc = d @ geom.yhat

    # Avoid division by zero for on-top voxels
    r = np.maximum(r, 1e-6)

    # frequency grid and broadcast
    f = f.reshape(1, -1)                  # (1,F)
    r_b = r.reshape(-1, 1)                # (Nd,1)
    Xb = Xloc.reshape(-1, 1)
    Yb = Yloc.reshape(-1, 1)

    phase = np.exp(-1j * 2.0 * np.pi * f * (r_b / geom.c))
    coef  = (geom.a * geom.b) / (2.0 * np.pi * r_b)

    argX = np.pi * f * (geom.a * Xb) / (geom.c * r_b)
    argY = np.pi * f * (geom.b * Yb) / (geom.c * r_b)

    wx = sinc(argX)
    wy = sinc(argY)

    return coef * phase * wx * wy        # (Nd,F)

# -----------------------------
# Frequency grid for sampling the traces
# -----------------------------

F = fft.fftfreq(Nt, d=dt)  # cycles/us
F = F.astype(np.float32)

# -----------------------------
# Helper: build S matrix (Nd x Nz) at a given (x,y)
# -----------------------------

def build_S_at_xy(x: float, y: float) -> np.ndarray:
    """For a fixed (x,y), form S where S[:,k] is the complex steering value at depth z[k]
    evaluated at the *dominant* arrival time sample for each detector. We do this in freq
    domain to include direction-dependent SIR, then take IFFT back to time and sample.
    """
    S_cols = []
    for z in z_vals:
        vox = np.array([x, y, z], dtype=np.float32)
        Hs = sir_weight_per_voxel(geom, vox, F)  # (Nd, F)
        # Convert to time-domain impulse response per detector (Nd, Nt)
        hs_t = fft.ifft(Hs, axis=1).real
        # For each detector, find expected arrival index (nearest) and pick hs_t value at 0 lag
        # Simpler: use hs_t[:,0] as matched-filter weight (peak at t≈r/c);
        # alternatively, roll by predicted lag. We'll roll to align peaks.
        r = np.linalg.norm(vox[None,:] - geom.centers, axis=1)   # (Nd,)
        tau = r / geom.c  # us
        idx = np.clip(np.round(tau / dt).astype(int), 0, Nt-1)
        # Gather column by sampling hs_t at those indices per detector
        col = hs_t[np.arange(Nd), (-idx) % Nt]  # approximate alignment via circular shift
        S_cols.append(col.astype(np.float32))
    S = np.stack(S_cols, axis=1)  # (Nd, Nz)
    return S

# -----------------------------
# CGNR solver (very small systems)
# -----------------------------

def cgnr(A: np.ndarray, b: np.ndarray, iters: int = 15, lam: float = 0.0) -> np.ndarray:
    """Solve min ||A x - b||_2^2 + lam||x||_2^2 via CGNR."""
    # Work with complex internally (A is real here after sampling, but keep generic)
    A = A.astype(np.float32)
    b = b.astype(np.float32)
    if lam > 0:
        # augment normal equations: (A^T A + lam I) x = A^T b
        def AtA(v):
            return A.T @ (A @ v) + lam * v
        rhs = A.T @ b
    else:
        def AtA(v):
            return A.T @ (A @ v)
        rhs = A.T @ b

    x = np.zeros(A.shape[1], dtype=np.float32)
    r = rhs.copy()
    p = r.copy()
    rs_old = np.dot(r, r)
    for _ in range(iters):
        Ap = AtA(p)
        alpha = rs_old / (np.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if rs_new < 1e-10:
            break
        p = r + (rs_new/rs_old) * p
        rs_old = rs_new
    return x

# -----------------------------
# Detection loop over (x,y)
# -----------------------------

# For speed, precompute a compressed observation per detector by taking the maximum
# over time magnitude (envelope-like). Alternative: small window around predicted lags.
obs_per_det = np.max(np.abs(pressure), axis=1)  # (Nd,)

for ix, x in enumerate(x_map):
    for iy, y in enumerate(y_map):
        # Build steering for this (x,y)
        S = build_S_at_xy(float(x), float(y))        # (Nd, Nz)
        # Solve small LSQ for alpha(z)
        alpha = cgnr(S, obs_per_det, iters=20, lam=1e-3)  # mild Tikhonov
        # Detection score and depth estimate
        k = int(np.argmax(np.abs(alpha)))
        theta = float(np.abs(alpha[k]))

        THETA[ix, iy] = theta
        ZSTAR[ix, iy] = float(z_vals[k])

        # --- Rough radius estimate from N-wave width on nearest detector ---
        # Pick detector closest in range for (x,y,z*) and measure temporal width
        vox = np.array([x, y, ZSTAR[ix, iy]], dtype=np.float32)
        dists = np.linalg.norm(geom.centers - vox[None,:], axis=1)
        dmin_idx = int(np.argmin(dists))
        tau = dists[dmin_idx] / geom.c
        it0 = int(np.clip(np.round(tau/dt) - 10, 0, Nt-1))
        it1 = int(np.clip(np.round(tau/dt) + 10, 0, Nt-1))
        trace = pressure[dmin_idx, it0:it1]
        # width via threshold at 20% of local max
        if trace.size > 4:
            thr = 0.2 * np.max(np.abs(trace))
            above = np.where(np.abs(trace) >= thr)[0]
            if above.size >= 2:
                T_width = (above[-1] - above[0]) * dt  # us
                R_est = 0.5 * geom.c * T_width        # mm (R ≈ c*Δt/2)
                RADIUS[ix, iy] = float(R_est)

print("Done detection.")

# -----------------------------
# Render requested 2D map
# -----------------------------

# Detection threshold: normalize THETA to [0,1] by its 95th percentile
thr_ref = np.percentile(THETA, 95.0) + 1e-12
MASK = (THETA / thr_ref) >= 0.5

# Brightness mapping: z ∈ [-35, -5] → grayscale in [gray, white]
#   - no sphere: black
#   - z = -35: gray (0.5)
#   - z = -5 : white (1.0)

brightness = np.zeros_like(THETA, dtype=np.float32)
valid = np.isfinite(ZSTAR) & MASK

# map z to [0.5, 1.0]
z_clamped = np.clip(ZSTAR[valid], -35.0, -5.0)
bright_vals = 0.5 + 0.5 * (z_clamped + 35.0) / (30.0)  # -35→0.5, -5→1.0
brightness[valid] = bright_vals

# Plot
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
im = ax.imshow(brightness.T, origin='lower',
               extent=[x_map.min(), x_map.max(), y_map.min(), y_map.max()],
               vmin=0.0, vmax=1.0, cmap='gray')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_title('Detected spheres (depth encoded: −35 gray → −5 white)')
plt.colorbar(im, ax=ax, label='Brightness')
plt.tight_layout()
plt.show()

# Save outputs for downstream use
np.savez(
    "output_data_sir_cg.npz",
    x_map=x_map, y_map=y_map,
    THETA=THETA, ZSTAR=ZSTAR, RADIUS=RADIUS,
    dt=dt, z_vals=z_vals,
    a_mm=a_width_mm, b_mm=b_height_mm,
)
print("Saved: output_data_sir_cg.npz")
