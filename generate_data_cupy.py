import os, sys
from pathlib import Path
import time

# ---- Make sure Windows can find CUDA 13.0 DLLs before importing CuPy
import os, sys
CUDA_ROOT   = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
CUDA_BINX64 = os.path.join(CUDA_ROOT, "bin", "x64")

os.environ.setdefault("CUDA_PATH", CUDA_ROOT)

# For Python 3.8+, this is the most reliable way on Windows
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(CUDA_BINX64)

# Also prepend to PATH for any child processes (harmless if duplicated)
if os.path.isdir(CUDA_BINX64):
    os.environ["PATH"] = CUDA_BINX64 + ";" + os.environ.get("PATH", "")

import numpy as np
import numpy.fft as np_fft
import cupy as cp
import cupy.fft as cp_fft

GPU = True   # or detect automatically
xp = cp if GPU else np
xp_fft = cp_fft if GPU else np_fft


# ---------------------------
# Imports that rely on selected backend
# ---------------------------
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------
# Original parameters
# ---------------------------
np.random.seed(42)

i = 0
n_runs = 500

plt_trace = 1
plt_das = 1
bump_func = 1
uniform = 0
n_spheres = 1

B  = 1.0
Cp = 1.0
c  = 1.5  # mm/us

t_min = 0.0
t_max = 40
sample_rate = 40.0    # samples/us
dt = 1.0 / sample_rate

t_vals = np.arange(t_min, t_max, dt, dtype=np.float32)
t_npoints = t_vals.size

detectors_center = np.array([0.0, 0.0 , 0.0], dtype=np.float32)

def make_spheres(n_spheres=1,
                 x_bounds=(-12, 12),
                 y_bounds=(-12, 12),
                 z_bounds=(-32,  -8),
                 r_bounds=(  0.5,   2.0)):
    x = np.random.uniform(*x_bounds, size=n_spheres).astype(np.float32)
    y = np.random.uniform(*y_bounds, size=n_spheres).astype(np.float32)
    z = np.random.uniform(*z_bounds, size=n_spheres).astype(np.float32)
    r = np.random.uniform(*r_bounds, size=n_spheres).astype(np.float32)

    if n_spheres > 1:
        for i_ in range(n_spheres-1):
            for j_ in range(i_+1, n_spheres):
                dx = x[i_] - x[j_]
                dy = y[i_] - y[j_]
                dz = z[i_] - z[j_]
                true_distance = np.linalg.norm((dx, dy, dz))
                min_distance = r[i_] + r[j_]
                if true_distance <= min_distance:
                    print("Real distance {:.2f} < {:.2f} min distance".format(true_distance, min_distance))
                    # regenerate all (keeps original behavior)
                    return make_spheres(n_spheres, x_bounds, y_bounds, z_bounds, r_bounds)

    spheres = np.column_stack([x, y, z, r]).astype(np.float32)
    print(spheres)
    return spheres

# Detector grid (CPU build; moved to GPU later)
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
Nt = int(t_vals.size)

# -----------------------------
# Geometry & discretization
# -----------------------------
pitch_x_mm           = 0.2       # center-to-center spacing along X (mm)
aperture_width_x_mm  = pitch_x_mm
aperture_height_y_mm = 10
n_div_y              = 100
n_div_x              = 1

# -----------------------------
# EIR parameters
# -----------------------------
f0_MHz = 8.0
sigma  = np.sqrt(f0_MHz / np.log(2.0))

Npad  = 2 * Nt

# We'll build freqs on GPU later as xp array to avoid host-device thrash

def H_function(x, k = 9, a0 = .5, b0 = 9, xp_mod=None):
    # Uses xp module (cp or np) to match array device
    xp_mod = xp_mod or xp
    sig_a = xp_mod.float32(1) / ( xp_mod.float32(1) + xp_mod.exp( -k * (xp_mod.abs(x) - a0)))
    sig_b = xp_mod.float32(1) / ( xp_mod.float32(1) + xp_mod.exp( -k * (xp_mod.abs(x) - b0)))
    scale = xp_mod.abs( sig_a - sig_b )
    return scale.astype(xp_mod.float32)

# -----------------------------
# Aperture sub-rectangles (patches)
# -----------------------------
ax = float(aperture_width_x_mm)
ay = float(aperture_height_y_mm)

x_centers = (np.arange(n_div_x, dtype=np.float32) + 0.5) / n_div_x * ax - ax / 2.0  # (n_div_x,)
y_centers = (np.arange(n_div_y, dtype=np.float32) + 0.5) / n_div_y * ay - ay / 2.0  # (n_div_y,)
XX, YY    = np.meshgrid(x_centers, y_centers, indexing='xy')
patch_xy  = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)  # (n_patch,2)
n_patch   = int(patch_xy.shape[0])

# -----------------------------
# Helper: DAS (unchanged math)
# -----------------------------
def compute_das_xp(pressure_traces, detectors_xyz, c_mm_us, t_min, dt_us, z_vals_mm, xp_mod):
    Nd, Nt = pressure_traces.shape
    das_x = detectors_xyz[:, 0].astype(xp_mod.float32)
    Xpix, Zpix = xp_mod.meshgrid(das_x, z_vals_mm.astype(xp_mod.float32), indexing='ij')
    Nx, Nz = Xpix.shape
    Npix = Nx * Nz

    pixels = xp_mod.column_stack([
        Xpix.ravel(),
        xp_mod.zeros(Npix, dtype=xp_mod.float32),
        Zpix.ravel()
    ]).astype(xp_mod.float32)

    diff = pixels[:, None, :] - detectors_xyz[None, :, :]
    dist = xp_mod.linalg.norm(diff, axis=2).astype(xp_mod.float32)  # (Npix,Nd)

    # NOTE: use t_min and dt_us directly (Python floats), no xp_mod.float32(...)
    f = (dist / c_mm_us - t_min) / dt_us
    f = xp_mod.clip(f, 0.0, Nt - 1.000001).astype(xp_mod.float32)

    i0 = xp_mod.floor(f).astype(xp_mod.int32)
    i1 = i0 + 1
    w  = (f - i0).astype(xp_mod.float32)

    det_idx = xp_mod.broadcast_to(xp_mod.arange(Nd, dtype=xp_mod.int32)[None, :], (Npix, Nd))
    P0 = pressure_traces[det_idx, i0]
    P1 = pressure_traces[det_idx, i1]
    vals = (1.0 - w) * P0 + w * P1
    return vals.sum(axis=1, dtype=xp_mod.float32).reshape(Nx, Nz)


# -----------------------------
# Build per-sphere traces (unchanged math)
# -----------------------------
def build_per_sphere_true_traces(det_xyz, sph, t_vals_dev, c, B=1.0, Cp=1.0, bump_func=True, xp_mod=None):
    xp_mod = xp_mod or xp
    Nd = det_xyz.shape[0]
    t  = t_vals_dev[None, :]  # (1,Nt)

    sph = sph.astype(xp_mod.float32, copy=False)
    sx = sph[0]; sy = sph[1]; sz = sph[2]; sr = sph[3]

    Rx = (sx - det_xyz[:, 0])[:, None]
    Ry = (sy - det_xyz[:, 1])[:, None]
    Rz = (sz - det_xyz[:, 2])[:, None]
    r  = xp_mod.sqrt(Rx**2 + Ry**2 + Rz**2).astype(xp_mod.float32)

    K  = (B * (c**2)) / (2.0 * Cp) / r
    A0 = (K * r).astype(xp_mod.float32)
    B0 = (K * c).astype(xp_mod.float32)

    t0 = ((r - sr) / c).astype(xp_mod.float32)
    t1 = ((r + sr) / c).astype(xp_mod.float32)

    if bump_func:
        mask = ((t > t0) & (t < t1)).astype(xp_mod.float32)
        t_mid    = xp_mod.float32(0.5) * (t0 + t1)
        t_radius = xp_mod.float32(0.5) * (t1 - t0)

        # Avoid division warnings inside exp by masking
        z = (t - t_mid) / (t_radius + xp_mod.float32(1e-12))
        bump = xp_mod.where(mask > 0, xp_mod.exp( xp_mod.float32(1.0) / (z**2 - xp_mod.float32(1.0)) ), xp_mod.float32(0.0))

        p_s = (A0 - B0 * t) * mask * bump
    else:
        mask = ((t >= t0) & (t <= t1)).astype(xp_mod.float32)
        p_s = (A0 - B0 * t) * mask
        bump = mask  # returned per your original signature

    return p_s.astype(xp_mod.float32), r.astype(xp_mod.float32), bump.astype(xp_mod.float32)

# -----------------------------
# Patch-sum SIR in frequency (chunked, same result)
# H_s(q,f) = (1/n_patch) * sum_m (r0 / rm) * exp(-i 2π f (rm - r0)/c)
# -----------------------------
def build_Hs_for_sphere_chunked(r0, sx, sy, sz,
                                det_x, det_y, det_z,
                                patch_xy_dev, f_row, c, xp_mod=None,
                                chunk_size=4096):
    xp_mod = xp_mod or xp
    Nd = r0.shape[0]
    Npad = f_row.shape[1]
    H_s = xp_mod.zeros((Nd, Npad), dtype=xp_mod.complex64)

    n_patch = patch_xy_dev.shape[0]
    for start in range(0, n_patch, chunk_size):
        stop = min(start + chunk_size, n_patch)
        px = det_x + patch_xy_dev[None, start:stop, 0]   # (Nd, chunk)
        py = det_y + patch_xy_dev[None, start:stop, 1]   # (Nd, chunk)
        pz = xp_mod.broadcast_to(det_z, (Nd, stop - start))

        # FIX: do not wrap sx/sy/sz with xp_mod.float32(·); keep device scalars
        Rxm = (sx - px)
        Rym = (sy - py)
        Rzm = (sz - pz)
        rm  = xp_mod.sqrt(Rxm**2 + Rym**2 + Rzm**2).astype(xp_mod.float32) + xp_mod.float32(1e-12)

        delta_r = (rm - r0).astype(xp_mod.float32)  # (Nd, chunk)
        phase = xp_mod.exp(-1j * xp_mod.float32(2.0 * np.pi) * (delta_r[..., None] / xp_mod.float32(c)) * f_row)  # (Nd,chunk,Npad)
        amp   = (r0 / rm)[..., None].astype(xp_mod.float32)  # (Nd,chunk,1)

        H_s += (amp * phase).sum(axis=1, dtype=xp_mod.complex64)

    H_s /= xp_mod.float32(n_patch)
    return H_s

# -----------------------------
# Main SIR+EIR runner (math preserved)
# -----------------------------
def run_sir_eir(det_dev, t_vals_dev, c, B, Cp, H_eir_dev,
                patch_xy_dev, det_x, det_y, det_z,
                spheres_cpu, Npad, Nt, das_z_values_dev, filename,
                aperture_width_x_mm, aperture_height_y_mm, pitch_x_mm,
                n_div_x, n_div_y, f0_MHz, Nd, xp_mod=None, xpfft_mod=None):
    xp_mod = xp_mod or xp
    xpfft_mod = xpfft_mod or xp_fft

    Ns = spheres_cpu.shape[0]

    # Frequency axis on device
    freqs_dev = xp_mod.asarray(np_fft.fftfreq(Npad, d=float(dt)), dtype=xp_mod.float32)  # [MHz] = 1/us
    f_row = freqs_dev[None, :]  # (1, Npad)

    P_sir_sum_f = xp_mod.zeros((Nd, Npad), dtype=xp_mod.complex64)
    pressure_analytic = xp_mod.zeros((Nd, Nt), dtype=xp_mod.float32)

    for s_idx in range(Ns):
        sph_cpu = spheres_cpu[s_idx]
        sph_dev = xp_mod.asarray(sph_cpu, dtype=xp_mod.float32)

        # True per-sphere time traces (Nd x Nt)
        p_s, r0, _ = build_per_sphere_true_traces(det_dev, sph_dev, t_vals_dev, c, B=B, Cp=Cp, bump_func=bool(bump_func), xp_mod=xp_mod)
        pressure_analytic += p_s

        # FFT with zero-padding (Nd x Npad)
        P_s = xpfft_mod.fft(xp_mod.concatenate([p_s, xp_mod.zeros_like(p_s)], axis=1), n=Npad, axis=1)

        # Ensure device float32 scalars without crossing devices
        sx = xp_mod.asarray(sph_dev[0], dtype=xp_mod.float32)
        sy = xp_mod.asarray(sph_dev[1], dtype=xp_mod.float32)
        sz = xp_mod.asarray(sph_dev[2], dtype=xp_mod.float32)

        H_s = build_Hs_for_sphere_chunked(r0, sx, sy, sz, det_x, det_y, det_z,
                                          patch_xy_dev, f_row, c, xp_mod=xp_mod, chunk_size=4096)

        # Apply SIR for this sphere and accumulate
        P_sir_sum_f += (P_s * H_s)

    # Apply EIR, then IFFT (Nd x Nt)
    P_sir_eir_f = P_sir_sum_f * H_eir_dev[None, :]
    p_sir_eir   = xp_mod.real(xpfft_mod.ifft(P_sir_eir_f, n=Npad, axis=1))[:, :Nt].astype(xp_mod.float32)

    # Baselines (analytical already built; EIR-only = apply EIR to true spectrum)
    P_true_f    = xpfft_mod.fft(xp_mod.concatenate([pressure_analytic, xp_mod.zeros_like(pressure_analytic)], axis=1),
                                n=Npad, axis=1)
    P_eir_f     = P_true_f * H_eir_dev[None, :]
    p_eir_only  = xp_mod.real(xpfft_mod.ifft(P_eir_f, n=Npad, axis=1))[:, :Nt].astype(xp_mod.float32)

    # DAS reconstructions (unchanged math)
    t_min_host = t_vals_dev.min().item()   # Python float (safe with CuPy arrays)
    das_analytic = compute_das_xp(pressure_analytic, det_dev, c, t_min_host, dt, das_z_values_dev, xp_mod)
    das_eir      = compute_das_xp(p_eir_only,        det_dev, c, t_min_host, dt, das_z_values_dev, xp_mod)
    das_sir_eir  = compute_das_xp(p_sir_eir,         det_dev, c, t_min_host, dt, das_z_values_dev, xp_mod)


    # Save to CPU .npz (ensure host arrays)
    def to_cpu(a):
        if GPU and hasattr(a, "get"):
            return a.get()
        return np.asarray(a)

    np.savez(
        filename,
        t_vals=to_cpu(t_vals_dev),
        p_eir_only=to_cpu(p_eir_only),
        p_sir_eir=to_cpu(p_sir_eir),
        das_analytic=to_cpu(das_analytic),
        das_eir=to_cpu(das_eir),
        das_sir_eir=to_cpu(das_sir_eir),
        dt=dt,
        c=c,
        detectors=to_cpu(det_dev),
        spheres=spheres_cpu,  # already CPU
        aperture_width_x_mm=aperture_width_x_mm,
        aperture_height_y_mm=aperture_height_y_mm,
        pitch_x_mm=pitch_x_mm,
        n_div_x=n_div_x,
        n_div_y=n_div_y,
        f0_MHz=f0_MHz
    )

# -----------------------------
# Precompute device-side constants/grids
# -----------------------------
# Move long-lived arrays to device up front
det_dev = xp.asarray(detectors, dtype=xp.float32)
det_x = det_dev[:, 0][:, None]
det_y = det_dev[:, 1][:, None]
det_z = det_dev[:, 2][:, None]

t_vals_dev = xp.asarray(t_vals, dtype=xp.float32)

# Npad frequency array for H_eir
freqs_dev = xp.asarray(np_fft.fftfreq(Npad, d=dt), dtype=xp.float32)
H_eir_dev = H_function(freqs_dev, xp_mod=xp)

# DAS grids
das_x_values = detectors[:, 0].astype(np.float32)
z_resolution = dt * c
das_z_values = np.arange(-35.0, 0.0, z_resolution, dtype=np.float32)
das_z_values_dev = xp.asarray(das_z_values, dtype=xp.float32)

# Patches on device
patch_xy_dev = xp.asarray(patch_xy, dtype=xp.float32)

# Output folder
folder = Path("root/training_data")
folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Main loop (unchanged logic, with timer init)
# -----------------------------
start = time.time()

while i < n_runs:
    filename = folder / f"sir_eir_outputs_{i}.npz"
    spheres_cpu = make_spheres(n_spheres)  # generate on CPU
    run_sir_eir(det_dev, t_vals_dev, c, B, Cp, H_eir_dev,
                patch_xy_dev, det_x, det_y, det_z,
                spheres_cpu, Npad, Nt, das_z_values_dev, filename,
                aperture_width_x_mm, aperture_height_y_mm,
                pitch_x_mm, n_div_x, n_div_y, f0_MHz, Nd,
                xp_mod=xp, xpfft_mod=xp_fft)
    print(i)
    i += 1

    if i % 10 == 0:
        end = time.time()
        print("Execution time:", end - start, "seconds")
        start = time.time()

# Optional: sync device before exit
if GPU:
    xp.cuda.Stream.null.synchronize()








last_idx = i - 1  # i was incremented after the last save
last_file = folder / f"sir_eir_outputs_{last_idx}.npz"

data = np.load(last_file)

# Axes
t_vals = data["t_vals"]                      # (Nt,)
det_x  = data["detectors"][:, 0].astype(np.float32)  # (Nd,)

# 1) Detector-time pressure "composite" heatmap (SIR+EIR)
P = data["p_sir_eir"]  # shape (Nd, Nt)

plt.figure(figsize=(10, 4))
plt.title(f"Pressure (SIR+EIR) — detector vs time (run {last_idx})")
plt.xlabel("Time (µs)")
plt.ylabel("Detector x (mm)")
plt.imshow(
    P, aspect="auto", origin="lower",
    extent=[t_vals.min(), t_vals.max(), det_x.min(), det_x.max()]
)
plt.colorbar(label="Pressure (a.u.)")
plt.tight_layout()
plt.savefig(folder / f"heatmap_last_pressure_{last_idx}.png", dpi=200)
plt.show()

# 2) DAS image heatmap (x–z) from the same run (optional but handy)
das_img = data["das_sir_eir"]  # shape (Nx, Nz)

# Rebuild z-axis to match generation (range -35..0 mm, step dt*c)
dt = float(data["dt"])
c  = float(data["c"])
z_vals = np.arange(-35.0, 0.0, dt * c, dtype=np.float32)

plt.figure(figsize=(6, 5))
plt.title(f"DAS (SIR+EIR) — x–z image (run {last_idx})")
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")
plt.imshow(
    das_img.T, aspect="auto", origin="lower",  # transpose to plot x (horizontal) vs z (vertical)
    extent=[det_x.min(), det_x.max(), z_vals.min(), z_vals.max()]
)
plt.colorbar(label="Amplitude (a.u.)")
plt.tight_layout()
plt.savefig(folder / f"heatmap_last_das_{last_idx}.png", dpi=200)
plt.show()
