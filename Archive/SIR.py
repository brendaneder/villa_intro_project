# SIR.py — now uses pressure_composit from bump_generation.py as the input data
# Pipeline:
#   (1) Import composite pressure traces from bump_generation.py
#   (2) (Optional) Apply EIR in frequency
#   (3) IFFT back to time
#   (4) DAS imaging for raw and EIR-only
# Notes:
#   - The previous build_per_sphere_true_traces() path is removed.
#   - Importing bump_generation.py will execute that script once.

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt_single_trace = 1
plt_das = 0

# -----------------------------
# Geometry & discretization
# -----------------------------
pitch_x_mm           = 0.2      # center-to-center spacing along X (mm)
aperture_width_x_mm  = pitch_x_mm  # element width in X (mm)
aperture_height_y_mm = 10      # element height in Y (mm)
n_div_y              = 10     # <-- number of sub-rectangles per y-axis (n). Increase for accuracy.
n_div_x              = 1       # number of x-divisions (usually 1)


# -----------------------------
# EIR parameters (placeholder)
# -----------------------------
f0_MHz = 8.0
sigma  = np.sqrt(f0_MHz / np.log(2.0))

def H_function(x, k = 9, a0 = 0.5, b0 = 9):
    sig_a = 1 / ( 1 + np.exp( -k * (np.abs(x) - a0)))
    sig_b = 1 / ( 1 + np.exp( -k * (np.abs(x) - b0)))

    scale = np.abs( sig_a - sig_b )
    return scale

# -----------------------------
# Load data from output_data_bump.npz
# -----------------------------
data = np.load("output_data_bump.npz", allow_pickle=True)

pressure_analytic = data["pressure_composit"].astype(np.float32)  # (Nd, Nt)
t_vals   = data["t_vals"].astype(np.float32)
dt       = float(data["dt"])
det      = data["detectors"].astype(np.float32)
spheres  = data["spheres"].astype(np.float32)
c        = float(data["c"])
B        = float(data["B"])
Cp       = float(data["Cp"])


Nd = det.shape[0]
Ns = spheres.shape[0]
Nt = int(t_vals.size)


# -----------------------------
# EIR (freq-domain)
# -----------------------------
Npad  = 2 * Nt
freqs = fft.fftfreq(Npad, d=dt).astype(np.float32)
H_eir = H_function(freqs).astype(np.float32)

P_true_f   = fft.fft(
    np.concatenate([pressure_analytic, np.zeros_like(pressure_analytic)], axis=1),
    n=Npad, axis=1
)
P_eir_f    = P_true_f * H_eir[None, :]
p_eir_only = np.real(fft.ifft(P_eir_f, n=Npad, axis=1))[:, :Nt].astype(np.float32)


# -----------------------------
# Patch positions (relative to element center)
# -----------------------------
# Sub-rectangles are equal-sized, with centers regularly spaced across the aperture.
ax = float(aperture_width_x_mm)
ay = float(aperture_height_y_mm)

# Create n_div centers in each axis spanning [-ax/2, ax/2] and [-ay/2, ay/2]
# Using "pixel center" convention (no patch exactly on the edge)
x_centers = (np.arange(n_div_x, dtype=np.float32) + 0.5) / n_div_x * ax - ax / 2.0  # (n_div,)
y_centers = (np.arange(n_div_y, dtype=np.float32) + 0.5) / n_div_y * ay - ay / 2.0  # (n_div,)
XX, YY    = np.meshgrid(x_centers, y_centers, indexing='xy')                    # (n_div,n_div)
patch_xy  = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)       # (n_patch,2)
n_patch   = int(patch_xy.shape[0])

# -----------------------------
# Pre-allocate accumulators
# -----------------------------
P_sir_sum_f = np.zeros((Nd, Npad), dtype=np.complex64)  # sum over spheres after SIR (freq domain)



# After summing all spheres with SIR, apply EIR, then IFFT
P_sir_eir_f = P_sir_sum_f * H_eir[None, :]
p_sir_eir   = np.real(fft.ifft(P_sir_eir_f, n=Npad, axis=1))[:, :Nt].astype(np.float32)


# -----------------------------
# Build SIR via patch summation PER SPHERE, then sum spectra
# -----------------------------
# Frequency row for broadcasting
f_row = freqs[None, :]  # (1,Npad)


# Detector centers
det_x = det[:, 0][:, None]  # (Nd,1)
det_y = det[:, 1][:, None]
det_z = det[:, 2][:, None]  # (Nd,1)



# -----------------------------
# Plot single-element trace
# -----------------------------
elt_plot = 62 if Nd >= 63 else Nd - 1

if plt_single_trace:
    plt.figure(figsize=(10, 5))
    plt.plot(t_vals, pressure_analytic[elt_plot], label="Raw (pressure_composit)")
    plt.plot(t_vals, p_eir_only[elt_plot],        label="EIR only")
    plt.title(f"Transducer element #{elt_plot+1} pressure vs time")
    plt.xlabel("time (µs)")
    plt.ylabel("pressure (arb.)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# DAS (delay-and-sum)
# -----------------------------
def compute_das(pressure_traces, detectors_xyz, c_mm_us, t_min, dt_us, z_vals_mm):
    Nd, Nt = pressure_traces.shape
    das_x = detectors_xyz[:, 0].astype(np.float32)
    Xpix, Zpix = np.meshgrid(das_x, z_vals_mm.astype(np.float32), indexing='ij')
    Nx, Nz = Xpix.shape
    Npix = Nx * Nz

    pixels = np.column_stack([
        Xpix.ravel(),
        np.zeros(Npix, dtype=np.float32),   # y=0 imaging plane
        Zpix.ravel()
    ]).astype(np.float32)  # (Npix,3)

    diff = pixels[:, None, :] - detectors_xyz[None, :, :]
    dist = np.linalg.norm(diff, axis=2).astype(np.float32)  # (Npix,Nd)

    f = (dist / c_mm_us - t_min) / dt_us
    f = np.clip(f, 0.0, Nt - 1.000001).astype(np.float32)
    i0 = np.floor(f).astype(np.int32)
    i1 = i0 + 1
    w  = (f - i0).astype(np.float32)

    det_idx = np.broadcast_to(np.arange(Nd, dtype=np.int32)[None, :], (Npix, Nd))
    P0 = pressure_traces[det_idx, i0]
    P1 = pressure_traces[det_idx, i1]
    vals = (1.0 - w) * P0 + w * P1
    return vals.sum(axis=1, dtype=np.float32).reshape(Nx, Nz)

das_x_values = det[:, 0].astype(np.float32)
z_resolution = dt * c
das_z_values = np.arange(-35.0, 0.0, z_resolution, dtype=np.float32)

das_raw = compute_das(pressure_analytic, det, c, t_vals.min(), dt, das_z_values)
das_eir = compute_das(p_eir_only,        det, c, t_vals.min(), dt, das_z_values)

# -----------------------------
# Plot DAS images
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
extent = [das_x_values.min(), das_x_values.max(), -35.0, 0.0]

def imshow_centered(ax, img, title):
    im = ax.imshow(
        img.T, extent=extent, aspect='auto', cmap='seismic', origin='upper',
        norm= mcolors.TwoSlopeNorm(vmin=img.min(), vcenter=0.0, vmax=img.max())
    )
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Depth (mm)")
    return im

im0 = imshow_centered(axes[0], das_raw, "DAS (Raw pressure_composit)")
im1 = imshow_centered(axes[1], das_eir, "DAS (EIR only)")
fig.colorbar(im0, ax=axes[0], label="DAS amplitude")
fig.colorbar(im1, ax=axes[1], label="DAS amplitude")
if plt_das:
    plt.show()

# -----------------------------
# Save outputs
# -----------------------------
np.savez(
    "sir_eir_outputs.npz",
    t_vals=t_vals,
    p_raw=pressure_analytic,
    p_eir_only=p_eir_only,
    das_raw=das_raw,
    das_eir=das_eir,
    dt=dt,
    c=c,
    detectors=det,
    spheres=spheres,
    aperture_width_x_mm=aperture_width_x_mm,
    aperture_height_y_mm=aperture_height_y_mm,
    pitch_x_mm=pitch_x_mm,
    f0_MHz=f0_MHz
)
print("Saved: sir_eir_outputs.npz")
