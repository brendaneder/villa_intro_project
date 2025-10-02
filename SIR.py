# 3_SIR_EIR_PACT.py  —  Rectangular receive SIR via n×n sub-rectangles (patch summation)
# Pipeline per your spec:
#   (1) True analytical (per-sphere time signals)
#   (2) Apply SIR on receive using patch method in frequency domain (no double 1/r or delay)
#   (3) Apply EIR in frequency
#   (4) IFFT back to time
# Outputs:
#   - Line plot at element #63: Analytical vs EIR-only vs SIR+EIR
#   - Three DAS images in one window (Analytical, EIR-only, SIR+EIR)
# Notes:
#   - "Analytical" traces are rebuilt per-sphere here (so we can attach H_s(q,f) per sphere).
#   - SIR uses relative factors: (r0/rm)*exp(-i 2π f (rm - r0)/c) to avoid double-propagation.
#   - Adjustable geometry: aperture_width_x_mm, aperture_height_y_mm, pitch_x_mm
#   - Adjustable sub-division: n_div (e.g., 2, 4, 6, 8...). n_div=1 reduces to element center only.

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt_das = 1

bump_func = 1

# -----------------------------
# Geometry & discretization
# -----------------------------
pitch_x_mm           = 0.2      # center-to-center spacing along X (mm)
aperture_width_x_mm  = pitch_x_mm  # element width in X (mm)
aperture_height_y_mm = 10      # element height in Y (mm)
n_div_y              = 100     # <-- number of sub-rectangles per y-axis (n). Increase for accuracy.
n_div_x              = 1       # number of x-divisions (usually 1)

# -----------------------------
# EIR parameters (Gaussian-in-frequency, matching your EIS.py style)
# -----------------------------
f0_MHz = 8.0
sigma  = np.sqrt(f0_MHz / np.log(2.0))

# -----------------------------
# Load analytical base data
# -----------------------------
if bump_func:
    D = np.load("output_data_bump.npz") 
else:
    D = np.load("output_data.npz")


t_vals   = D["t_vals"].astype(np.float32)              # [us]
dt       = float(D["dt"])                              # [us]
det      = D["detectors"].astype(np.float32)           # (Nd,3)
spheres  = D["spheres"].astype(np.float32)             # (Ns,4): (x,y,z,r) in mm
c        = float(D["c"])                               # [mm/us]
# Optional material constants used in your analytical derivation (fallback to 1)
B  = float(D.get("B", 1.0))
Cp = float(D.get("Cp", 1.0))

Nd = det.shape[0]
Ns = spheres.shape[0]
Nt = int(t_vals.size)

# -----------------------------
# Frequency axis and EIR
# -----------------------------
Npad  = 2 * Nt
freqs = fft.fftfreq(Npad, d=dt).astype(np.float32)  # [MHz] = 1/us

# sigmoid
def H_function(x, k = 9, a0 = 0.5, b0 = 9):
    sig_a = 1 / ( 1 + np.exp( -k * (np.abs(x) - a0)))
    sig_b = 1 / ( 1 + np.exp( -k * (np.abs(x) - b0)))

    scale = np.abs( sig_a - sig_b )
    return np.ones(len(x))

H_eir = H_function(freqs).astype(np.float32)


# -----------------------------
# Helper
# -----------------------------
def build_per_sphere_true_traces(det_xyz, sph, t_vals, c, B=1.0, Cp=1.0):
    """
    Recreate per-sphere analytical contributions p_s(q,t) (Nd×Nt) matching your time-of-flight
    and geometric-spreading model. This mirrors the construction used in Analytical.py
    (triangle/line segment between t0=(r-sr)/c and t1=(r+sr)/c with 1/r scaling).
    """
    Nd = det_xyz.shape[0]
    t  = t_vals[None, :]  # (1,Nt)

    sx, sy, sz, sr = sph
    Rx = (sx - det_xyz[:, 0])[:, None].astype(np.float32)  # (Nd,1)
    Ry = (sy - det_xyz[:, 1])[:, None].astype(np.float32)
    Rz = (sz - det_xyz[:, 2])[:, None].astype(np.float32)
    r  = np.sqrt(Rx**2 + Ry**2 + Rz**2).astype(np.float32) # (Nd,1)

    # Same scalings as previously used in your analytical code
    # (If your Analytical.py uses a slightly different closed form, port it here.)
    K  = (B * (c**2)) / (2.0 * Cp) / r                     # (Nd,1)
    A0 = (K * r).astype(np.float32)                        # (Nd,1)
    B0 = (K * c).astype(np.float32)                        # (Nd,1)

    t0 = ((r - sr) / c).astype(np.float32)                 # (Nd,1)
    t1 = ((r + sr) / c).astype(np.float32)                 # (Nd,1)
    mask = ((t >= t0) & (t <= t1)).astype(np.float32)      # (Nd,Nt)

    # Linear segment: A0 - B0 * t over [t0, t1]
    p_s = (A0 - B0 * t) * mask                             # (Nd,Nt)
    return p_s, r  # return r for SIR reference

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

# -----------------------------
# Build SIR via patch summation PER SPHERE, then sum spectra
# -----------------------------
# Frequency row for broadcasting
f_row = freqs[None, :]  # (1,Npad)

# Detector centers
det_x = det[:, 0][:, None]  # (Nd,1)
det_y = det[:, 1][:, None]
det_z = det[:, 2][:, None]  # (Nd,1)

for s_idx in range(Ns):
    sph = spheres[s_idx]

    # 1) True per-sphere time signal at each detector + center ranges r0(q)
    p_s, r0 = build_per_sphere_true_traces(det, sph, t_vals, c, B=B, Cp=Cp)  # p_s: (Nd,Nt), r0: (Nd,1)

    # FFT with zero-padding → P_s(q,f)
    P_s = fft.fft(np.concatenate([p_s, np.zeros_like(p_s)], axis=1), n=Npad, axis=1)  # (Nd,Npad)

    # 2) Patch distances rm(q, m) for m=1..n_patch (vectorized)
    sx, sy, sz, sr = sph
    # Patch absolute coordinates for every detector: (Nd, n_patch, 3)
    # Each detector center at (det_x, det_y, det_z); add patch XY offsets in plane z=det_z
    px = det_x + patch_xy[None, :, 0][:, None, :]  # shape wrong; fix to (Nd, n_patch)
    # Better: broadcast properly
    px = det_x + patch_xy[None, :, 0]             # (Nd, n_patch)
    py = det_y + patch_xy[None, :, 1]             # (Nd, n_patch)
    pz = np.broadcast_to(det_z, (Nd, n_patch)).astype(np.float32)

    # Vector from patch to sphere center: (Nd, n_patch)
    Rxm = (sx - px).astype(np.float32)
    Rym = (sy - py).astype(np.float32)
    Rzm = (sz - pz).astype(np.float32)

    rm = np.sqrt(Rxm**2 + Rym**2 + Rzm**2).astype(np.float32) + 1e-12  # (Nd, n_patch)

    # 3) Build near-field receive SIR H_s(q,f) by patch summation with RELATIVE factors
    #    H_s(q,f) = (1/n_patch) * sum_m (r0 / rm) * exp(-i 2π f (rm - r0)/c)
    #    Shapes: r0: (Nd,1) → broadcast; rm: (Nd,n_patch); result: (Nd,Npad)
    delta_r = (rm - r0)                              # (Nd,n_patch)
    # phase term: (Nd, n_patch, Npad)
    # To keep memory reasonable, do the sum over patches in chunks if needed:
    # Here we do it in one go for clarity.

    # Compute exp(-i 2π f Δr / c) → (Nd, Npad, n_patch)
    # We'll compute as (Nd, n_patch, Npad) then sum over patches
    # Use outer product style: Δr[..., None] with f_row to get (Nd,n_patch,Npad)
    phase = np.exp(-1j * 2.0 * np.pi * (delta_r[..., None] / c) * f_row)  # (Nd, n_patch, Npad)

    # amplitude ratio r0/rm: (Nd, n_patch) → (Nd, n_patch, 1)
    amp = (r0 / rm)[..., None].astype(np.float32)

    # patch contribution: (Nd, n_patch, Npad)
    H_parts = amp * phase

    # average over patches → (Nd, Npad)
    H_s = H_parts.mean(axis=1).astype(np.complex64)

    # 4) Apply H_s to per-sphere spectrum and accumulate over spheres
    P_sir_sum_f += (P_s * H_s)

# 5) After summing all spheres with SIR, apply EIR, then IFFT
P_sir_eir_f = P_sir_sum_f * H_eir[None, :]
p_sir_eir   = np.real(fft.ifft(P_sir_eir_f, n=Npad, axis=1))[:, :Nt].astype(np.float32)

# -----------------------------
# Also compute baselines for comparison:
#   - Analytical (sum over spheres, no SIR/EIR)
#   - EIR only (no SIR)
# -----------------------------
# Analytical (time domain) — rebuild by summing p_s across spheres
pressure_analytic = np.zeros((Nd, Nt), dtype=np.float32)
for s_idx in range(Ns):
    p_s, _ = build_per_sphere_true_traces(det, spheres[s_idx], t_vals, c, B=B, Cp=Cp)
    pressure_analytic += p_s

# EIR-only
P_true_f    = fft.fft(np.concatenate([pressure_analytic, np.zeros_like(pressure_analytic)], axis=1),
                      n=Npad, axis=1)
P_eir_f     = P_true_f * H_eir[None, :]
p_eir_only  = np.real(fft.ifft(P_eir_f, n=Npad, axis=1))[:, :Nt].astype(np.float32)

# -----------------------------
# Plot single-element line (element #63 → index 62)
# -----------------------------
elt_plot = 62 if Nd >= 63 else Nd - 1
plt.figure(figsize=(10, 5))
plt.plot(t_vals, pressure_analytic[elt_plot], label="Analytical (true)")
plt.plot(t_vals, p_eir_only[elt_plot],      label="EIR only")
plt.plot(t_vals, p_sir_eir[elt_plot],      label=f"SIR(X x Y ={n_div_x}×{n_div_y}) + EIR")
plt.title(f"Transducer element #{elt_plot+1} pressure vs time")
plt.xlabel("time (µs)")
plt.ylabel("pressure (arb.)")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# DAS (delay-and-sum) helper with linear interp
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

das_analytic = compute_das(pressure_analytic, det, c, t_vals.min(), dt, das_z_values)
das_eir      = compute_das(p_eir_only,        det, c, t_vals.min(), dt, das_z_values)
das_sir_eir  = compute_das(p_sir_eir,         det, c, t_vals.min(), dt, das_z_values)

# -----------------------------
# Plot the three DAS images in a single window
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
extent = [das_x_values.min(), das_x_values.max(), -35.0, 0.0]

def imshow_centered(ax, img, title):
    im = ax.imshow(
        img.T,
        extent=extent, aspect='auto', cmap='seismic', origin='upper',
        norm=mcolors.TwoSlopeNorm(vmin=img.min(), vcenter=0.0, vmax=img.max())
    )
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Depth (mm)")
    return im

im0 = imshow_centered(axes[0], das_analytic, f"DAS (Analytical)")
im1 = imshow_centered(axes[1], das_eir,      f"DAS (EIR only)")
im2 = imshow_centered(axes[2], das_sir_eir,  f"DAS (SIR n={n_div_x}×{n_div_y}) + EIR")

fig.colorbar(im0, ax=axes[0], label="DAS amplitude")
fig.colorbar(im1, ax=axes[1], label="DAS amplitude")
fig.colorbar(im2, ax=axes[2], label="DAS amplitude")

if plt_das:
    plt.show()

# -----------------------------
# Save arrays for later use
# -----------------------------
np.savez(
    "sir_eir_outputs.npz",
    t_vals=t_vals,
    p_eir_only=p_eir_only,
    p_sir_eir=p_sir_eir,
    das_analytic=das_analytic,
    das_eir=das_eir,
    das_sir_eir=das_sir_eir,
    dt=dt,
    c=c,
    detectors=det,
    spheres=spheres,
    aperture_width_x_mm=aperture_width_x_mm,
    aperture_height_y_mm=aperture_height_y_mm,
    pitch_x_mm=pitch_x_mm,
    n_div_x=n_div_x,
    n_div_y=n_div_y,
    f0_MHz=f0_MHz
)
print("Saved: sir_eir_outputs.npz")
