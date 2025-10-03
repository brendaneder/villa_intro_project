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


plt_trace = 1
plt_das = 0
plt_3d = 1
bump_func = 1
uniform = 1
n_spheres = 6

x_bounds=(-15, 15)
y_bounds=(-15, 15)
z_bounds=(-35,  -5)
r_bounds=(  0.5,   2.0)

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
                 x_bounds=x_bounds,
                 y_bounds=y_bounds,
                 z_bounds=z_bounds,
                 r_bounds=r_bounds):
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

det      = detectors          # (Nd,3)









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







Nd = det.shape[0]
Ns = spheres.shape[0]
Nt = int(t_vals.size)



# -----------------------------
# Frequency axis and EIR
# -----------------------------
Npad  = 2 * Nt
freqs = fft.fftfreq(Npad, d=dt).astype(np.float32)  # [MHz] = 1/us

# sigmoid
def H_function(x, k = 9, a0 = .5, b0 = 9):
    sig_a = 1 / ( 1 + np.exp( -k * (np.abs(x) - a0)))
    sig_b = 1 / ( 1 + np.exp( -k * (np.abs(x) - b0)))

    scale = np.abs( sig_a - sig_b )
    return scale

H_eir = H_function(freqs).astype(np.float32)


# -----------------------------
# Helper
# -----------------------------
def build_per_sphere_true_traces(det_xyz, sph, t_vals, c, B=1.0, Cp=1.0, bump_func=bump_func):
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
    


    
    if bump_func:
        mask = ((t > t0) & (t < t1)).astype(np.float32)      # (Nd,Nt)
        
        t_mid    = 0.5 * (t0 + t1)        # (Nd,1)
        t_radius = 0.5 * (t1 - t0)        # (Nd,1)


        bump = np.where( mask , np.exp( 1 /( ((t-t_mid) / t_radius) ** 2 -1) ), 0.0)

        # Multiply the N-wave by the bump; bump broadcasts across time (Nd,1) → (Nd,Nt)
        p_s = (A0 - B0 * t) * mask * bump                  # (Nd,Nt)

    else:
        # Standard N-wave (no bump)
        mask = ((t >= t0) & (t <= t1)).astype(np.float32)      # (Nd,Nt)
        p_s = (A0 - B0 * t) * mask                         # (Nd,Nt)

    
    return p_s, r, bump  # return r for SIR reference

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

pressure_analytic = np.zeros((Nd, Nt), dtype=np.float32)


for s_idx in range(Ns):
    sph = spheres[s_idx]

    # 1) True per-sphere time signal at each detector + center ranges r0(q)
    p_s, r0, bump = build_per_sphere_true_traces(det, sph, t_vals, c, B=B, Cp=Cp)  # p_s: (Nd,Nt), r0: (Nd,1)
    pressure_analytic += p_s

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


# EIR-only
P_true_f    = fft.fft(np.concatenate([pressure_analytic, np.zeros_like(pressure_analytic)], axis=1),
                      n=Npad, axis=1)
P_eir_f     = P_true_f * H_eir[None, :]
p_eir_only  = np.real(fft.ifft(P_eir_f, n=Npad, axis=1))[:, :Nt].astype(np.float32)

# -----------------------------
# Plot single-element line (element #63 → index 62)
# -----------------------------
elt_plot = 62 if Nd >= 63 else Nd - 1
if plt_trace:
    plt.figure(figsize=(10, 5))
    plt.plot(t_vals, pressure_analytic[elt_plot], label="Analytical (true)")
    plt.plot(t_vals, p_eir_only[elt_plot],      label="EIR only")
    plt.plot(t_vals, p_sir_eir[elt_plot],      label=f"SIR, (x, y) = {n_div_x}×{n_div_y}) + EIR")
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


if plt_3d:
    def _map_y_to_intensity(y, y_min, y_max):

        if y_max == y_min:
            # Avoid divide-by-zero: constant gray at 100/255
            return np.full_like(y, 100 / 255.0, dtype=float)
        y_clipped = np.clip(y, y_min, y_max)
        return (100.0 + (y_clipped - y_min) * 155.0 / (y_max - y_min)) / 255.0


    def plot_spheres_xz(
        spheres: np.ndarray,
        xlim: tuple,
        ylim: tuple,
        zlim: tuple,
        resolution: int = 800,
        show: bool = True,
        save_path: str | None = None,
    ):
        x_min, x_max = xlim
        y_min, y_max = ylim
        z_min, z_max = zlim

        width = x_max - x_min
        height = z_max - z_min
        if width <= 0 or height <= 0:
            raise ValueError("xlim and zlim must have positive extents (max > min).")

        # Grid size preserving aspect ratio
        if width >= height:
            nx = int(resolution)
            nz = max(1, int(round(resolution * (height / width))))
        else:
            nz = int(resolution)
            nx = max(1, int(round(resolution * (width / height))))

        xs = np.linspace(x_min, x_max, nx)
        zs = np.linspace(z_min, z_max, nz)
        X, Z = np.meshgrid(xs, zs)

        # Start with black background
        img = np.zeros((nz, nx), dtype=float)

        # Paint each sphere's circle in XZ with intensity based on its Y
        # Overlaps: keep the brighter (max) intensity
        if spheres is not None and spheres.size > 0:
            for sx, sy, sz, r in np.asarray(spheres, dtype=float):
                if r <= 0:
                    continue
                # Quick reject if outside view
                if (sx + r) < x_min or (sx - r) > x_max or (sz + r) < z_min or (sz - r) > z_max:
                    continue
                mask = (X - sx) ** 2 + (Z - sz) ** 2 <= r ** 2
                if not np.any(mask):
                    continue
                intensity = _map_y_to_intensity(np.array([sy]), y_min, y_max)[0]
                img[mask] = np.maximum(img[mask], intensity)

        extent = (x_min, x_max, z_min, z_max)

        if show or save_path is not None:
            plt.figure(figsize=(6, 6 * (nz / nx)))
            plt.imshow(img, extent=extent, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
            plt.xlabel("X")
            plt.ylabel("Z")
            plt.title("XZ Projection of Spheres (Y → intensity)")
            if save_path is not None:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()

        return img, extent

    plot_spheres_xz(spheres, x_bounds, y_bounds, z_bounds, resolution=1000, show=True)
