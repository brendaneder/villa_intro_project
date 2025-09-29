# current_SIR_Rectangular_DAS_Final.py
# Adds rectangular-element Spatial Impulse Response (SIR) to your DAS.
# - Loads Program 1 output (output_data.npz)
# - (Optionally) applies your EIR in frequency domain (matching Program 2)
# - Recomputes DAS using far-field rectangular SIR weights (sinc × sinc)
# - Supports a simple near-field patch average (m=2) toggle
#
# Units:
#   c in mm/us, t in us, distances in mm, sample_rate in samples/us (MHz)
#
# Geometry (defaults from your notes):
#   128 elements, pitch_x = 0.2 mm, element width a = 0.2 mm (X), height b = 3.0 mm (Y)
#   Elements on Z=0 plane, normal to +Z.
#
# Output:
#   - Renders SIR-weighted DAS image with 0 mapped to pure white
#   - Saves "sir_das_output.npz" = { 'das_img_sir', 'das_x_values', 'das_z_values', ... }

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------------
# Config (edit as needed)
# -----------------------------
APPLY_EIR = True          # set False to skip EIR here
USE_PATCH_M2 = True      # True => 2×2 sub-patch averaging (near-field helper)

# Element geometry
N_EL_X   = 128            # elements along X (your array)
PITCH_X  = 0.2            # mm (center-to-center spacing)
A_WIDTH  = 0.2            # mm, element width along X
B_HEIGHT = 3.0            # mm, element height along Y (adjustable variable)

# EIR model (matches your Program 2 default)
F0_MHZ   = 8.0            # center frequency in MHz
def build_eir_response(freqs_abs_mhz):
    sigma = np.sqrt(F0_MHZ / np.log(2.0))
    return np.exp(-((freqs_abs_mhz - F0_MHZ) ** 2) / (2.0 * sigma**2))

# -----------------------------
# Helpers
# -----------------------------
def sinc_pi(x):
    out = np.ones_like(x, dtype=np.float64)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(x[nz]) / x[nz]
    return out

def rectangular_SIR_weight_f0(dx, dy, r, a, b, f0_mhz, c_mm_per_us):
    """
    Far-field rectangular piston directivity at a single frequency f0 (MHz):
      w = sinc( pi f a X / (c r) ) * sinc( pi f b Y / (c r) )
    where X=dx, Y=dy in the element's local frame (X along width a, Y along height b).
    """
    argX = np.pi * f0_mhz * (a * dx) / (c_mm_per_us * r)
    argY = np.pi * f0_mhz * (b * dy) / (c_mm_per_us * r)
    return sinc_pi(argX) * sinc_pi(argY)

def rectangular_SIR_weight_patch_m2(px, py, pz, ex, ey, ez, a, b, f0_mhz, c):
    """
    2×2 patch averaging: split a×b into four (a/2×b/2) sub-patches centered at offsets
    (+/- a/4, +/- b/4) around the element center. Average the four far-field weights.
    """
    dx_offsets = np.array([-a/4, +a/4, -a/4, +a/4], dtype=np.float64)
    dy_offsets = np.array([-b/4, -b/4, +b/4, +b/4], dtype=np.float64)

    weights = []
    for ox, oy in zip(dx_offsets, dy_offsets):
        ex_i = ex + ox
        ey_i = ey + oy
        dx = px - ex_i
        dy = py - ey_i
        dz = pz - ez
        r  = np.sqrt(dx*dx + dy*dy + dz*dz)
        w  = rectangular_SIR_weight_f0(dx, dy, r, a/2.0, b/2.0, f0_mhz, c)
        weights.append(w)
    return np.mean(np.stack(weights, axis=0), axis=0)

# -----------------------------
# Load forward data (Program 1)
# -----------------------------
dat = np.load("output_data.npz")
t_vals             = dat["t_vals"]               # us
pressure_composit  = dat["pressure_composit"]    # [Nd, Nt]
dt_us              = float(dat["dt"])            # us

# If Program 1 starts saving these, prefer them; otherwise use defaults below
c_mm_per_us = float(dat["c"]) if "c" in dat else 1.5

# -----------------------------
# Rebuild array geometry (matches Program 1 centering)
# -----------------------------
Nd = pressure_composit.shape[0]
Nt = pressure_composit.shape[1]

x_npoints_detector, y_npoints_detector = N_EL_X, 1
x_spacing_detector, y_spacing_detector = PITCH_X, 0.2  # y spacing irrelevant for single row

ix = np.arange(x_npoints_detector, dtype=np.float64)
jy = np.arange(y_npoints_detector, dtype=np.float64)
X, Y = np.meshgrid(ix * x_spacing_detector, jy * y_spacing_detector, indexing='ij')

detectors = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size, dtype=np.float64)))
detectors -= detectors.mean(axis=0)  # center the array

# -----------------------------
# Optional: Apply your EIR (Program 2’s model) to channel data
# -----------------------------
data_to_transform = pressure_composit.astype(np.float64)

if APPLY_EIR:
    pad = np.zeros_like(data_to_transform)
    padded = np.concatenate((data_to_transform, pad), axis=1)
    n_time = data_to_transform.shape[1]
    Npad   = padded.shape[1]

    freqs_cyc_per_us = fft.fftfreq(2 * n_time, d=dt_us)  # cycles/us == MHz
    H = build_eir_response(np.abs(freqs_cyc_per_us))

    DATA_F = fft.fft(padded, n=Npad, axis=1)
    DATA_F *= H.reshape(1, -1)
    data_eir = np.real(fft.ifft(DATA_F, n=Npad, axis=1))[:, :n_time]
    channel_data = data_eir
else:
    channel_data = data_to_transform

# -----------------------------
# Build DAS grid (same logic as Program 1)
# -----------------------------
das_x_values = detectors[:, 0].astype(np.float64)
z_resolution = dt_us * c_mm_per_us
das_z_values = np.arange(-35.0, 0.0, z_resolution, dtype=np.float64)

Xpix, Zpix = np.meshgrid(das_x_values, das_z_values, indexing='ij')
Nx, Nz = Xpix.shape
Npix = Nx * Nz

pixels = np.column_stack((Xpix.ravel(), np.zeros(Npix, dtype=np.float64), Zpix.ravel()))

# -----------------------------
# Geometric delays and SIR weights (far-field rectangular)
# -----------------------------
diff = pixels[:, None, :] - detectors[None, :, :]   # (Npix, Nd, 3)
dist = np.linalg.norm(diff, axis=2)                 # (Npix, Nd)

# Time sample indices (linear interpolation)
f_idx = (dist / c_mm_per_us - t_vals[0]) / dt_us
f_idx = np.clip(f_idx, 0.0, Nt - 1.000001)
i0 = np.floor(f_idx).astype(np.int32)
i1 = i0 + 1
wlin  = (f_idx - i0).astype(np.float64)

# Local element-frame components (X along width a, Y along height b)
dx = diff[..., 0]
dy = diff[..., 1]
dz = diff[..., 2]
r  = dist

# SIR weights
if USE_PATCH_M2:
    weights = rectangular_SIR_weight_patch_m2(
        px=pixels[:, 0][:, None], py=pixels[:, 1][:, None], pz=pixels[:, 2][:, None],
        ex=detectors[:, 0][None, :], ey=detectors[:, 1][None, :], ez=detectors[:, 2][None :],
        a=A_WIDTH, b=B_HEIGHT, f0_mhz=F0_MHZ, c=c_mm_per_us
    )
else:
    weights = rectangular_SIR_weight_f0(dx, dy, r, A_WIDTH, B_HEIGHT, F0_MHZ, c_mm_per_us)

# -----------------------------
# Interpolate channel traces, apply SIR, sum
# -----------------------------
Nd = detectors.shape[0]
det_idx = np.broadcast_to(np.arange(Nd, dtype=np.int32)[None, :], (Npix, Nd))

P0 = channel_data[det_idx, i0]
P1 = channel_data[det_idx, i1]
vals = (1.0 - wlin) * P0 + wlin * P1

# Apply SIR weighting per (pixel, detector)
vals *= weights

das_flat = vals.sum(axis=1, dtype=np.float64)
das_img_sir = das_flat.reshape(Nx, Nz)

# -----------------------------
# Plot with 0 mapped to pure white
# -----------------------------
# Custom colormap: blue → white → red, with vcenter=0
cmap_bwr_white = mcolors.LinearSegmentedColormap.from_list(
    "blue_white_red", ["blue", "white", "red"]
)
norm_center0 = mcolors.TwoSlopeNorm(
    vmin=np.nanmin(das_img_sir),
    vcenter=0.0,
    vmax=np.nanmax(das_img_sir)
)

plt.figure(figsize=(7,6))
plt.imshow(
    np.flipud(das_img_sir.T),
    extent=[das_x_values.min(), das_x_values.max(), -35, 0],
    aspect='auto', cmap=cmap_bwr_white, norm=norm_center0, origin='upper'
)
plt.xlabel("x (mm)")
plt.ylabel("Depth (mm)")
plt.title(f"SIR-weighted DAS (f0={F0_MHZ:.1f} MHz, a={A_WIDTH} mm, b={B_HEIGHT} mm)")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.show()

# -----------------------------
# Save results
# -----------------------------
np.savez(
    "sir_das_output.npz",
    das_img_sir=das_img_sir,
    das_x_values=das_x_values,
    das_z_values=das_z_values,
    a_width_mm=A_WIDTH,
    b_height_mm=B_HEIGHT,
    f0_mhz=F0_MHZ,
    used_patch_m2=USE_PATCH_M2
)
print("Saved: sir_das_output.npz")
