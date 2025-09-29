# Top-down (Figure 3–style) view of received data.
# Shows z (depth) "into" the medium vs. lateral element position x.
# Triple plot: Analytical, EIR-only, and SIR+EIR.
#
# Assumes you have:
#   - output_data.npz   (from Analytical.py)
#   - sir_eir_outputs.npz (from 3_SIR_EIR_PACT.py)  -> contains p_eir_only, p_sir_eir
#
# If sir_eir_outputs.npz is missing, the code will compute EIR-only on the fly
# using the same Gaussian-in-frequency model used earlier.

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ------------------------
# Load base analytical data
# ------------------------
D = np.load("output_data.npz")
t_vals            = D["t_vals"].astype(np.float32)            # [us]
pressure_analytic = D["pressure_composit"].astype(np.float32)  # (Nd, Nt)
detectors         = D["detectors"].astype(np.float32)          # (Nd, 3)
c                 = float(D["c"])                              # [mm/us]
dt                = float(D["dt"])                             # [us]

Nd, Nt = pressure_analytic.shape
x_mm   = detectors[:, 0].astype(np.float32)

# Convert time to depth (mm): set z=0 at earliest recorded time
t0     = float(t_vals.min())
z_mm   = c * (t_vals - t0)   # [mm]; increasing depth into the medium

# ------------------------
# Try to load EIR-only and SIR+EIR from the SIR file
# ------------------------
p_eir_only = None
p_sir_eir  = None
try:
    S = np.load("sir_eir_outputs.npz")
    p_eir_only = S["p_eir_only"].astype(np.float32)  # (Nd, Nt)
    p_sir_eir  = S["p_sir_eir"].astype(np.float32)   # (Nd, Nt)
except Exception:
    # Compute EIR-only inline (Gaussian in frequency)
    Npad  = 2 * Nt
    freqs = fft.fftfreq(Npad, d=dt).astype(np.float32)   # [MHz] = 1/us
    f0_MHz = 8.0
    sigma  = np.sqrt(f0_MHz / np.log(2.0))
    H_eir  = np.exp(-((np.abs(freqs) - f0_MHz)**2) / (2.0 * sigma**2)).astype(np.float32)
    P_true_f   = fft.fft(np.concatenate([pressure_analytic, np.zeros_like(pressure_analytic)], axis=1), n=Npad, axis=1)
    P_eir_f    = P_true_f * H_eir[None, :]
    p_eir_only = np.real(fft.ifft(P_eir_f, n=Npad, axis=1))[:, :Nt].astype(np.float32)
    # SIR+EIR not available in this fallback
    p_sir_eir  = np.zeros_like(pressure_analytic)

# ------------------------
# Common visualization scaling (symmetric about 0)
# ------------------------
def sym_limits(*arrays):
    vmax = max(float(np.max(np.abs(a))) for a in arrays if a is not None)
    return -vmax, vmax

vmin, vmax = sym_limits(pressure_analytic, p_eir_only, p_sir_eir)

# ------------------------
# Plot: top-down (z into the page), x along array
# Each panel: imshow of (Nd × Nt) with axes (x_mm, z_mm)
# ------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
extent = [float(x_mm.min()), float(x_mm.max()), float(z_mm.min()), float(z_mm.max())]

def show_panel(ax, data, title):
    im = ax.imshow(
        data.T,                       # (Nt, Nd) so depth is vertical axis
        extent=extent,                # x along horizontal, z along vertical
        aspect='auto',
        origin='lower',               # depth increases upward from z=0
        cmap='seismic',
        norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
        interpolation='nearest'
    )
    ax.set_title(title)
    ax.set_xlabel("Lateral position x (mm)")
    ax.set_ylabel("Depth z (mm)")
    return im

im0 = show_panel(axes[0], pressure_analytic, "Top-down: Analytical")
im1 = show_panel(axes[1], p_eir_only,        "Top-down: EIR only")
im2 = show_panel(axes[2], p_sir_eir,         "Top-down: SIR + EIR")

# Individual colorbars (helps compare absolute differences)
fig.colorbar(im0, ax=axes[0], label="Pressure (arb.)")
fig.colorbar(im1, ax=axes[1], label="Pressure (arb.)")
fig.colorbar(im2, ax=axes[2], label="Pressure (arb.)")

plt.show()
