
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from typing import Tuple, Optional

###############################################################################
# Config (edit these defaults if needed)
###############################################################################

# Rectangular element geometry (mm)
ELEMENT_WIDTH_X_MM = 0.2           # along X; center-to-center pitch is also 0.2 mm per your setup
ELEMENT_HEIGHT_Y_MM = 3.0          # along Y; make this adjustable

# Array
NUM_ELEMENTS = 128                 # along X only (row of elements)
PITCH_X_MM   = 0.2                 # spacing along X (mm)

# EIR model (same semantics as your existing "2_EIS_Fourier_Transform_Final.py")
# Gaussian band-pass-like envelope around f0 (MHz) with width controlled by sigma.
EIR_FC_MHZ = 8.0

# Plot/Output
MAKE_PLOT = True
PLOT_ELEMENT_INDEX = 63            # zero-based index (0..127)
OUTPUT_NPZ = "sir_eir_output.npz"


###############################################################################
# Utility: safe sinc (numpy's normalized sinc uses pi inside; we want sin(x)/x)
###############################################################################
def sinc_unorm(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x, dtype=np.float64)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


###############################################################################
# Core: far-field rectangular SIR for a single (q, n) pair across frequencies.
# Implements Eq. (10) form and normalizes DC gain to 1 to preserve scaling.
###############################################################################
def Hs_rect_far_field(
    freqs_MHz: np.ndarray,
    rnq_mm: float,
    Xnq_mm: float,
    Ynq_mm: float,
    a_mm: float,
    b_mm: float,
    c_mm_per_us: float,
) -> np.ndarray:
    """
    freqs_MHz : array of frequency samples in MHz (matches FFT bins for 1/us units)
    rnq_mm    : distance from sphere center n to element q center (mm)
    Xnq_mm    : local X coordinate of source n w.r.t element q center (mm)
    Ynq_mm    : local Y coordinate of source n w.r.t element q center (mm)
    a_mm, b_mm: element dimensions (mm) — (width_x, height_y)
    c_mm_per_us: speed of sound (mm/us)
    """
    # Convert to angular arguments used in the sinc terms
    # x_arg = pi * f * a * X / (c * r), y_arg = pi * f * b * Y / (c * r)
    f = freqs_MHz.astype(np.float64)  # MHz
    # Phase term exp(-i 2 pi f rnq / c) with f in MHz, c in mm/us, rnq in mm
    phase = np.exp(-1j * 2.0 * np.pi * (f * rnq_mm) / c_mm_per_us)

    # Sinc arguments (dimensionless); adopt radians for unnormalized sinc
    x_arg = np.pi * f * a_mm * (Xnq_mm / rnq_mm) / c_mm_per_us
    y_arg = np.pi * f * b_mm * (Ynq_mm / rnq_mm) / c_mm_per_us

    Sx = sinc_unorm(x_arg)
    Sy = sinc_unorm(y_arg)

    # Far-field magnitude factor (ab / (2 pi rnq)); we normalize so DC gain = 1.
    dc_gain = (a_mm * b_mm) / (2.0 * np.pi * rnq_mm)

    # Full Hs from the paper (Eq. 10), then normalize by DC gain so Hs(0)=1.
    Hs = (a_mm * b_mm) * phase * Sx * Sy / (2.0 * np.pi * rnq_mm)
    Hs_norm = Hs / dc_gain
    return Hs_norm


###############################################################################
# EIR builder: matches your existing EIR style (Gaussian envelope around f0)
###############################################################################
def build_EIR(freqs_MHz: np.ndarray, f0_MHz: float) -> np.ndarray:
    # Keep the same derivation you used: sigma chosen from f0/log(2)
    sigma = np.sqrt(f0_MHz / np.log(2.0))
    H = np.exp(-(np.abs(freqs_MHz) - f0_MHz) ** 2 / (2.0 * sigma ** 2))
    return H


###############################################################################
# Rebuild analytical N-wave per-sphere (identical math you used originally)
# Returns p_true_per_sphere: shape (Nd, Ns, Nt)
###############################################################################
def synthesize_true_pressure_per_sphere(
    detectors_mm: np.ndarray,   # (Nd, 3)
    spheres: np.ndarray,        # (Ns, 4) columns: x,y,z,r (mm)
    t_vals_us: np.ndarray,      # (Nt,)
    c_mm_per_us: float,
    B: float,
    Cp: float,
) -> np.ndarray:
    Nd = detectors_mm.shape[0]
    Ns = spheres.shape[0]
    Nt = t_vals_us.size
    # Distances D (Nd, Ns)
    D = np.linalg.norm(detectors_mm[:, None, :] - spheres[None, :, :3], axis=2).astype(np.float64)
    R = spheres[None, :, 3].astype(np.float64)  # (1, Ns) broadcast to (Nd, Ns)
    K = (B * (c_mm_per_us ** 2)) / (2.0 * Cp) / D

    # Active window per (det, sphere)
    t0 = (D - R) / c_mm_per_us
    t1 = (D + R) / c_mm_per_us

    dt = float(t_vals_us[1] - t_vals_us[0])
    Nt1 = Nt + 1

    # Linear coefficients: p(t) = A - B_* t on active interval
    A = (K * D)      # (Nd, Ns)
    B_ = (K * c_mm_per_us)

    # Build via difference arrays per sphere so we can keep sphere separation
    p_true = np.zeros((Nd, Ns, Nt), dtype=np.float64)

    rows = np.arange(Nd, dtype=np.int32)[:, None]
    cols = np.arange(Ns, dtype=np.int32)[None, :]

    # We'll loop Ns if memory is a concern; Ns is typically small (few spheres)
    for n in range(Ns):
        addA = np.zeros((Nd, Nt1), dtype=np.float64)
        addB = np.zeros((Nd, Nt1), dtype=np.float64)

        i0 = np.maximum(0, np.ceil((t0[:, n] - t_vals_us[0]) / dt)).astype(np.int32)
        i1 = np.minimum(Nt, np.floor((t1[:, n] - t_vals_us[0]) / dt + 1.0)).astype(np.int32)

        # scatter-add starts/ends
        np.add.at(addA, (rows[:, 0], i0),  A[:, n])
        np.add.at(addB, (rows[:, 0], i0),  B_[:, n])
        np.add.at(addA, (rows[:, 0], i1), -A[:, n])
        np.add.at(addB, (rows[:, 0], i1), -B_[:, n])

        A_run = np.cumsum(addA, axis=1)[:, :Nt]
        B_run = np.cumsum(addB, axis=1)[:, :Nt]
        p_true[:, n, :] = A_run - B_run * t_vals_us[None, :]

    return p_true  # (Nd, Ns, Nt)


###############################################################################
# Compute SIR-modulated pressure in frequency domain (per-sphere, then sum)
###############################################################################
def apply_SIR_far_field(
    p_true_per_sphere: np.ndarray,  # (Nd, Ns, Nt)
    detectors_mm: np.ndarray,       # (Nd, 3)
    spheres: np.ndarray,            # (Ns, 4)
    dt_us: float,
    c_mm_per_us: float,
    a_mm: float,
    b_mm: float,
    patch_m: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (p_sir_time, freqs_MHz)
    p_sir_time has shape (Nd, Nt) after summation over spheres.
    If patch_m > 1, uses the "patch approximation" averaging m x m per element.
    """
    Nd, Ns, Nt = p_true_per_sphere.shape
    # Frequency axis (FFT)
    n_fft = 2 * Nt
    freqs = fft.fftfreq(n_fft, d=dt_us)  # in cycles per microsecond == MHz
    freqs_MHz = freqs.astype(np.float64)

    # Pre-FFT (Nd, Ns, n_fft)
    pad = np.zeros((Nd, Ns, Nt), dtype=np.float64)
    p_pad = np.concatenate([p_true_per_sphere, pad], axis=2)
    P_f = fft.fft(p_pad, n=n_fft, axis=2)

    # Geometry helpers
    # Local element frames: X along array (global X), Y vertical (global Y)
    elem_centers = detectors_mm[:, :2]  # (Nd, 2) XY
    elem_z = detectors_mm[:, 2]         # (Nd,) (should be ~0)

    # Prepare accumulation for SIR-modulated spectrum
    P_sir = np.zeros((Nd, n_fft), dtype=np.complex128)

    # Optional patch division (m x m); patch centers grid offsets
    if patch_m < 1:
        patch_m = 1
    ax = a_mm / patch_m
    by = b_mm / patch_m
    # Patch center offsets in local (X,Y): symmetric grid
    offs_x = (np.arange(patch_m) + 0.5) * ax - a_mm / 2.0
    offs_y = (np.arange(patch_m) + 0.5) * by - b_mm / 2.0
    patch_offsets = np.array(np.meshgrid(offs_x, offs_y, indexing="xy")).reshape(2, -1).T  # (m^2, 2)

    # Loop over spheres (Ns is usually small)
    for n in range(Ns):
        # Vector from each element center to this sphere center
        dvec = spheres[n, :3][None, :] - detectors_mm  # (Nd, 3)
        rnq = np.linalg.norm(dvec, axis=1)             # (Nd,)

        # Project into local element (X,Y) coordinates; here we assume elements are aligned to global axes.
        Xnq = dvec[:, 0]   # local X
        Ynq = dvec[:, 1]   # local Y

        # Build Hs for each detector across frequency
        # If using patches, average Hs over patches shifted by patch_offsets.
        Hs_all = np.zeros((Nd, n_fft), dtype=np.complex128)
        for (dx, dy) in patch_offsets:
            # Patch center shifts element center -> effectively shifts (Xnq,Ynq)
            Hs = Hs_rect_far_field(
                freqs_MHz=freqs_MHz,
                rnq_mm=rnq[:, None] if np.ndim(rnq) == 1 else rnq,
                Xnq_mm=(Xnq - dx)[:, None],
                Ynq_mm=(Ynq - dy)[:, None],
                a_mm=ax,
                b_mm=by,
                c_mm_per_us=c_mm_per_us,
            )
            # Hs currently shape (Nd, n_fft) due to broadcasting
            if Hs.ndim == 1:
                Hs = np.broadcast_to(Hs[None, :], (Nd, n_fft))
            Hs_all += Hs
        Hs_all /= patch_m * patch_m  # average over m^2 patches

        # Multiply this sphere's spectrum and accumulate
        P_sir += Hs_all * P_f[:, n, :]

    # IFFT and crop
    p_sir_time = np.real(fft.ifft(P_sir, n=n_fft, axis=1))[:, :Nt]
    return p_sir_time, freqs_MHz


###############################################################################
# Driver
###############################################################################
def run_pipeline(
    npz_path: str = "output_data.npz",
    element_height_mm: Optional[float] = None,
    patch_m: int = 1,
    make_plot: bool = True,
    plot_element_index: int = 63,
    out_npz: str = OUTPUT_NPZ,
):
    data = np.load(npz_path, allow_pickle=True)
    t_vals = data["t_vals"].astype(float)                 # us
    pressure_true_sum = data["pressure_composit"].astype(float)  # (Nd, Nt)
    detectors = data["detectors"].astype(float)           # (Nd, 3)
    spheres   = data["spheres"].astype(float)             # (Ns, 4)
    c         = float(data["c"])                          # mm/us
    B         = float(data.get("B", 1.0))
    Cp        = float(data.get("Cp", 1.0))

    dt_us = float(t_vals[1] - t_vals[0])
    Nd, Nt = pressure_true_sum.shape
    Ns = spheres.shape[0]

    # Ensure geometry matches expectation
    a_mm = ELEMENT_WIDTH_X_MM
    b_mm = element_height_mm if element_height_mm is not None else ELEMENT_HEIGHT_Y_MM

    # 1) Re-synthesize per-sphere pressure traces so we can apply a sphere-dependent SIR
    p_true_per_sphere = synthesize_true_pressure_per_sphere(
        detectors_mm=detectors,
        spheres=spheres,
        t_vals_us=t_vals,
        c_mm_per_us=c,
        B=B,
        Cp=Cp,
    )  # (Nd, Ns, Nt)

    # Sanity: summed should closely match provided aggregate trace
    # (Minor numeric differences expected; you can assert if desired.)

    # 2) EIR-only (on summed trace), in freq domain
    n_fft = 2 * Nt
    freqs = fft.fftfreq(n_fft, d=dt_us)          # MHz
    H_eir = build_EIR(freqs, f0_MHz=EIR_FC_MHZ)  # (n_fft,)
    P_sum = fft.fft(np.concatenate([pressure_true_sum, np.zeros_like(pressure_true_sum)], axis=1), n=n_fft, axis=1)
    P_eir = (P_sum * H_eir.reshape(1, -1))
    p_eir_only = np.real(fft.ifft(P_eir, n=n_fft, axis=1))[:, :Nt]  # (Nd, Nt)

    # 3) Apply SIR (per sphere) then EIR
    p_sir_time, freqs_MHz = apply_SIR_far_field(
        p_true_per_sphere=p_true_per_sphere,
        detectors_mm=detectors,
        spheres=spheres,
        dt_us=dt_us,
        c_mm_per_us=c,
        a_mm=a_mm,
        b_mm=b_mm,
        patch_m=patch_m,
    )  # (Nd, Nt)

    # EIR on top of SIR
    P_sir = fft.fft(np.concatenate([p_sir_time, np.zeros_like(p_sir_time)], axis=1), n=n_fft, axis=1)
    P_sir_eir = P_sir * H_eir.reshape(1, -1)
    p_sir_eir = np.real(fft.ifft(P_sir_eir, n=n_fft, axis=1))[:, :Nt]

    # Save matrices
    np.savez(
        out_npz,
        t_vals=t_vals,
        pressure_true=pressure_true_sum,   # Nd x Nt
        pressure_eir=p_eir_only,           # Nd x Nt
        pressure_eir_sir=p_sir_eir,        # Nd x Nt
        dt=dt_us,
        element_height_mm=b_mm,
        element_width_mm=a_mm,
        patch_m=patch_m,
        freqs_MHz=freqs,
    )

    if make_plot:
        idx = int(plot_element_index)
        if not (0 <= idx < Nd):
            idx = Nd // 2
        plt.figure(figsize=(9, 5))
        plt.plot(t_vals, pressure_true_sum[idx, :], label="True analytical (no EIR/SIR)", linewidth=1.25)
        plt.plot(t_vals, p_eir_only[idx, :], label="EIR only", linewidth=1.25)
        plt.plot(t_vals, p_sir_eir[idx, :], label="EIR + SIR", linewidth=1.25)
        plt.title(f"Transducer element {idx} (0-based)")
        plt.xlabel("Time (µs)")
        plt.ylabel("Pressure (arb. units)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return out_npz


if __name__ == "__main__":
    # Default run; will look for output_data.npz produced by your analytical script.
    try:
        out = run_pipeline(
            npz_path="output_data.npz",
            element_height_mm=ELEMENT_HEIGHT_Y_MM,
            patch_m=1,                  # set to 2 for the paper's 2x2 patch averaging if near-field
            make_plot=MAKE_PLOT,
            plot_element_index=PLOT_ELEMENT_INDEX,
            out_npz=OUTPUT_NPZ,
        )
        print(f"Wrote: {out}")
    except FileNotFoundError:
        # If output_data.npz isn't present, provide a tiny self-test scene.
        print("output_data.npz not found; generating a tiny test scene...")
        # Build a minimal scene consistent with your geometry
        c = 1.5  # mm/us
        dt = 1.0 / 40.0
        t_vals = np.arange(0.0, 40.0, dt, dtype=np.float64)

        # 128 elements along X, centered at 0, z=0
        x = (np.arange(NUM_ELEMENTS, dtype=np.float64) - (NUM_ELEMENTS - 1) / 2.0) * PITCH_X_MM
        detectors = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

        # One sphere at (0,0,-20) r=1.0 mm
        spheres = np.array([[0.0, 0.0, -20.0, 1.0]], dtype=np.float64)

        # Synthesize and save to output_data.npz for the normal path
        B, Cp = 1.0, 1.0
        Nd, Ns = detectors.shape[0], spheres.shape[0]
        p_true_per_sphere = synthesize_true_pressure_per_sphere(detectors, spheres, t_vals, c_mm_per_us=c, B=B, Cp=Cp)
        p_sum = p_true_per_sphere.sum(axis=1)

        np.savez("output_data.npz",
                 t_vals=t_vals,
                 pressure_composit=p_sum,
                 das_img=np.zeros((Nd, 10)),
                 dt=dt,
                 detectors=detectors,
                 spheres=spheres,
                 c=c,
                 B=B,
                 Cp=Cp)

        # Run the normal path
        out = run_pipeline(
            npz_path="output_data.npz",
            element_height_mm=ELEMENT_HEIGHT_Y_MM,
            patch_m=1,
            make_plot=MAKE_PLOT,
            plot_element_index=PLOT_ELEMENT_INDEX,
            out_npz=OUTPUT_NPZ,
        )
        print(f"Wrote: {out}")
