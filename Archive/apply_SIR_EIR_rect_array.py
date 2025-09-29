# apply_SIR_EIR_rect_array.py
# ------------------------------------------------------------
# Adds rectangular Spatial Impulse Response (SIR) to your pipeline,
# then applies EIR in frequency domain. Produces:
#   - original analytical pressure (from Program 1 npz),
#   - EIR-only filtered pressure,
#   - SIR+EIR filtered pressure,
# for all 128 elements (rows) and time samples (cols),
# and a plot for element #63 (1-based).

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt


def sinc(x: np.ndarray) -> np.ndarray:
    # normalized sinc: sinc(x) = sin(x)/x with sinc(0)=1
    out = np.ones_like(x, dtype=np.float64)
    nz = (np.abs(x) > 1e-12)
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def load_program1_npz(path="output_data.npz"):
    data = np.load(path, allow_pickle=True)
    # Required keys from Program 1 (we'll ask you to add detectors, spheres, and c)
    t_vals = data["t_vals"]               # time samples (us)
    pressure_composit = data["pressure_composit"]  # (Nd, Nt)
    dt_us = float(data["dt"])             # us per sample

    # Optional: we try to fetch geometry and constants; if missing, we error with a clear message.
    missing = []
    for k in ["detectors", "spheres", "c"]:
        if k not in data.files:
            missing.append(k)
    if missing:
        raise RuntimeError(
            f"'{path}' is missing required arrays {missing}. "
            "Please modify Program 1 to save 'detectors', 'spheres', and 'c' into output_data.npz."
        )
    detectors = data["detectors"]         # (Nd, 3) mm
    spheres   = data["spheres"]           # (Ns, 4) columns: x,y,z,r  (mm, mm, mm, mm)
    c         = float(data["c"])          # mm/us

    # Optional physical constants (defaults if absent)
    B  = float(data["B"]) if "B" in data.files else 1.0
    Cp = float(data["Cp"]) if "Cp" in data.files else 1.0

    return t_vals, pressure_composit, dt_us, detectors, spheres, c, B, Cp


def compute_per_sphere_traces(t_vals, detectors, spheres, c, B=1.0, Cp=1.0):
    """
    Rebuild the *per-sphere* analytical pressure traces for all detectors,
    matching the piecewise 'N-wave' used in Program 1, so we can apply SIR
    per (detector,sphere) in frequency domain afterwards.
    Returns P (Nd, Ns, Nt).
    """
    Nd = detectors.shape[0]
    Ns = spheres.shape[0]
    Nt = t_vals.size
    dt = float(t_vals[1] - t_vals[0])

    # geometry
    dvec = detectors[:, None, :] - spheres[None, :, :3]        # (Nd, Ns, 3)
    D    = np.linalg.norm(dvec, axis=2)                        # (Nd, Ns)
    R    = spheres[None, :, 3]                                 # (1, Ns) -> (Nd, Ns)

    # piecewise-linear N-wave params (as in Program 1)
    K    = (B * (c**2)) / (2.0 * Cp) / D                       # (Nd, Ns)
    t0   = (D - R) / c
    t1   = (D + R) / c
    A    = K * D
    B_   = K * c

    # Build per-sphere traces (vectorized over detectors; loop over Ns which is small)
    P = np.zeros((Nd, Ns, Nt), dtype=np.float64)
    t0_idx = np.maximum(0, np.ceil((t0 - t_vals[0]) / dt)).astype(np.int32)
    t1_idx = np.minimum(Nt, np.floor((t1 - t_vals[0]) / dt + 1.0)).astype(np.int32)

    t_broadcast = t_vals[None, :]  # (1, Nt)
    for s in range(Ns):
        # active slice for all detectors for this sphere
        i0 = t0_idx[:, s]
        i1 = t1_idx[:, s]
        # build per-detector segments
        for d in range(Nd):
            lo, hi = int(i0[d]), int(i1[d])
            if hi > lo:
                # p(t) = A - B_* t on [lo:hi)
                P[d, s, lo:hi] = (A[d, s] - B_[d, s] * t_broadcast[0, lo:hi])
    return P


def sir_rectangular_farfield(detectors, spheres, c, a_mm=0.2, b_mm=3.0, freqs_MHz=None):
    """
    Compute rectangular SIR in the frequency domain for each (detector,sphere,freq):
        h_s = (a*b)/(2*pi*r) * exp(-i*2*pi*f*r/c) * sinc(pi*f*a*X/(c*r)) * sinc(pi*f*b*Y/(c*r))
    where X,Y are local in-plane coords of the source relative to the element face.
    Here, array elements lie along X, normal is +Z, so local axes are:
        Xhat = (1,0,0), Yhat = (0,1,0), normal = (0,0,1).
    Returns Hs of shape (Nd, Ns, Nf) complex128.
    """
    if freqs_MHz is None:
        raise ValueError("freqs_MHz must be provided")

    Nd = detectors.shape[0]
    Ns = spheres.shape[0]
    Nf = freqs_MHz.size

    # geometry differences: source wrt detector center
    dvec = spheres[None, :, :3] - detectors[:, None, :]    # (Nd, Ns, 3)
    r    = np.linalg.norm(dvec, axis=2)                    # (Nd, Ns)
    X    = dvec[..., 0]                                    # (Nd, Ns) local X
    Y    = dvec[..., 1]                                    # (Nd, Ns) local Y

    # broadcast dimensions
    r_b  = r[:, :, None]               # (Nd, Ns, 1)
    X_b  = X[:, :, None]
    Y_b  = Y[:, :, None]
    f_b  = freqs_MHz[None, None, :]    # (1,1,Nf)

    # constants
    a = float(a_mm)
    b = float(b_mm)
    two_pi = 2.0 * np.pi

    # phase term
    phase = np.exp(-1j * two_pi * f_b * r_b / c)          # f in MHz, c in mm/us -> units consistent

    # sinc args
    argX = np.pi * f_b * (a * X_b) / (c * r_b + 1e-20)
    argY = np.pi * f_b * (b * Y_b) / (c * r_b + 1e-20)

    # assemble
    prefac = (a * b) / (two_pi * (r_b + 1e-20))
    Hs = prefac * phase * sinc(argX) * sinc(argY)          # (Nd, Ns, Nf)
    return Hs


def apply_eir(data_fft, freqs_MHz, f0_MHz=8.0):
    """
    Same EIR model as Program 3: Gaussian around f0.
    """
    sigma = np.sqrt(f0_MHz / np.log(2.0))
    h = np.exp(-(np.abs(freqs_MHz) - f0_MHz)**2 / (2.0 * sigma**2))
    return data_fft * h[None, None, :]   # broadcast over (Nd, Ns, Nf) or (Nd, Nf)


def main(
    npz_path="output_data.npz",
    a_mm=0.2,
    b_mm=3.0,
    element_to_plot_1based=63
):
    # ---- load Program 1 artifacts ----
    (t_vals, pressure_composit, dt_us,
     detectors, spheres, c, B, Cp) = load_program1_npz(npz_path)

    Nd, Nt = pressure_composit.shape
    Ns = spheres.shape[0]

    # ---- rebuild per-sphere time traces: (Nd, Ns, Nt) ----
    P_ns_t = compute_per_sphere_traces(t_vals, detectors, spheres, c, B=B, Cp=Cp)

    # ---- zero-pad and FFT along time ----
    Nt_pad = 2 * Nt
    freqs_MHz = fft.fftfreq(Nt_pad, d=dt_us)   # MHz because t in us
    P_ns_f = fft.fft(P_ns_t, n=Nt_pad, axis=2)  # (Nd, Ns, Nf)

    # ---- rectangular SIR in frequency domain for each (d,s,f) ----
    Hs = sir_rectangular_farfield(detectors, spheres, c, a_mm=a_mm, b_mm=b_mm, freqs_MHz=freqs_MHz)

    # ---- apply SIR (per sphere), then sum over spheres -> (Nd, Nf) ----
    P_summed_f_with_SIR = np.sum(P_ns_f * Hs, axis=1)  # (Nd, Nf)

    # ---- EIR-only branch: apply EIR to *summed* original ----
    P_sum_t = np.sum(P_ns_t, axis=1)                   # (Nd, Nt) == pressure_composit (sanity)
    P_sum_f = fft.fft(P_sum_t, n=Nt_pad, axis=1)       # (Nd, Nf)
    P_sum_f_EIR = apply_eir(P_sum_f, freqs_MHz)        # (Nd, Nf)
    p_eir_time = np.real(fft.ifft(P_sum_f_EIR, n=Nt_pad, axis=1))[:, :Nt]  # (Nd, Nt)

    # ---- SIR + EIR branch ----
    P_with_SIR_EIR = apply_eir(P_summed_f_with_SIR, freqs_MHz)             # (Nd, Nf)
    p_sir_eir_time = np.real(fft.ifft(P_with_SIR_EIR, n=Nt_pad, axis=1))[:, :Nt]

    # ---- Save outputs ----
    out = {
        "t_vals": t_vals,
        "pressure_original": pressure_composit,  # from Program 1
        "pressure_eir": p_eir_time,
        "pressure_sir_eir": p_sir_eir_time,
        "dt": dt_us,
        "detectors": detectors,
        "spheres": spheres,
        "c": c,
        "a_mm": float(a_mm),
        "b_mm": float(b_mm),
        "freqs_MHz": freqs_MHz
    }
    np.savez("output_with_SIR.npz", **out)

    # ---- Plot one element (1-based index from request) ----
    idx = int(element_to_plot_1based) - 1
    idx = max(0, min(idx, Nd-1))

    plt.figure(figsize=(9,5))
    plt.plot(t_vals, pressure_composit[idx, :], label="Analytical (original)", linewidth=1.5)
    plt.plot(t_vals, p_eir_time[idx, :], label="EIR only", linewidth=1.2)
    plt.plot(t_vals, p_sir_eir_time[idx, :], label="SIR + EIR", linewidth=1.2)
    plt.xlabel("Time (µs)")
    plt.ylabel("Pressure (arb. units)")
    plt.title(f"Transducer element #{element_to_plot_1based} (a={a_mm} mm, b={b_mm} mm)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Default: 0.2 mm × 3.0 mm element, as requested; easy to change via CLI edit if needed.
    main(npz_path="output_data.npz", a_mm=0.2, b_mm=3.0, element_to_plot_1based=63)
