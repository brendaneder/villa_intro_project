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

def run_sir_eir(det, t_vals, c, B, Cp, H_eir, patch_xy, det_x, det_y, det_z,
                spheres, Npad, Nt, das_z_values, filename,
                aperture_width_x_mm, aperture_height_y_mm, pitch_x_mm,
                n_div_x, n_div_y, f0_MHz, Nd):



    
    






    Ns = spheres.shape[0]




    # -----------------------------
    # Frequency axis and EIR
    # -----------------------------


    # sigmoid



    # -----------------------------
    # Helper
    # -----------------------------


    # -----------------------------
    # Patch positions (relative to element center)
    # -----------------------------
    # Sub-rectangles are equal-sized, with centers regularly spaced across the aperture.


    # -----------------------------
    # Pre-allocate accumulators
    # -----------------------------
    P_sir_sum_f = np.zeros((Nd, Npad), dtype=np.complex64)  # sum over spheres after SIR (freq domain)

    # -----------------------------
    # Build SIR via patch summation PER SPHERE, then sum spectra
    # -----------------------------
    # Frequency row for broadcasting


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



    # -----------------------------
    # DAS (delay-and-sum) helper with linear interp
    # -----------------------------






    das_analytic = compute_das(pressure_analytic, det, c, t_vals.min(), dt, das_z_values)
    das_eir      = compute_das(p_eir_only,        det, c, t_vals.min(), dt, das_z_values)
    das_sir_eir  = compute_das(p_sir_eir,         det, c, t_vals.min(), dt, das_z_values)

    # -----------------------------
    # Plot the three DAS images in a single window
    # -----------------------------

    # -----------------------------
    # Save arrays for later use
    # -----------------------------
    np.savez(
        filename,
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

