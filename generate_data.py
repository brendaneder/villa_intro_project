from pathlib import Path
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time


np.random.seed(42)

starting_run = 45
n_runs = 500

start = time.time()

plt_trace = 0
plt_das = 0
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
        for i in range(n_spheres-1):
            for j in range(i+1, n_spheres):

                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dz = z[i] - z[j]

                true_distance = np.linalg.norm((dx, dy, dz))
                min_distance = r[i] + r[j]

                
                if true_distance <= min_distance:
                    print("Real distance {:.2f} < {:.2f} min distance".format(true_distance, min_distance))
                    make_spheres(n_spheres,
                    x_bounds,
                    y_bounds,
                    z_bounds,
                    r_bounds)
    spheres = np.column_stack([x, y, z, r]).astype(np.float32)
    print(spheres)
    return spheres



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
Ns = n_spheres
Nt = t_npoints

det      = detectors          # (Nd,3)
Nd = det.shape[0]
Nt = int(t_vals.size)


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

Npad  = 2 * Nt
freqs = fft.fftfreq(Npad, d=dt).astype(np.float32)  # [MHz] = 1/us


def H_function(x, k = 9, a0 = .5, b0 = 9):
    sig_a = 1 / ( 1 + np.exp( -k * (np.abs(x) - a0)))
    sig_b = 1 / ( 1 + np.exp( -k * (np.abs(x) - b0)))

    scale = np.abs( sig_a - sig_b )
    return scale

H_eir = H_function(freqs).astype(np.float32)



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

ax = float(aperture_width_x_mm)
ay = float(aperture_height_y_mm)

# Create n_div centers in each axis spanning [-ax/2, ax/2] and [-ay/2, ay/2]
# Using "pixel center" convention (no patch exactly on the edge)
x_centers = (np.arange(n_div_x, dtype=np.float32) + 0.5) / n_div_x * ax - ax / 2.0  # (n_div,)
y_centers = (np.arange(n_div_y, dtype=np.float32) + 0.5) / n_div_y * ay - ay / 2.0  # (n_div,)
XX, YY    = np.meshgrid(x_centers, y_centers, indexing='xy')                    # (n_div,n_div)
patch_xy  = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)       # (n_patch,2)
n_patch   = int(patch_xy.shape[0])

f_row = freqs[None, :]  # (1,Npad)

# Detector centers
det_x = det[:, 0][:, None]  # (Nd,1)
det_y = det[:, 1][:, None]
det_z = det[:, 2][:, None]  # (Nd,1)

elt_plot = 62 if Nd >= 63 else Nd - 1


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


folder = Path("root/training_data")
folder.mkdir(parents=True, exist_ok=True)





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


i = starting_run
while i < n_runs:
    if i % 10 == 0:
        end = time.time()
        print("Execution time:", end - start, "seconds")
        start = time.time()
    
    filename = folder / f"sir_eir_outputs_{i}.npz"
    spheres = make_spheres(n_spheres)
    run_sir_eir(det, t_vals, c, B, Cp, H_eir,
                patch_xy, det_x, det_y, det_z,
                spheres, Npad, Nt, das_z_values, filename,
                aperture_width_x_mm, aperture_height_y_mm,
                pitch_x_mm, n_div_x, n_div_y, f0_MHz, Nd)
    print(i)


    i+=1