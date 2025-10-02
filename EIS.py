import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import matplotlib.colors as mcolors

import time
start = time.time()

data = np.load("output_data.npz")
t_vals = data["t_vals"]               # microseconds
pressure_composit = data["pressure_composit"]
delay_and_sum_data = data["das_img"]  # shape: [scanline/angle, depth(time)]
dt_us = float(data["dt"])             # microseconds between time samples

plt_heatmaps = False

data_to_transform = pressure_composit




n_time = len(t_vals)

# pad the data
pad = np.zeros_like(data_to_transform)

padded_data = np.concatenate( ( data_to_transform , pad ) , axis = 1 )

len_padded_data = padded_data.shape[1]


# solve for sigma 
f0 = 8
sigma = np.sqrt( f0 / np.log( 2 ) ) 

# create frequeny bins
freqs = fft.fftfreq( 2 * n_time, d = dt_us)

# Create H, the mdoulation based on EIR
def H_function(x, k = 9, a0 = 0.5, b0 = 9):
    sig_a = 1 / ( 1 + np.exp( -k * (np.abs(x) - a0)))
    sig_b = 1 / ( 1 + np.exp( -k * (np.abs(x) - b0)))

    scale = np.abs( sig_a - sig_b )
    return scale

h = H_function(freqs).astype(np.float32)

# perform the FFT
data_fft = fft.fft( padded_data , n = len_padded_data , axis = 1 )

# convert H into a 1 x pad_len array, apply h
H_broadcast = h.reshape(1 , -1)
data_fft_eir = data_fft * h

# do inverse fft to convert back to spacial/time domain
data_eir_spacial = fft.ifft( data_fft_eir, n = len_padded_data, axis = 1)
data_eir_spacial_real = np.real( data_eir_spacial )

# cut off extra padded 0's
data_eir_final = data_eir_spacial_real[:, :n_time]
transducer_element = 64

plt.plot(t_vals, data_to_transform[transducer_element, :], label = 'Analytical Solution')
plt.plot(t_vals , data_eir_final[transducer_element, :], label = 'EIR Applied')
plt.xlabel('z-axis depth, mm')
plt.ylabel('Pressure amplitude, relative units')
plt.title('Pressure trace of a single transducer, analytical solution, {} spheres, {} radius'.format('5', '1mm'))
plt.show()


# ======================================================
# ADDED: Compute & plot Delay-and-Sum (DAS) without/with EIS
#        making 0 pressure map to white (centered colormap)
#        to mirror the style in 2_Analytical_Solution_Final.py
# ======================================================

    
if plt_heatmaps:
    # Expect these from output_data.npz produced by 2_Analytical_Solution_Final.py
    detectors = data["detectors"]      # (Nd, 3)
    c         = float(data["c"])       # speed [mm/us]
    dt        = float(data["dt"])      # time step [us]
    # pressure_composit already loaded above (raw). data_eir_final computed above (EIS-applied).

    # DAS grid: reuse detector x-positions as lateral axis; z from -35..0 at c*dt spacing
    das_x_values = detectors[:, 0].astype(np.float32)
    z_resolution = dt * c
    das_z_values = np.arange(-35.0, 0.0, z_resolution, dtype=np.float32)

    # Helper: vectorized DAS with linear interpolation from pressure traces
    def compute_das(pressure_traces, detectors, c, t_min, dt, z_vals):
        import numpy as _np
        Nd, Nt = pressure_traces.shape
        # Pixel grid (Nx, Nz)
        Xpix, Zpix = _np.meshgrid(detectors[:, 0].astype(_np.float32), z_vals.astype(_np.float32), indexing='ij')
        Nx, Nz = Xpix.shape
        Npix = Nx * Nz

        pixels = _np.column_stack((
            Xpix.ravel(),
            _np.zeros(Npix, dtype=_np.float32),
            Zpix.ravel()
        )).astype(_np.float32)  # (Npix, 3)

        # Distances (Npix, Nd)
        diff = pixels[:, None, :] - detectors[None, :, :]
        dist = _np.linalg.norm(diff, axis=2).astype(_np.float32)

        # Fractional sample index per (pixel, detector)
        f = (dist / c - t_min) / dt
        f = _np.clip(f, 0.0, Nt - 1.000001).astype(_np.float32)
        i0 = _np.floor(f).astype(_np.int32)
        i1 = i0 + 1
        w  = (f - i0).astype(_np.float32)

        det_idx = _np.broadcast_to(_np.arange(Nd, dtype=_np.int32)[None, :], (Npix, Nd))
        P0 = pressure_traces[det_idx, i0]
        P1 = pressure_traces[det_idx, i1]
        vals = (1.0 - w) * P0 + w * P1
        das_flat = vals.sum(axis=1, dtype=_np.float32)
        return das_flat.reshape(Nx, Nz)

    # Build both DAS images
    das_no_eis  = compute_das(pressure_composit, detectors, c, t_vals.min(), dt_us, das_z_values)
    das_with_eis = compute_das(data_eir_final,  detectors, c, t_vals.min(), dt_us, das_z_values)

    # Plot side-by-side with white at 0
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Common extent
    extent = [das_x_values.min(), das_x_values.max(), -35.0, 0.0]

    imL = axes2[0].imshow(
        das_no_eis.T,             # depth is vertical
        extent=extent,
        aspect='auto',
        cmap='seismic',
        origin='upper',
        norm=mcolors.TwoSlopeNorm(vmin=das_no_eis.min(), vcenter=0, vmax=das_no_eis.max())
    )
    axes2[0].set_title("DAS (no EIS)")
    axes2[0].set_xlabel("x (mm)")
    axes2[0].set_ylabel("Depth (mm)")
    fig2.colorbar(imL, ax=axes2[0], label="DAS amplitude")

    imR = axes2[1].imshow(
        das_with_eis.T,
        extent=extent,
        aspect='auto',
        cmap='seismic',
        origin='upper',
        norm=mcolors.TwoSlopeNorm(vmin=das_with_eis.min(), vcenter=0, vmax=das_with_eis.max())
    )
    axes2[1].set_title("DAS (with EIS)")
    axes2[1].set_xlabel("x (mm)")
    axes2[1].set_ylabel("Depth (mm)")
    fig2.colorbar(imR, ax=axes2[1], label="DAS amplitude")

    plt.show()

