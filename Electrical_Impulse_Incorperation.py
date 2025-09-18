import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft

# ----- Load -----
data = np.load("output_data.npz")
t_vals = data["t_vals"]               # microseconds
pressure_composit = data["pressure_composit"]
delay_and_sum_data = data["das_img"]  # shape: [scanline/angle, depth(time)]
dt_us = float(data["dt"])             # microseconds between time samples

# ----- Choose axis to filter along -----
# Assumption: columns are time/depth (closest -> furthest), so axis=1
time_axis = 1
n_time = delay_and_sum_data.shape[time_axis]

# ----- Zero-padding length (next power of 2, ~2x for nicer FFT grid) -----
def nextpow2(n): 
    return 1 << (int(n - 1).bit_length())

pad_len = nextpow2(2 * n_time)

# ----- Frequency vector in MHz (because d=dt_us where dt is in microseconds) -----
freqs_mhz = fft.fftfreq(pad_len, d=dt_us)  # cycles per microsecond = MHz

# ----- Electrical impulse response: Gaussian band centered at 8 MHz -----
f0 = 8.0         # MHz
# sigma from constraint H(4 or 12 MHz) = 0.5
sigma = np.sqrt(8.0 / np.log(2.0))   # MHz
H = np.exp(-((np.abs(freqs_mhz) - f0)**2) / (2.0 * sigma**2))  # shape [pad_len]

# ----- FFT along time axis with zero-padding -----
# pad only on the "end" of time axis to keep alignment with "near" at index 0
pad_width = [(0,0)] * delay_and_sum_data.ndim
pad_width[time_axis] = (0, pad_len - n_time)
data_padded = np.pad(delay_and_sum_data, pad_width, mode='constant')

# FFT
DATA_F = fft.fft(data_padded, n=pad_len, axis=time_axis)

# ----- Apply impulse response (broadcast across non-time dims) -----
# Reshape H for broadcasting along the chosen axis
shape = [1] * delay_and_sum_data.ndim
shape[time_axis] = -1
H_broadcast = H.reshape(shape)

DATA_F_filt = DATA_F * H_broadcast

# ----- Inverse FFT and crop back to original length -----
data_filt_padded = fft.ifft(DATA_F_filt, n=pad_len, axis=time_axis).real
# crop
slices = [slice(None)] * delay_and_sum_data.ndim
slices[time_axis] = slice(0, n_time)
data_filt = data_filt_padded[tuple(slices)]

# ----- Plot original vs filtered -----
fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

im0 = axs[0].imshow(delay_and_sum_data, aspect='auto', origin='upper')
axs[0].set_title('Original DAS')
axs[0].set_xlabel('z-axis depth (pixels)')
axs[0].set_ylabel('Transducer Element')
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(data_filt, aspect='auto', origin='upper')
axs[1].set_title(' Electrical Impulse')
axs[1].set_xlabel('z-axis depth (pixels)')
axs[1].set_ylabel('Transducer Element')
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

plt.show()
