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
detectors = data["detectors"]

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
h = np.exp( -1 * ( ( np.abs( freqs ) - f0 ) ** 2 ) / ( 2 * sigma ** 2 ) )

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






