import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

uniform = False

B = 1
Cp = 1
c = 1.5 # mm/us

t_min = 0 # us
t_max = 40 # us
Hz_sample_rate = 60 # 1/us
delta_t_sample_rate = 1 / Hz_sample_rate
t_npoints = Hz_sample_rate * t_max # MHz sampling rate times max us

detectors_center = np.array( [0, 0 , 0] )

t_vals = np.linspace(t_min, t_max, t_npoints)

detectors_n = 128
detectors_delta_x = 0.2 # mm spacing per detector

pixel_x = detectors_delta_x
pixel_z = c / Hz_sample_rate

x_bounds = np.arange( 0, detectors_n, detectors_delta_x )
x_bounds -= np.mean(x_bounds)
print(x_bounds)


