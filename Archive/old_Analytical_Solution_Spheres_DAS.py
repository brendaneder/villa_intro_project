import numpy as np
import matplotlib.pyplot as plt

import time
start = time.time()


np.random.seed(42)

uniform = False

B = 1
Cp = 1
c = 1.5 # mm/us

t_min = 0
t_max = 40
sample_rate = 60 # Hz, times per us
dt = 1 / sample_rate # us


detectors_center = np.array( [0, 0 , 0] )

t_vals = np.arange(t_min, t_max, dt)
t_npoints = len(t_vals)

def make_spheres(n_spheres = 1, x_bounds = [-15, 15], y_bounds = [-15, 15], z_bounds = [-35, -5], r_bounds = [.5, 2]):
    spheres = np.zeros( (n_spheres, 4 ) )
    for i in range(n_spheres):
        x = np.random.uniform( *x_bounds )
        y = np.random.uniform( *y_bounds )
        z = np.random.uniform( *z_bounds )
        r = np.random.uniform( *r_bounds )

        sphere = np.array( [x, y, z, r] )

        spheres[i] = sphere

    return spheres



if uniform == True:
    spheres = np.zeros( (5, 4) )

    for i in range(5):
        spheres[i] = np.array( [0, 0, -30 + 5*i, 1])

else:
    spheres = make_spheres(n_spheres = 6)







x_npoints_detector, y_npoints_detector = 128, 1 # number of points
x_spacing_detector, y_spacing_detector = .2, .2 # mm

detectors = np.zeros((x_npoints_detector * y_npoints_detector , 3))

for i in range(x_npoints_detector):
    for j in range(y_npoints_detector):
        x = i * x_spacing_detector
        y = j * y_spacing_detector
        z = 0
        coordinates = np.array( [x, y, z] )
        index = i * y_npoints_detector + j
        detectors[index] = coordinates

detectors_mean = np.mean(detectors, axis = 0)
detectors -= detectors_mean
detectors += detectors_center



def pressure_single_sphere(detector, sphere, t_vals = t_vals, B = B, Cp = Cp, c = c):
    
    sphere_center = sphere[:3] # sphere position

    Rs = sphere[-1] # sphere radius
    
    d = np.linalg.norm( detector - sphere_center )

    pressure_vals = np.zeros( len(t_vals) )

    for i, t in enumerate( t_vals ):
        if np.abs( c*t - d ) <= Rs:
            pressure = ( ( B * c**2 ) / (2 * Cp * d) ) * (d - c * t)
            pressure_vals[i] = pressure

    return pressure_vals
 
pressure_composit = np.zeros(( len(detectors), t_npoints ))

for i, detector in enumerate(detectors):
    p = np.zeros( t_npoints )

    for sphere in spheres:

        p += pressure_single_sphere( detector, sphere )
    

    pressure_composit[i] = p









# Implement DAS method here



das_x_values = detectors[:,0] # x-values of pixels
z_resolution = ( 1 / sample_rate ) * c   
das_z_values = np.arange(-35, 0, z_resolution)

das_img = np.zeros( ( len(das_x_values) , len(das_z_values) ))

das_pixels = np.zeros((len(das_x_values) * len(das_z_values), 3))

for i, x_val in enumerate(das_x_values):
    for j, z_val in enumerate(das_z_values):
        
        pixel = np.array( [ x_val , 0 , z_val ])
        das_sum_value = 0

        for di, detector in enumerate( detectors ):

            distance = abs( np.linalg.norm( pixel - detector ) )
            time_delay = distance / c
            p_trace = pressure_composit[di]
            p_val = np.interp( time_delay, t_vals, p_trace , left = 0, right = 0)

            das_sum_value += p_val


        das_img[i, j] = das_sum_value



end = time.time()
print(f"Execution time: {end - start:.4f} seconds")


# create the images


# Convert time to depth in mm
depth_vals = c * t_vals   # mm

fig, axes = plt.subplots(1, 2, figsize=(14,6), constrained_layout=True)

# --- Left: pressure traces ---
im0 = axes[0].imshow(
    pressure_composit.T,   # transpose so depth is vertical
    extent=[0, pressure_composit.shape[0], depth_vals.min(), depth_vals.max()],
    aspect='auto',
    origin='upper',        # 0 mm at top, depth increases downward
    cmap='viridis'
)
axes[0].set_xlabel("Detector index")
axes[0].set_ylabel("Depth (mm)")
axes[0].set_title("Pressure traces")
fig.colorbar(im0, ax=axes[0], label="Pressure")

# --- Right: DAS image ---
im1 = axes[1].imshow(
    das_img.T,
    extent=[das_x_values.min(), das_x_values.max(),
            das_z_values.min(), das_z_values.max()],
    aspect='auto',
    origin='upper',        # 0 mm at top, depth increases downward
    cmap='viridis'
)
axes[1].set_xlabel("x (mm)")
axes[1].set_ylabel("Depth (mm)")
axes[1].set_title("DAS image")
fig.colorbar(im1, ax=axes[1], label="DAS amplitude")

plt.show()





