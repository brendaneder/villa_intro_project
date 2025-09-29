import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

uniform = False

B = 1
Cp = 1
c = 1.5 # mm/us

t_min = 0
t_max = 40

sample_rate = 40.0    # Hz, times per us  (== samples/us)
dt = 1.0 / sample_rate

t_vals = np.arange(t_min, t_max, dt, dtype=np.float32)
t_npoints = t_vals.size

detectors_center = np.array( [0, 0 , 0] )

t_vals = np.linspace(t_min, t_max, t_npoints)

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


import matplotlib.pyplot as plt
import numpy as np

# first subplot

plt.plot(t_vals, np.average(pressure_composit, axis=0))

np.save('analytical_pressure_matrix', pressure_composit)

plt.imshow(pressure_composit)

plt.show()


np.savez("output_data.npz", t_vals=t_vals, pressure_composit=pressure_composit, das_img=das_img, dt=dt)
print("Saved: output_data.npz")




