import numpy as np
import matplotlib.pyplot as plt

B = 1 # constant
Cp = 1
c = 1.5 # mm/us
rc = np.array([10, 0, 0]) # mm, object position (centered)
ro = np.array([65, 0, 0]) # mm, dectector position
Rs = 5 # mm, sphere diameter

n_time = 1000

time_array = np.linspace(0,100,n_time)

def pressure_calc(sphere_position, detector_position, radius_sphere, time_array = time_array, B = B, Cp = Cp, c = c):

    pressure_array = np.zeros(len(time_array))

    distance = np.linalg.norm(sphere_position - detector_position)

    for i, t in enumerate(time_array):
        if np.abs(c*t - distance) <= radius_sphere:
            pressure = B * c**2 / (2 * Cp * distance) * (distance - c*t)
            pressure_array[i] = pressure
    
    return pressure_array

# p = pressure_calc(rc, ro, Rs)


# Parameters
Nx, Ny = 8, 16   # number of points along x and y
Ax, Ay = .2, .2   # spacing between points along x and y

# Build centered coordinates
x_vals = np.arange(0, Nx*Ax, Ax)
y_vals = np.arange(0, Ny*Ay, Ay)
x_vals = x_vals - np.mean(x_vals)
y_vals = y_vals - np.mean(y_vals)

z_vals = np.array([0])


# Cartesian product
sensor_locations = np.array([(x, y, z) for x in x_vals for y in y_vals for z in z_vals])

# Find means
x_mean = np.mean(sensor_locations[:, 0])
y_mean = np.mean(sensor_locations[:, 1])
z_mean = np.mean(sensor_locations[:, 2])

# Center grid
sensor_locations = sensor_locations - np.array([x_mean, y_mean, z_mean])
print(sensor_locations)


sensor_and_time_output = np.empty( (len(sensor_locations) , n_time) )

sphere_center = []

for i, sensor in enumerate(sensor_locations):
    
    pressure_array = pressure_calc(rc, sensor, Rs)

    sensor_and_time_output[i] = pressure_array

print(sensor_and_time_output)

for i in range( len( sensor_locations ) ):
    plt.plot(time_array, sensor_and_time_output[i])

plt.show()
