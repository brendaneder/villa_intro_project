import numpy as np
import matplotlib.pyplot as plt

B = 1 # constant
Ao = 1 # constant
c = 1.5 # mm/us
Cp = 1 # constant
R = 5 # mm
a = 1

n_points = 1000
t_start = 0
t_end = 10 # #us

t_points = np.linspace(t_start, t_end, n_points)
p_values = np.zeros(n_points)

for i, t in enumerate(t_points):

    if (R - a) / c > t or (R + a) / c < t:
        p = 0

    else:

        p = B* Ao * c / Cp * (R - c * t) / np.sqrt( a**2 - (R - c*t)**2 )

    p_values[i] = p

buffer = 200
max_index = min(np.argmax(p_values) + buffer, n_points - 1)
min_index = max(np.argmin(p_values) - buffer, 0)

plt.plot(t_points, p_values, color = 'k')
plt.plot( t_points[min_index: max_index], p_values[min_index:max_index], color = 'y' )
plt.show()