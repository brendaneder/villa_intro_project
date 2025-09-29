import numpy as np
import matplotlib.pyplot as plt

B = 1
Cp = 1
c = 1.5 # mm/us
rc = 0 # mm
ro = 65 # mm
Rs = 5 # mm

t_vals = np.linspace(0,60,1000)
p = np.zeros(1000)

for i, t in enumerate(t_vals):
    if np.abs(c*t - ro) <= Rs:
        pressure = B * c**2 / (2 * Cp * ro) * (ro - c*t)
        p[i] = pressure

plt.plot(t_vals, p)
plt.show()
print('done')