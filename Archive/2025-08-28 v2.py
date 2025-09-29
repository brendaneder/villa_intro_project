import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

R_s = 7 # sphere radius in mm
d_o = 50 # detector distance from sphere center, mm
d_e = d_o - R_s # shortest distance from detector to sphere edge

t_min = 27
t_max = 35 # us
t_n = 2000 # us

t_points = np.linspace(t_min, t_max, t_n)

a = 100 # scalar for final value (IDK proper value)

# assume r >> R_s, such that 1/2 of sphere face is exposed to detector

c = 1.5 #mm/us

pulse_period = 1 # us
pulse_max = 1
pulse_min = -1 * pulse_max

# create pressure equation

def pressure_wave(t, pulse_period, pulse_max, pulse_min):
    if t < 0:
        return 0
    elif t > pulse_period:
        return 0
    else:
        pulse_slope = (pulse_max - pulse_min) / pulse_period
        return (pulse_max - (t * pulse_slope))

def distance(d_e, R_s, x):
    y = np.sqrt(R_s ** 2 - x ** 2) # height from sphere center to point on edge of sphere measured
    d = np.sqrt(x** 2 + (d_e + (R_s - y))**2) # distance from detector to sphere edge measured is lateral distance (x) and vertical distance (edge + radius - sphere height)
    return d




def detector_pressure(x, t, pulse_period, pulse_max, pulse_min, a, R_s, c, d_e, style = "ring"):
    d = distance(d_e, R_s, x)
    t_impact = d / c
    t_effective = t - t_impact
    f = pressure_wave(t_effective, pulse_period, pulse_max, pulse_min)
    if style == "point":
        return a * f / d**2
    elif style == "ring":
        return a * f * 2 * np.pi * x / d**2
    

Y = []
for t_val in t_points:
    res, err = sp.integrate.quad(detector_pressure, 0, R_s,
                          args=(t_val, pulse_period, pulse_max, pulse_min, a, d_o, c, d_e, "ring"))
    Y.append(res)


plt.plot(t_points,Y)
plt.show()








