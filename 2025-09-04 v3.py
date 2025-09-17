import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

R_o = 10 # sphere radius in mm
r = np.array([50,0,0])
r_o = np.array([0,0,0])
d = np.linalg.norm(r - r_o) # detector distance from sphere center, mm

d_min = d - R_o # shortest distance from detector to object
d_max = d + R_o # longest distance from detector to object

c = 1.5 # mm/us speed of sound
hz_s = 1000000 # pulse frequency 1/s
hz = 1 / 1000000 # pulse frequency 1/us

t_wave = 1/hz # us

x_wave = c * t_wave # mm of wavelength

t_min = d_min / c # us until first wave hits
t_max = np.linalg.norm( d - R_o ) + t_wave # us until end of last wave hits


t_start_observing = 0
t_stop_observing = 80 # us
t_n = 200 # us

t_points = np.linspace(t_start_observing, t_stop_observing, t_n) # number of points to sample across

print(1)

def p_xt(x, x1, x2, R_o, d):

    
    circumfrence = ( 1 / ( 2 * d ) ) * np.sqrt( (4 * d**2 * x**2 ) - ( d**2 - R_o**2 + x**2 ) ** 2 )
    pressure = 2 * ( x - x1 ) / (x2 - x1) - 1
    integrand = circumfrence * pressure

    return integrand

p = np.empty( len(t_points) , dtype = float)
for i,t_val in enumerate(t_points):

    x1 = t_val * c
    x2 = x1 + x_wave

    if t_val < t_min or t_val > t_max:
        p[i] = 0
    else:
        x_lower_bound = max( x1, d_min )
        x_upper_bound = min(x2, d_max)

        res, err = sp.integrate.quad(p_xt, x_lower_bound, x_upper_bound,
                                 args=( x1, x2, R_o, d))
    
        p[i] = res







plt.plot(t_points,p)
plt.show()








