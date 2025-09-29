import numpy as np
import matplotlib.pyplot as plt

# Params (assume external detector: R >= a)
c = 1.5   # mm/us
R = 5.0   # mm (detector-to-center distance)
a = 1.0   # mm (sphere radius)
assert R >= a, "For a single external N-wave, require R >= a."

# Time axis
n = 1000
t = np.linspace(0.0, 10.0, n)

# Correct arrival window for single N-wave (external detector)
t1 = (R - a) / c       # first arrival (near side)
t2 = (R + a) / c       # last arrival (far side)
Delta = t2 - t1

# Normalized N-wave: +1 at t1, -1 at t2, linear in between, 0 outside
Y = np.zeros_like(t)
mask = (t >= t1) & (t <= t2)
Y[mask] = (t1 + t2 - 2.0 * t[mask]) / Delta

# Plot
plt.plot(t, Y)
plt.axvline(t1, ls="--", label=f"t1 = {t1:.3f} μs")
plt.axvline(t2, ls="--", label=f"t2 = {t2:.3f} μs")
plt.xlabel("time t (μs)")
plt.ylabel("normalized pressure p_N(t)")
plt.title("N-wave at detector: +1 → −1 over arrival window")
plt.legend()
plt.tight_layout()
plt.show()
