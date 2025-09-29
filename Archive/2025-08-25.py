import numpy as np
import matplotlib.pyplot as plt




s = 1 #reflection scaling factor from surface pressure
Lx = int(2e3) #number of cells x
Ly = int(2e3) #number of cells y
l = 100 # um side length of cell
co = 1e7 # speed of sound, um/s
hz = 50000 # vibrational frequency, 1/s
t_osc = 1/hz # period, s
t_step = 10*l/co # step time
t_max = l*Ly*5/co #max time
t_max = 5*t_step
print("max steps =", t_max//t_step)

t = 0 # set t=0
A = np.zeros([Ly, Lx]) #create matrix
dA = np.zeros([Ly, Lx])
dB = np.zeros([Ly, Lx])
print("aa")
for x in range(Lx):
    for y in range(Ly):
        dist_a = np.sqrt((l*x)**2 + (l*y)**2)
        dist_b = np.sqrt( (l*(2*Ly-y))**2 + (l*x)**2)
        dA[y,x] = dist_a
        dB[y,x]=dist_b
print("a")

while t < t_max:
    t += t_step
    print(t // t_step)
    p_t = np.zeros([Ly,Lx]) #pressure map at time t
    for y in range(Ly):
        for x in range(Lx):
            ta = dA[y,x] / co
            tb = dB[y,x] / co
            if t < ta:
                p_t[y,x] = 0

            else:
                pressure = 1 - 2 * np.mod(t - ta, t_osc)/t_osc
            
            if t >= tb:
                pressure += s * (1 - 2 * np.mod(t - tb, t_osc)/t_osc)
            
            p_t[y,x] = pressure


    plt.imshow(p_t, cmap='hot', interpolation='nearest')
    plt.show()




            



