
import math
import numpy as np
import matplotlib.pyplot as plt

def u(x):
    """define u(x,t) at t=0"""
    return math.cos(math.pi*x/2.)

def IC(N, a, b, f):
    """define initial condition"""
    x = np.zeros([N+1, 1])
    dx = 1.0*(b-a)/N
    for i in xrange(N+1):
        x[i, 0] = f(a + 1.0*(b-a)/N * i)
    return x, dx

def GridFD(init, t, dt):
    """define basing grid FD"""
    x_1, dx = init
    N = x_1.shape[0]
    x_2 = np.zeros([N, 1])
    tt = 0
    for tt in np.arange(dt, t, dt):
        x_2[0, 0] = x_1[0, 0] + 2. * dt / (dx**2) * (x_1[1, 0] - x_1[0, 0]) # boundary value
        x_2[N-1, 0] = 0 #boundary value
        for i in range(1, N-1):
            x_2[i, 0] = x_1[i, 0] + dt / (dx**2) * (x_1[i-1, 0] - 2.*x_1[i, 0] + x_1[i+1, 0])
        x_1 = x_2.copy()

    ddt = t - tt
    x_2[0, 0] = x_1[0, 0] + 2. * ddt / (dx**2) * (x_1[1, 0] - x_1[0, 0])
    x_2[N-1, 0] = 0
    for i in range(1, N-1):
        x_2[i, 0] = x_1[i, 0] + ddt / (dx**2) * (x_1[i-1, 0] - 2.*x_1[i, 0] + x_1[i+1, 0])
    x_1 = x_2.copy()

    return x_1

def run():
    """excute the two scheme"""
    x_init = IC(10, 0, 1, u)

    GridFD_output = []
    for t in [0.06, 0.1, 0.9]:
        GridFD_output.append(GridFD(x_init, t, 0.004))

    x = np.arange(0, 1.0001, x_init[1])

    ### plot the image
    plt.figure(figsize=(12, 10))

    plt.plot(x, GridFD_output[0], 'r', label='t=0.06') # t=0.06
    plt.plot(x, GridFD_output[1], 'b', label='t=0.1') # t=0.1
    plt.plot(x, GridFD_output[2], 'g', label='t=0.9') # t=0.9

    plt.title('Base Grid FD scheme at t=0.06, 0.1, 0.9')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    run()
