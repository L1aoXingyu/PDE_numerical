
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    """define u(x, y, t) at t=0"""
    return sin(pi*x)*sin(2*pi*y)

def IC(J_x, J_y, x_a, x_b, y_a, y_b):
    """define initial value"""
    dx = 1.0*(x_b - x_a)/J_x
    dy = 1.0*(y_b - y_a)/J_y
    ic = np.zeros([J_x+1, J_y+1])
    for i in range(0, J_x+1):
        for j in range(0, J_y+1):
            ic[i, j] = f(x_a+i*dx, y_b+j*dy)
    return ic, dx, dy

def FTCS_2(ic, dt, T):
    """define 2 dimentional FTCS scheme"""
    x1 = ic[0]
    dx = ic[1]
    dy = ic[2]
    x2 = np.zeros([x1.shape[0], x1.shape[1]])
    t = 0
    for t in np.arange(dt, T, dt):
        x2[:, 0] = 0
        x2[0, :] = 0
        x2[:, x1.shape[1]-1] = 0
        x2[x1.shape[0]-1, :] = 0
        for i in range(1, x1.shape[0]-1):
            for j in range(1, x1.shape[1]-1):
                x2[i, j] = x1[i, j] + dt/(dx**2) * (x1[i+1, j] - 2 * x1[i, j] + x1[i-1, j]) + dt/(dy**2) * (x1[i, j+1] - 2 * x1[i, j] + x1[i, j-1])
        x1 = x2.copy()
    ddt = T - t
    x2[:, 0] = 0
    x2[0, :] = 0
    x2[:, x1.shape[1]-1] = 0
    x2[x1.shape[0]-1, :] = 0
    for i in range(1, x1.shape[0]-1):
        for j in range(1, x1.shape[1]-1):
            x2[i, j] = x1[i, j] + ddt/(dx**2) * (x1[i+1, j] - 2 * x1[i, j] + x1[i-1, j]) + ddt/(dy**2) * (x1[i, j+1] - 2 * x1[i, j] + x1[i, j-1])
    x1 = x2.copy()
    return x1

def run():
    ic = IC(20, 20, 0, 1, 0, 1)
    FTCS_2_output = []
    x = np.arange(0, 1.0002, ic[1])
    y = np.arange(0, 1.0002, ic[2])
    x, y =np.meshgrid(x, y)
    for dt in [0.0005, 0.001]:
        for t in [0.06, 0.1, 0.9]:
            FTCS_2_output.append(FTCS_2(ic, dt, t))

    for i in range(6):
        fig = plt.figure(figsize=(12, 10))
        ax = Axes3D(fig)
        ax.plot_surface(x, y, FTCS_2_output[i], rstride=1, cstride=1, cmap='rainbow')
        plt.show()

if __name__ == '__main__':
    run()
