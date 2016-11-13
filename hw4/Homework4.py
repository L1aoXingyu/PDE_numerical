
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def u(x):
    """define u(x,t) at t=0"""
    return 1.+math.sin(x)+math.sin(10.*x)

def IC(N, a, b, f):
    """define initial condition"""
    x = np.zeros([N+1, 1])
    dx = 1.0*(b-a)/N
    for i in xrange(N+1):
        x[i, 0] = f(a + 1.0*(b-a)/N * i)
    return x, dx

def FTCS(init, t, delta):
    """define FTCS Scheme"""
    x1, dx = init
    dt = dx**2 * delta
    N = x1.shape[0]
    x2 = np.zeros([N, 1])
    tt = 0
    for tt in np.arange(dt, t, dt):
        x2[0, 0] = x1[0, 0] + delta * (x1[1, 0] - 2. * x1[0, 0] + x1[N-2, 0])
        x2[N-1, 0] = x1[N-1, 0] + delta * (x1[1, 0] - 2. * x1[N-1, 0] + x1[N-2, 0])
        for i in xrange(1, N-1):
            x2[i, 0] = x1[i, 0] + delta * (x1[i+1, 0] - 2.*x1[i, 0] + x1[i-1, 0])
        x1 = x2.copy()

    ddt = t - tt
    ddelta = ddt/dx/dx
    x2[0, 0] = x1[0, 0] + ddelta * (x1[1, 0] - 2. * x1[0, 0] + x1[N-2, 0])
    x2[N-1, 0] = x1[N-1, 0] + ddelta * (x1[1, 0] - 2. * x1[N-1, 0] + x1[N-2, 0])
    for i in xrange(1, N-1):
        x2[i, 0] = x1[i, 0] + ddelta * (x1[i+1, 0] - 2.*x1[i, 0] + x1[i-1, 0])
    x1 = x2.copy()
    return x1

def BTCS(init, t, delta):
    """defin BTCS Scheme"""
    x1, dx = init
    dt = dx**2 * delta
    N = x1.shape[0]
    x2 = np.zeros([N, 1])
    ### construct matrix of BTCS
    A = np.zeros([N-1, N-1])
    for i in xrange(1, N-2):
        A[i, i-1] = -delta
        A[i, i] = 1. + 2. * delta
        A[i, i+1] = -delta
    A[0, 0] = 1. + 2. * delta
    A[0, 1] = -delta
    A[0, N-2] = -delta
    A[N-2, 0] = -delta
    A[N-2, N-2] = 1. + 2. * delta
    A[N-2, N-3] = -delta
    ###
    inverse_A = np.linalg.inv(A)
    tt = 0
    for tt in np.arange(dt, t, dt):
        x2[0:N-1, 0] = np.dot(inverse_A, x1[0:N-1, 0])
        x2[N-1, 0] = x2[0, 0]
        x1 = x2.copy()
    ### for last part t
    ddt = t - tt
    ddelta = ddt/dx/dx
    ### construct matrix of BTCS
    A = np.zeros([N-1, N-1])
    for i in xrange(1, N-2):
        A[i, i-1] = -ddelta
        A[i, i] = 1. + 2. * ddelta
        A[i, i+1] = -ddelta
    A[0, 0] = 1. + 2. * ddelta
    A[0, 1] = -ddelta
    A[0, N-2] = -ddelta
    A[N-2, 0] = -ddelta
    A[N-2, N-2] = 1. + 2. * ddelta
    A[N-2, N-3] = -ddelta
    ###
    inverse_A = np.linalg.inv(A)
    x2[0:N-1, 0] = np.dot(inverse_A, x1[0:N-1, 0])
    x2[N-1, 0] = x2[0, 0]
    x1 = x2.copy()
    ###
    return x1

def run():
    """execute the two schemes"""
    x_init = IC(100, 0, 2*math.pi, u) # initial condition

    ### FTCS
    FTCS_output = []
    for d in [0.5, 0.8, 1.0]:
        for i in [0.01, 1.0]:
            FTCS_output.append(FTCS(x_init, i, d))
    ###
    ### BTCS
    BTCS_output = []
    for j in [0.01, 1.0]:
        BTCS_output.append(BTCS(x_init, j, 100./(2.*math.pi)))
    ###
    ### plot the image of initial
    x = np.zeros([x_init[0].shape[0], 1])
    for i in xrange(x_init[0].shape[0]):
        x[i, 0] = 2.*math.pi/100 * i
    plt.figure(figsize=(12, 10))
    plt.plot(x[:, 0], x_init[0], 'r', label='t=0')
    plt.title('t=0 initial value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    ###
    ### plot FTCS and BTCS
    ### t = 0.01, delta = 0.5
    plt.figure(figsize=(12, 10))
    plt.plot(x[:, 0], FTCS_output[0], label='(FTCS)t=0.01, delta=0.5')
    plt.plot(x[:, 0], BTCS_output[0], label='(BTCS)t=0.01')
    plt.title('FTCS and BTCS (t=0.01, delta=0.5)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    ###
    ### t = 1, delta = 0.5
    plt.figure(figsize=(12, 10))
    plt.plot(x[:, 0], FTCS_output[1], label='(FTCS)t=1.0, delta=0.5')
    plt.plot(x[:, 0], BTCS_output[1], label='(BTCS)t=1.0')
    plt.title('FTCS and BTCS (t=1.0, delta=0.5)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    ###
    ### t = 0.01, delta = 0.8
    plt.figure(figsize=(12, 10))
    plt.plot(x[:, 0], FTCS_output[2], label='(FTCS)t=0.01, delta=0.8')
    plt.plot(x[:, 0], BTCS_output[0], label='(BTCS)t=0.01')
    plt.title('FTCS and BTCS (t=0.01, delta=0.8)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    ###
    ### t = 1.0, delta = 0.8
    plt.figure(figsize=(12, 10))
    plt.plot(x[:, 0], FTCS_output[3], label='(FTCS)t=1.0, delta=0.8')
    plt.plot(x[:, 0], BTCS_output[1], label='(BTCS)t=1.0')
    plt.title('FTCS and BTCS (t=1.0, delta=0.8)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    ###
    ### t = 0.01, delta = 1.0
    plt.figure(figsize=(12, 10))
    plt.plot(x[:, 0], FTCS_output[4], label='(FTCS)t=0.01, delta=1.0')
    plt.plot(x[:, 0], BTCS_output[0], label='(BTCS)t=0.01')
    plt.title('FTCS and BTCS (t=0.01, delta=1.0)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    ###
    ### t = 1.0, delta = 1.0
    plt.figure(figsize=(12, 10))
    plt.plot(x[:, 0], FTCS_output[5], label='(FTCS)t=1.0, delta=1.0')
    plt.plot(x[:, 0], BTCS_output[1], label='(BTCS)t=1.0')
    plt.title('FTCS and BTCS (t=1.0, delta=1.0)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    ###


if __name__ == '__main__':
    run()
