# Programming Homework 5

*Sherlock Liao    SA16001031*

## 1 Question

For question
$$
\begin{cases}
u_{t} = u_{xx}, \quad x \in (0,1),t>0 \\
u(x,0) = cos(\frac{\pi x}{2}),\quad x \in (0,1] \\
u(1,t) = 0 \\
u_{x}(0,t) = 0
\end{cases}
$$
Let $n_{x}$ = 10, $\Delta t = 0.004$, make grid function as unknown function to construct numerical scheme, get the approximate result at t=0.06,0.1,0.9, and analysis the result. 

## 2 Algorithm

Let $I_{j}=[x_{j-\frac{1}{2}},x_{j+\frac{1}{2}}]$ and we can get  $\Omega_j^n = I_{j} \times [t_n, t_{n+1}]$ .For $j=1,2, \dots, n_{x}-1$, we can integrate on $\Omega_j^n$ and using numerical integrate and we can get the formula below:
$$
v_j^{n+1} = v_j^n + \Delta t \delta_0 v_j^n \quad j=1,2,\dots , n_x-1 \\

\delta_0 = D_+D_-
$$
And for the boundary, we have two boundaries, x=0 and x=1. For x=1, we know from the partial differential equation, $v_{n_x}^n = 0$ for all n. For x=0, let $I_0 = [x_0, x_{\frac{1}{2}}]$ , so we can integrate on $\Omega_0^n$ and get the formula:
$$
v_0^{n+1}=v_0^n + \frac{2\Delta t (v_1^n-v_0^n)}{\Delta x^2}
$$
So for all the situations, we can compute the function values at the grid and iterate until the required T.

## 3 Result and Analysis

By coding, we write the program to compute the approximate result and plot the picture at t=0.06,0.1,0.9.  ![screenshot](/Users/sherlockliao/Library/Group Containers/Q79WDW8YH9.com.evernote.Evernote/Evernote/quick-note/514160797___Evernote-China/quick-note-KA4lpJ/attachment--VLHRjs/screenshot.png)



From this picture, we can see that at t=0.06,0.1,0.9, the result is converged, and with time going by, the u(x,t) is smaller and smaller. The energy is less and less.



## 4 Code

```python

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
```



