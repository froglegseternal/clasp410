#!/usr/bin/env python3
'''
A set of tools for solving and visualizing the heat equation in 1D.
'''

import numpy as np
import matplotlib.pyplot as plt

def heat_solve(dt=0.02, dx=0.2,c2=1.0, xmax=1.0, tmax=0.2):
    '''
    Stuff.

    Parameters
    ----------
    dt, dx : float, default=0.02, 0.2 
        Time and space step.
    c2 : float, default=1.0
        Thermal diffusivity.

    xmax, tmax : float, default=1.0, 0.2
        Set max values for space and time grids
    Returns
    -------
    x : numpy vector
        Array of position locations/x-grid
    t : numpy vector
        Array of time points/y-grid
    temp : numpy 2D array
        Temperature as a function of time and space.
    
    '''

    # Set constants:
    r = c2 * dt/dx**2

    # Create space and time grids
    x = np.arange(0, xmax+dx, dx)
    t = np.arange(0, tmax+dt, dt)
    
    # Save number of points.
    M, N = x.size, t.size

    # Create temperature solution array:
    temp = np.zeros([M, N])

    # Set initial and boundary conditions.
    temp[0, :] = 0
    temp[-1, :] = 0
    temp[:, 0] = 4*x - 4*x**2

    # Solve!
    for j in range(0, N-1):
        for i in range(1, M-1):
            temp[i, j+1] = (1-2*r)*temp[i, j] + r*(temp[i+1, j] + temp[i-1, j])

    return x, t, temp 