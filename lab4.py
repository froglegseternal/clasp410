#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
# Correct temperature values for the heat equation to validate against:
correct = np.array([[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
    [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
    [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
    [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
    [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
    [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
    [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
    [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
    [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
    [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
    [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]])
correct = correct.transpose()

def sample_init(x):
    '''Simple initial boundary condition function'''
    return 4*x - 4*x**2

def fwddiff(dt=0.02, dx=0.2,c2=1.0, xmax=1.0, tmax=0.2, init=sample_init):
    '''
    Forward difference solver.

    Parameters
    ----------
    dt, dx : float, default=0.02, 0.2 
        Time and space step.
    c2 : float, default=1.0
        Thermal diffusivity.

    xmax, tmax : float, default=1.0, 0.2
        Set max values for space and time grids
    init : scalar or function
        Set initial condition. If a function, should take position
        as an input and return temperature using same units as x, temp.
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
    # Set initial condition.
    if callable(init):
        temp[:, 0] = init(x)
    else:
        temp[:, 0] = init


    # Solve!
    for j in range(0, N-1):
        temp[1:-1, j+1] = (1-2*r)*temp[1:-1, j] + r*(temp[2:, j] + temp[:-2, j])

    return x, t, temp

def gen_heatmap(dt=0.02, dx=0.2,c2=1.0, xmax=1.0, tmax=0.2, init=sample_init):
    '''
    Generates the heatmap for a given dt, dx, c^2, xmax, tmax, and init. Does this by running the forward difference solver with these values and then using
    matplotlib's pcolor and colorbar functions.
    '''
    #Get solution using the forward-difference solver:
    x, time, heat = fwddiff(dt, dx, c2, xmax, tmax, init)

    # Create a figure
    fig, axes = plt.subplots(1, 1) 

    # Create a color map and add a color bar.
    map = axes.pcolor(time, x, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')

def gengroundprof(dt=0.02, dx=0.2,c2=1.0, xmax=1.0, tmax=0.2, init=sample_init):
    '''
    Generates the ground profile for a given dt, dx, c2, xmax, tmax, and init. Does this by running the forward difference solver with these values and then
    finding the minimum values in the final year of the solution.
    '''
    # Get solution using the forward-difference solver:
    x, time, heat = fwddiff(dt, dx, c2, xmax, tmax, init)

    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result.

    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)

    # Create a temp profile plot:
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.plot(winter, x, label='Winter')

def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for Kangerlussuaq, Greenland.
    Parameters
    ==========
    t: array
        The times to generate temperatures for
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp * np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()

def check_correct(error):
    '''
    Validates that the forward difference solver works for the check condition for a given error.

    Parameters
    ==========
    error: float
        The error to check if the solver function is correct to.
    Returns
    =======
    within: boolean
        Whether the function falls within the given error.
    '''
    within = (abs(fwddiff()[2] - correct) < 0.1).all()
    return within
