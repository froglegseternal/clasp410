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
def fwddiff_neumann(dt=0.02, dx=0.2,c2=1.0, xmax=1.0, tmax=0.2, init=sample_init):
    '''
    Neumann Forward difference solver.

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

    # Set initial condition.
    if callable(init):
        temp[:, 0] = init(x)
    else:
        temp[:, 0] = init


    # Solve!
    for j in range(0, N-1):
        temp[1:-1, j+1] = (1-2*r)*temp[1:-1, j] + r*(temp[2:, j] + temp[:-2, j])
        temp[0, j+1] = temp[1,j+1]
        temp[-1, j+1] = temp[-2, j+1]

    return x, t, temp
def fwddiff(dt=0.02, dx=0.2,c2=1.0, xmax=1.0, tmax=0.2, init=sample_init,max_bound=0.0,min_bound=0.0,xmin=0.0,tmin=0.0):
    '''
    Forward difference solver. Taking in a subset of the time step, x step, thermal diffusivity, maximum x value, maximum time, and initializing and boundary
    functions/values, minimum value of x, and minimum value of t,
    this function will either generate and return the x array, time array, and temperature array (if the solution is numerically stable)
    or it will return a tuple of -1 values (if the solution is not stable.) 

    Parameters
    ----------
    dt, dx : float, default=0.02, 0.2 
        Time and space step.
    c2 : float, default=1.0
        Thermal diffusivity.

    xmax, tmax : float, default=1.0, 0.2
        Set max values for space and time grids
    init, max_bound, min_bound : scalar or function
        Set initial condition and boundary conditions. If functions, should take position
        as an input and return temperature using same units as x, temp.
    xmin: float, default = 0.0
        Initial minimum value for the space grid
    tmin: float, default = 0.0
        Initial minimum value for the time grid
    Returns
    -------
    x : numpy vector
        Array of position locations/x-grid
    t : numpy vector
        Array of time points/y-grid
    temp : numpy 2D array
        Temperature as a function of time and space.
    
    '''
    # Check for instability
    if dt > (dx ** 2)/(2*c2):
        return -1, -1, -1
    # Set constants:
    r = c2 * dt/dx**2

    # Create space and time grids
    x = np.arange(xmin, xmax+dx, dx)
    t = np.arange(tmin, tmax+dt, dt)
    
    # Save number of points.
    M, N = x.size, t.size

    # Create temperature solution array:
    temp = np.zeros([M, N])

    # Set initial and boundary conditions.
    if callable(min_bound):
        temp[0, :] = min_bound(t)
    else:
        temp[0, :] = min_bound
    if callable(max_bound):
        temp[-1,:] = max_bound(t)
    else:
        temp[-1, :] = max_bound
    # Set initial condition.
    if callable(init):
        temp[:, 0] = init(x)
    else:
        temp[:, 0] = init


    # Solve!
    for j in range(0, N-1):
        temp[1:-1, j+1] = (1-2*r)*temp[1:-1, j] + r*(temp[2:, j] + temp[:-2, j])

    return x, t, temp

def gen_heatmap(dt=0.02, dx=0.2,c2=1.0, xmax=1.0, tmax=0.2, init=sample_init,max_bound=0.0,min_bound=0.0,xmin=0.0,tmin=0.0):
    '''
    Generates the heatmap for a given dt, dx, c^2, xmax, tmax, xmin, tmin, and initial and boundary conditions. Does this by running the forward difference solver with these values and then using
    matplotlib's pcolor and colorbar functions. This function instead simply returns -1 if the given conditions are numerically unstable.
    '''
    #Get solution using the forward-difference solver:
    x, time, heat = fwddiff(dt=dt, dx=dx, c2=c2, xmax=xmax, tmax=tmax, init=init,max_bound=max_bound,min_bound=min_bound,xmin=xmin,tmin=tmin)

    # Check for instability.
    if type(x)=="int" and type(time)=="int" and type(heat)=="int":
        return -1

    # Create a figure
    fig, axes = plt.subplots(1, 1) 

    # Create a color map and add a color bar.
    map = axes.pcolor(time/365, x, heat, cmap='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    axes.set_xlabel('Time (years)')
    axes.set_ylabel('Height (m)')
    return fig

def gengroundprof(dt=0.02, dx=0.2,c2=1.0, xmax=1.0, tmax=0.2, init=sample_init,max_bound=0.0,min_bound=0.0,xmin=0.0,tmin=0.0):
    '''
    Generates the ground profile for a given dt, dx, c2, xmax, tmax, xmin, tmin, and initial and boundary conditions. Does this by running the forward difference solver with these values and then
    finding the minimum values in the final year of the solution.
    '''
    # Get solution using the forward-difference solver:
    x, time, heat = fwddiff(dt=dt, dx=dx, c2=c2, xmax=xmax, tmax=tmax, init=init,max_bound=max_bound,min_bound=min_bound,xmin=xmin,tmin=tmin)

    # Check for instability
    if type(x)=="int" and type(time)=="int" and type(heat)=="int":
        return -1
    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result.
    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)

    # Extract the max values over the final year:
    summer = heat[:, loc:].max(axis=1)
    # Create a temp profile plot:
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.plot(winter, x, 'b-', label='Winter')
    ax2.plot(summer, x, 'r', label='Summer')
    ax2.legend()
    ax2.set_xlabel('Temperature ($C$)')
    ax2.set_ylabel('Height (m)')

    return fig

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

def check_unstable():
    '''
    Validates that the forward difference solver quits if the solution is unstable.
    
    Returns
    =======
    bad_sol: boolean
        True if the solution does not quit for an unstable solution.
    '''
    bad_sol = False
    for i in range(100):
        dx = np.random.rand()
        c2 = np.random.rand()
        dt = ((dx ** 2) / (2*c2))+1
        i, j, k = fwddiff(dt = dt, dx=dx, c2=c2)
        if i!=-1 or j!=-1 or k!=-1:
            bad_sol = True
    return bad_sol
def homework_problem():
    # Get solution using the forward-difference solver:
    x, time, heat = fwddiff(dt=0.0002, dx=0.02)
    x1, time1, heat1 = fwddiff_neumann(dt=0.0002,dx=0.02)

    # Create a figure
    fig, axes = plt.subplots(2, 1) 

    # Create a color map and add a color bar.
    map = axes[0].pcolor(time, x, heat, cmap='seismic', vmin=-1, vmax=1,label='Dirichlet')
    map2 = axes[1].pcolor(time1, x1, heat1, cmap='seismic', vmin=-1, vmax=1,label='Neumann')
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')
    axes[0].legend()
    axes[1].legend()
    plt.savefig('homework_problem')

def prob_two():
    '''
    Solves problem 2 of the assignment.
    '''
    # Set the conditions
    t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
    xmax = 0.0
    xmin = -100.0
    max_bound = temp_kanger
    min_bound = 5.0
    tmax = 5 * 365
    c2 = 0.25
    dx = 1
    dt = (dx**2)/(2*c2)
    steady = False

    # Find the number of years needed to reach steady state
    
    while not steady:
        x, time, heat = fwddiff(dt=dt, dx=dx, c2=c2, init=0, xmax=xmax, tmax=tmax, max_bound=max_bound,min_bound=min_bound,xmin=xmin)
        
        # Set indexing for the final year of results:
        loc = int(-365/dt) # Final 365 days of the result.

        # Extract the min values over the final year:
        winter = heat[:, loc:].min(axis=1)

        # Extract the max values over the final year:
        summer = heat[:, loc:].max(axis=1)

        # Extract the min values over the penultimate year
        winter2 = heat[:,2*loc:loc].min(axis=1)

        # Extract the max values over the penultimate year
        summer2 = heat[:,2*loc:loc].max(axis=1)

        # Temporarily say we're in steady state, even if not true.
        steady = True

        for i in range(len(summer)):
            if abs(summer[i] - winter[i]) < 10:
                if abs(summer[i] - summer2[i]) > 0.001 :
                    steady = False

        # Add on another year
        tmax += 365

    print("Years to stabilization: " + str(tmax/365))
    fig = gen_heatmap(dx=dx,dt=dt, c2=c2, xmax=xmax, tmax=tmax, init=0,max_bound=max_bound,min_bound=min_bound,xmin=xmin) 
    fig.show()
    fig.savefig('heatmap_prob_2')
    fig2 = gengroundprof(dx=dx,dt=dt, c2=c2, xmax=xmax, tmax=tmax, init=0,max_bound=max_bound,min_bound=min_bound,xmin=xmin)
    fig2.show()
    fig2.savefig('groundprof_prob_2')

#Now, solve problem three of the assignment.

# First, add a uniform 0.5 temperature shift.

# Set the conditions
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
t_kanger = t_kanger + 0.5
xmax = 0.0
xmin = -100.0
max_bound = temp_kanger
min_bound = 5.0
tmax = 5 * 365
c2 = 0.25
dx = 1
dt = (dx**2)/(2*c2)
steady = False

# Find the number of years needed to reach steady state
    
while not steady:
    x, time, heat = fwddiff(dt=dt, dx=dx, c2=c2, init=0, xmax=xmax, tmax=tmax, max_bound=max_bound,min_bound=min_bound,xmin=xmin)
        
    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result.

    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)

    # Extract the max values over the final year:
    summer = heat[:, loc:].max(axis=1)

    # Extract the min values over the penultimate year
    winter2 = heat[:,2*loc:loc].min(axis=1)

    # Extract the max values over the penultimate year
    summer2 = heat[:,2*loc:loc].max(axis=1)

    # Temporarily say we're in steady state, even if not true.
    steady = True

    for i in range(len(summer)):
        if abs(summer[i] - winter[i]) < 10:
            if abs(summer[i] - summer2[i]) > 0.001 :
                steady = False
    # Add on another year
    tmax += 365

print("Years to stabilization for a 0.5 shift: " + str(tmax/365))
fig = gen_heatmap(dx=dx,dt=dt, c2=c2, xmax=xmax, tmax=tmax, init=0,max_bound=max_bound,min_bound=min_bound,xmin=xmin) 
fig.show()
fig.savefig('heatmap_prob_3_0_5')
fig2 = gengroundprof(dx=dx,dt=dt, c2=c2, xmax=xmax, tmax=tmax, init=0,max_bound=max_bound,min_bound=min_bound,xmin=xmin)
fig2.show()
fig2.savefig('groundprof_prob_3_0_5')

# Now, add a uniform 1 degree temperature shift.

# Set the conditions
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
t_kanger = t_kanger + 1
xmax = 0.0
xmin = -100.0
max_bound = temp_kanger
min_bound = 5.0
tmax = 5 * 365
c2 = 0.25
dx = 1
dt = (dx**2)/(2*c2)
steady = False

# Find the number of years needed to reach steady state
    
while not steady:
    x, time, heat = fwddiff(dt=dt, dx=dx, c2=c2, init=0, xmax=xmax, tmax=tmax, max_bound=max_bound,min_bound=min_bound,xmin=xmin)
        
    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result.

    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)

    # Extract the max values over the final year:
    summer = heat[:, loc:].max(axis=1)

    # Extract the min values over the penultimate year
    winter2 = heat[:,2*loc:loc].min(axis=1)

    # Extract the max values over the penultimate year
    summer2 = heat[:,2*loc:loc].max(axis=1)

    # Temporarily say we're in steady state, even if not true.
    steady = True

    for i in range(len(summer)):
        if abs(summer[i] - winter[i]) < 10:
            if abs(summer[i] - summer2[i]) > 0.001 :
                steady = False
    # Add on another year
    tmax += 365

print("Years to stabilization for a 1 shift: " + str(tmax/365))
fig = gen_heatmap(dx=dx,dt=dt, c2=c2, xmax=xmax, tmax=tmax, init=0,max_bound=max_bound,min_bound=min_bound,xmin=xmin) 
fig.show()
fig.savefig('heatmap_prob_3_1')
fig2 = gengroundprof(dx=dx,dt=dt, c2=c2, xmax=xmax, tmax=tmax, init=0,max_bound=max_bound,min_bound=min_bound,xmin=xmin)
fig2.show()
fig2.savefig('groundprof_prob_3_1')

# Finally, add a uniform 3 degree temperature shift.

# Set the conditions
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
t_kanger = t_kanger + 3
xmax = 0.0
xmin = -100.0
max_bound = temp_kanger
min_bound = 5.0
tmax = 5 * 365
c2 = 0.25
dx = 1
dt = (dx**2)/(2*c2)
steady = False

# Find the number of years needed to reach steady state
    
while not steady:
    x, time, heat = fwddiff(dt=dt, dx=dx, c2=c2, init=0, xmax=xmax, tmax=tmax, max_bound=max_bound,min_bound=min_bound,xmin=xmin)
        
    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result.

    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)

    # Extract the max values over the final year:
    summer = heat[:, loc:].max(axis=1)

    # Extract the min values over the penultimate year
    winter2 = heat[:,2*loc:loc].min(axis=1)

    # Extract the max values over the penultimate year
    summer2 = heat[:,2*loc:loc].max(axis=1)

    # Temporarily say we're in steady state, even if not true.
    steady = True

    for i in range(len(summer)):
        if abs(summer[i] - winter[i]) < 10:
            if abs(summer[i] - summer2[i]) > 0.001 :
                steady = False
    # Add on another year
    tmax += 365

print("Years to stabilization for a 3 shift: " + str(tmax/365))
fig = gen_heatmap(dx=dx,dt=dt, c2=c2, xmax=xmax, tmax=tmax, init=0,max_bound=max_bound,min_bound=min_bound,xmin=xmin) 
fig.show()
fig.savefig('heatmap_prob_3_3')
fig2 = gengroundprof(dx=dx,dt=dt, c2=c2, xmax=xmax, tmax=tmax, init=0,max_bound=max_bound,min_bound=min_bound,xmin=xmin)
fig2.show()
fig2.savefig('groundprof_prob_3_3')