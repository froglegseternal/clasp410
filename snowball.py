#!/usr/bin/env python3

'''
Tools for solving our Snowball Earth problem.
'''

import numpy as np
import matplotlib.pyplot as plt

S0 = 1370 # Solar flux (S0) in terms of W/m^2
radearth = 6378000. # Earth radius in meters.
lamb = 100. # Thermal diffusivity of atmosphere.
mxdlyr = 50. # depth of mixed layer (m)
sigma = 5.67e-8 # Steffan-Boltzman constant
C = 4.2e6 # Heat capacity of water
rho = 1020 # Density of sea-water (kg/m^3)
albedo_ice = 0.6 # Albedo of ground when ground-cover is ice/snow
albedo_gnd = 0.3 # Albedo of ground when ground-cover is water/ground.

def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.
    Parameters
    ----------
    lats_in : Numpy array
    Array of latitudes in degrees where temperature is required.
    0 corresponds to the south pole, 180 to the north.
    Returns
    -------
    temp : Numpy array
    Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
    23, 19, 14, 9, 1, -11, -19, -47])
    # Get base grid:
    npoints = T_warm.size
    dlat = 180 / npoints # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.
    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)
    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2
    return temp

def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.
    Parameters
    ----------
    S0 : float
    Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
    Latitudes to output insolation. Following the grid standards set in
    the diffusion program, polar angle is defined from the south pole.
    In other words, 0 is the south pole, 180 the north.
    Returns
    -------
    insolation : numpy array
    Insolation returned over the input latitudes.
    '''
    # Constants:
    max_tilt = 23.5 # tilt of earth in degrees
    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)
    # Daily rotation of earth reduces solar constant by distributing the sun
    # energy all along a zonal band
    dlong = 0.01 # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)
    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]
    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.
    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365
    return insolation

def basic_diffusion(dt=31536000., t_final=315360000000., debug=False):
    '''
    The basic diffusion solver.
    
    Parameters
    ==========
    dt: integer
        Time step in seconds. Default value is number of seconds in one year.
    t_final: integer
        Final time in seconds. Default value is number of seconds in 10,000 years.
    debug: boolean
        Optional debug with print statements
    '''
    npoints = 200 # Number of points in the y direction.
    dlat = 180 / npoints # Latitude spacing.

    dy = np.pi * dlat * radearth /180 # Latitude spacing in meters.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.

    A = np.zeros((npoints,npoints)) # Initialize the tridiagonal matrix
    for i in range(npoints):
        for j in range(npoints):
            if i == j:
                A[i,j] = -2./(dy**2)
            elif j == 1 and i == 0:
                A[i,j] = 2/(dy**2)
            elif j == npoints -2 and i == npoints -1:
                A[i,j] = 2/(dy**2)
            elif j == i + 1:
                A[i,j] = 1/(dy**2)
            elif j == i - 1:
                A[i,j] = 1/(dy**2)
    
    L = np.identity(npoints) - lamb*dt*A # Define the L matrix, the one to be inverted.
    if debug:
        print("L matrix: "+str(L))
    temps = temp_warm(lats) # Initialize the temperature vector

    for i in range(int(np.ceil(t_final/dt))):
        inv_L = np.linalg.inv(L)
        if debug:
            print("Inverted L matrix: " + str(inv_L))
        np.matmul(inv_L,temps,out=temps)
        if debug:
            print("Temperature array: " + str(temps))
    return temps

def spherical_correction(dt=31536000., t_final=315360000000., debug=False):
    '''
    The basic diffusion solver, with added spherical correction term.
    
    Parameters
    ==========
    dt: integer
        Time step in seconds. Default value is number of seconds in one year.
    t_final: integer
        Final time in seconds. Default value is number of seconds in 10,000 years.
    debug: boolean
        Optional debug with print statements
    '''
    dt_sec = dt # dt in seconds, same as dt for now but in case of future changes this value is put here.
    npoints = 200 # Number of points in the y direction.
    dlat = 180 / npoints # Latitude spacing.

    dy = np.pi * dlat * radearth/180 # Latitude spacing in meters.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.
    edge = np.linspace(0, 180, npoints+1) # Lat cell edges

    A = np.zeros((npoints,npoints)) # Initialize the tridiagonal matrix
    for i in range(npoints):
        for j in range(npoints):
            if i == j:
                A[i,j] = -2./(dy**2)
            elif j == 1 and i == 0:
                A[i,j] = 2/(dy**2)
            elif j == npoints -2 and i == npoints -1:
                A[i,j] = 2/(dy**2)
            elif j == i + 1:
                A[i,j] = 1/(dy**2)
            elif j == i - 1:
                A[i,j] = 1/(dy**2)
    
    L = np.identity(npoints) - lamb*dt*A # Define the L matrix, the one to be inverted.
    if debug:
        print("L matrix: "+str(L))
    temps = temp_warm(lats) # Initialize the temperature vector

    Area = np.zeros((npoints)) # The area array
    # Create matrix "B" to assist with adding spherical-correction factor
    # Corner values for 1st order accurate Neumann boundary conditions
    B = np.zeros((npoints,npoints))
    B[np.arange(npoints-1),np.arange(npoints-1)+1] = 1
    B[np.arange(npoints-1)+1,np.arange(npoints-1)] = -1
    B[0, :] = B[-1, :] = 0
    # Set the surface area of the "side" of each latitude ring at bin center.
    Area = np.pi*((radearth+50.)**2-radearth**2)*np.sin(lats*np.pi/180.)
    # Now find dAxz/dlat. This  never changes. It contains an extra dlat for the Temp derivative, too.
    dAxz = np.matmul(B, Area) / (Area * 4 * dy **2)
    

    for i in range(int(np.ceil(t_final/dt))):
        inv_L = np.linalg.inv(L)
        if debug:
            print("Inverted L matrix: " + str(inv_L))
            print("B times temps: " + str(np.matmul(B,temps)) + "B * area: " + str(np.matmul(B,Area)))
        sphcorr = lamb * dt_sec * np.matmul(B,temps)*dAxz #Spherical correction term
        if debug:
            print("Spherical correction factor: "+str(sphcorr))
        np.matmul(inv_L,temps+sphcorr,out=temps)
        if debug:
            print("Temperature array: " + str(temps))
    return temps

def rad_forcing(dt=31536000., t_final=315360000000., debug=False,albedo=0.3,emissivity=1.0,lamb=100.):
    '''
    The basic diffusion solver, with added spherical correction and radiative forcing terms.
    
    Parameters
    ==========
    dt: integer
        Time step in seconds. Default value is number of seconds in one year.
    t_final: integer
        Final time in seconds. Default value is number of seconds in 10,000 years.
    debug: boolean
        Optional debug with print statements
    albedo: either a function or a float
        The albedo - or, a function to generate the albedo.
    emissivity: either a function or a float
        The emissivity - or, a function to generate the emissivity.
    lamb: float
        The diffusivity value
    '''
    dt_sec = dt # dt in seconds, same as dt for now but in case of future changes this value is put here.
    npoints = 100 # Number of points in the y direction.
    dlat = 180 / npoints # Latitude spacing.

    dy = np.pi * dlat * radearth/180 # Latitude spacing in meters.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.

    A = np.zeros((npoints,npoints)) # Initialize the tridiagonal matrix
    for i in range(npoints):
        for j in range(npoints):
            if i == j:
                A[i,j] = -2./(dy**2)
            elif j == 1 and i == 0:
                A[i,j] = 2/(dy**2)
            elif j == npoints -2 and i == npoints -1:
                A[i,j] = 2/(dy**2)
            elif j == i + 1:
                A[i,j] = 1/(dy**2)
            elif j == i - 1:
                A[i,j] = 1/(dy**2)
    
    L = np.identity(npoints) - lamb*dt*A # Define the L matrix, the one to be inverted.
    if debug:
        print("L matrix: "+str(L))
    temps = temp_warm(lats) # Initialize the temperature vector

    Area = np.zeros((npoints,npoints)) # The area array
    # Create matrix "B" to assist with adding spherical-correction factor
    # Corner values for 1st order accurate Neumann boundary conditions
    B = np.zeros((npoints,npoints))
    B[np.arange(npoints-1),np.arange(npoints-1)+1] = 1
    B[np.arange(npoints-1)+1,np.arange(npoints-1)] = -1
    B[0, :] = B[-1, :] = 0
    # Set the surface area of the "side" of each latitude ring at bin center.
    Area = np.pi*((radearth+50.)**2-radearth**2)*np.sin(lats*np.pi/180.)
    # Now find dAxz/dlat. This  never changes. It contains an extra dlat for the Temp derivative, too.
    dAxz = np.matmul(B, Area) / (Area * 4 * dy **2)

    # Set insolation:
    insol = insolation(S0, lats)

    # Solve!
    for i in range(int(np.ceil(t_final/dt))):
        inv_L = np.linalg.inv(L)
        if debug:
            print("Inverted L matrix: " + str(inv_L))
        sphcorr = lamb * dt_sec * np.matmul(B,temps)*dAxz #Spherical correction term
        
        # Add insolation term:
        radiative = (1-albedo)*insol - emissivity*sigma*(temps+273)**4
        forc_term = dt_sec * radiative / (rho*C*mxdlyr)# Radiative forcing term
        np.matmul(inv_L,temps+sphcorr+forc_term,out=temps)
        if debug:
            print("Temperature array: " + str(temps))
    return temps    

def varying_albedo(temps):
    '''
    Helper function that is responsible for varying albedo based on specific conditions.

    Parameters
    ==========
    temps: numpy array
        the temperature array of current temperatures
    '''
    albedo = np.zeros((temps.size))

    # Update albedo based on conditions:
    loc_ice = temps <= -10
    albedo[loc_ice] = albedo_ice
    albedo[~loc_ice] = albedo_gnd


def vary_diffusivity():
    '''
    Function that varies the diffusivity of the model and then plots the results.
    '''
    ranger = [0, 25, 50, 75, 100, 125, 150]
    results = []
    
    for x in ranger:
        results.append(rad_forcing(lamb=x))

    npoints = 100
    dlat = 180 / npoints # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    
    for i in range(len(ranger)):
        axes.plot(lats, )

    
def vary_emissivity():
    '''
    Function that varies the emissivity of the model and then plots the results.
    '''
    range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    for x in range:
        results.append(rad_forcing(emissivity=x))