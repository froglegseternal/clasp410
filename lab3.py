#!usr/bin/env python3

# Standard Imports
import numpy as np
import  matplotlib.pyplot as plt


def initialize_array(nlayers, epsilon, ground_epsilon, solar_flux, debug=False):
    '''
    Create the A and b arrays, and then return them as a tuple.

    Parameters
    ==========
    nlayers: int
        The number of layers for the model to use
    epsilon: float
        Our emissivity value
    ground_epsilon: float
        The albedo value 
    solar_flux: float
        The value of solar irradiance [W/m^2]
    debug: boolean
        Whether this is a debugging run

    Returns
    =======
        A: numpy array
            Coefficient matrix
        b: numpy array
            Vector of constants 
    '''
    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers + 1):
        for j in range(nlayers + 1):
            if j == 0:
                if i == 0:
                    A[i,j] = -1
                elif i == 1:
                    A[i, j] = epsilon
                elif i == nlayers:
                    A[i,j] = ground_epsilon*pow(1 - epsilon, abs(j - i) - 1)
                else:
                    A[i,j] = ground_epsilon * pow(1 - epsilon, abs(j - i) - 1)
            elif i == j:
                A[i,j] = -2
            else:
                A[i,j] = epsilon * pow(1 - epsilon, abs(j - i) - 1)
            if debug:
                print(f'A[i={i},j={j}] = {A[i, j]}')
    b[0] = -1/4 * (1-ground_epsilon) * solar_flux
    if debug:
        print(A)
    return A, b

def initialize_array_adj(nlayers, epsilon, ground_epsilon, solar_flux, debug=False):
    '''
    Create the A and b arrays, and then return them as a tuple.

    Parameters
    ==========
    nlayers: int
        The number of layers for the model to use
    epsilon: float
        Our emissivity value
    ground_epsilon: float
        The albedo value 
    solar_flux: float
        The value of solar irradiance [W/m^2]
    debug: boolean
        Whether this is a debugging run

    Returns
    =======
        A: numpy array
            Coefficient matrix
        b: numpy array
            Vector of constants 
    '''
    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers + 1):
        for j in range(nlayers + 1):
            if j == 0:
                if i == 0:
                    A[i,j] = -1
                elif i == 1:
                    A[i, j] = epsilon
                elif i == nlayers:
                    A[i,j] = ground_epsilon*pow(1 - epsilon, abs(j - i) - 1)
                else:
                    A[i,j] = ground_epsilon * pow(1 - epsilon, abs(j - i) - 1)
            elif i == j:
                A[i,j] = -2
            else:
                A[i,j] = epsilon * pow(1 - epsilon, abs(j - i) - 1)
            if debug:
                print(f'A[i={i},j={j}] = {A[i, j]}')
    if debug:
        print(A)
    return A, b
def solve_equation(A, b):
    '''
    Given the A and b matrices, solve for the flux matrix and return it. It does this by inverting the A matrix and
    multiplying it by the b one.

    Parameters 
    ==========
    A: numpy array
        Our A matrix
    b: numpy array
        Our b matrix

    Returns
    =======
    fluxes: numpy array
        Our solution matrix
    '''

    # Invert matrix:
    Ainv = np.linalg.inv(A)

    # Get solution:
    fluxes = np.matmul(Ainv, b)

    # Return solution:
    return fluxes

def convert_to_temps(fluxes, emissivity):
    '''
    Given the flux array and emissivity, return the array of temperatures.

    Parameters
    ==========
    fluxes: numpy array
        The array of fluxes
    emissivity: float
        Our emissivity value

    Returns
    =======
    temps: numpy array
        The array of temperatures
    '''

    # First, define temps as being equal to fluxes in order to easily set the matrix to the right size.
    temps = fluxes

    # Define the Stefan-Boltzmann constant
    sigma = 5.6703 * (10. ** -8.)

    for x in range(len(fluxes)):
        if x == 0:
            temps[x] = (fluxes[x]/sigma) ** 0.25
        else:
            temps[x] = (fluxes[x]/(sigma * emissivity)) ** 0.25

    return temps

def solve_model(nlayers, epsilon, ground_epsilon, solar_flux, debug=False):
    '''
    Run the entire model, returning the temperature matrix.

    Parameters
    ==========
    nlayers: int
        The number of layers for the model to use
    epsilon: float
        Our emissivity value
    ground_epsilon: float
        The albedo value 
    solar_flux: float
        The value of solar irradiance
    debug: boolean
        Whether this is a debugging run

    Returns
    =======
    temps: numpy array
        The temperature matrix
    '''
    A, b = initialize_array(nlayers, epsilon, ground_epsilon, solar_flux, debug=debug)
    fluxes = solve_equation(A, b)
    temps = convert_to_temps(fluxes, epsilon)
    return temps

def solve_model_adj(nlayers, epsilon, ground_epsilon, solar_flux, debug=False):
    '''
    Run the entire model, returning the temperature matrix.

    Parameters
    ==========
    nlayers: int
        The number of layers for the model to use
    epsilon: float
        Our emissivity value
    ground_epsilon: float
        The albedo value 
    solar_flux: float
        The value of solar irradiance
    debug: boolean
        Whether this is a debugging run

    Returns
    =======
    temps: numpy array
        The temperature matrix
    '''
    A, b = initialize_array_adj(nlayers, epsilon, ground_epsilon, solar_flux, debug=debug)
    fluxes = solve_equation(A, b)
    temps = convert_to_temps(fluxes, epsilon)
    return temps
def problem_three():
    '''
    Solves problem three of the assignment.
    '''
    # Set earth's albedo to the current value of such.
    albedo = 0.3
    # For the first part of this question, there should be a single layer atmosphere.
    layers = 1
    # For both parts of this question, there should be a solar flux of 1350.
    s_flux = 1350.
    # Define a matrix of the emissivities to iterate over.
    emisses = np.linspace(0, 1, 100)
    # Initialize the matrix of surface temperatures.
    temps = np.zeros(100)
    for i in range(len(emisses)):
        temps[i] = solve_model(layers, emisses[i], albedo, s_flux)[0]

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    axes.plot(emisses, temps, ls='solid',color='C2')
    axes.set_title("Temperature vs. Emissivity")
    axes.set_xlabel('Emissivity (unitless)')
    axes.set_ylabel('Temperature (Kelvin)')
    fig.savefig("lab3_3_part1")

    # Part 2 of the problem is below here
    # Set the emissivity as specified.
    emissivity = 0.255
    # Define the range of values for the number of layers.
    layers = np.arange(1, 100, 1)
    # Initialize the matrix of surface temperatures.
    temps = np.zeros(len(layers))
    # Define an empty array of values that work
    answers = []
    for i in range(len(layers)):
        temps[i] = solve_model(layers[i], emissivity, albedo, s_flux)[0]
        try:
           if abs(temps[i] - 288) < 10:
                print(layers[i])
                answers.append(layers[i])
        except IndexError:
            print("No solutions found")

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    axes.plot(layers, temps, ls='solid',color='C2')
    axes.set_title('Number of Layers vs. Temperature')
    axes.set_xlabel('Number of Layers')
    axes.set_ylabel('Temperature (Kelvin)')
    fig.savefig("lab3_3_part2")
    try:
        # The first value in layers should be the minimum number of layers needed
        answer = answers[0]
    except:
        answer = 100 # If there are no solutions, default to one hundred layers.
    
    answer_temps = solve_model(answer, emissivity, albedo, s_flux)

    # Set an array of altitudes.
    altitudes = np.zeros(len(answer_temps))

    # Altitude of the uppermost part of Earth's atmosphere in miles
    atmo_alt = 6214
    
    # Set the altitude values to the correct value.
    for i in range(len(altitudes)):
        altitudes[i] = (i/100)*atmo_alt


    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    axes.plot(answer_temps, altitudes, ls='solid',color='C2')
    axes.set_title("Altitude vs. Temperature")
    axes.set_ylabel('Altitude (miles)')
    axes.set_xlabel('Temperature (Kelvin)')
    fig.savefig("lab3_3_part3")
def problem_four():
    '''
    Solves problem four of the assignment.
    '''
    # Define our expected surface temperature.
    surf_temp = 700
    # Define our solar flux in watts per square meter
    sol_flux = 2600
    # Define our emissivity
    emmis = 1.
    # Define Venus' albedo
    albedo = 0.7

    # Define the range of values for the number of layers.
    layers = np.arange(1, 100, 1)
    # Initialize the matrix of surface temperatures.
    temps = np.zeros(len(layers))
    # Define an empty array of values that work
    answers = []

    for i in range(len(layers)):
        temps[i] = solve_model(layers[i], emmis, albedo, sol_flux)[0]
        if abs(temps[i] - surf_temp) < 10:
            answers.append(layers[i])

    # The first value in layers should be the minimum number of layers needed
    answer = answers[0]
    print(str(answer) + " layers are needed.")

def problem_five():
    '''
    Solves problem five of the assignment.
    '''
    # Define our solar flux in watts per square meter
    sol_flux = 1350
    # Define our emissivity
    emmis = 0.5
    # Define Earth's albedo
    albedo = 0.3
    temps = solve_model_adj(100, emmis, albedo, sol_flux)
    # Set an array of altitudes.
    altitudes = np.zeros(len(temps))

    # Altitude of the uppermost part of Earth's atmosphere in miles
    atmo_alt = 6214
    
    # Set the altitude values to the correct value.
    for i in range(len(altitudes)):
        altitudes[i] = (i/100)*atmo_alt


    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    axes.plot(temps, altitudes, ls='solid',color='C2')
    axes.set_title("Altitude vs. Temperature")
    axes.set_ylabel('Altitude (miles)')
    axes.set_xlabel('Temperature (Kelvin)')
    fig.savefig("lab3_5")

    print("The surface temperature is " + str(temps[0]) + " Kelvin")
