#!usr/bin/env python3

# Standard Imports
import numpy as np
import  matplotlib as plt


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
        The value of solar irradiance
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
            if i == 0:
                if j == 0:
                    A[i,j] = -1
                elif j == nlayers:
                    A[i,j] = pow(1 - epsilon, abs(j - i) - 1)
                else:
                    A[i,j] = ground_epsilon * pow(1 - epsilon, abs(j - i) - 1)
            elif i == j:
                A[i,j] = -2
            else:
                A[i,j] = epsilon * pow(1 - epsilon, abs(j - i) - 1)
            if debug:
                print(f'A[i={i},j={j}] = {A[i, j]}')
    b[0] = -1 * solar_flux
    return A, b

def solve_equation(A, b):
    '''
    Given the A and b matrices, solve for the x matrix and return it. It does this by inverting the A matrix and
    multiplying it by the b one.

    Parameters 
    ==========
    A: numpy array
        Our A matrix
    b: numpy array
        Our b matrix

    Returns
    =======
    x: numpy array
        Our solution matrix
    '''

    # Invert matrix:
    Ainv = np.linalg.inv(A)

    # Get solution:
    fluxes = np.matmul(Ainv, b)