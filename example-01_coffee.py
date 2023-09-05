#!/usr/bin/env python3

'''
Coffee problem in class.
'''

import numpy as np
import matplotlib.pyplot as plt

# Create a time array:
tfinal, tstep = 600, 1
time = np.arange(0, tfinal, tstep)

def solve_temp(time, k = 1, T_env = 25, T_init=90):
    '''
    This function takes an array of times and returns an array of temperatures corresponding to each time.
    
    Parameters
    ==========

    time : Numpy array of times
        Array of time inputs for which you want corresponding temps
    
    Other Parameters
    ================


    Returns
    =======

    '''
    temp = T_env + (T_init - T_env) * np.exp(-k * time)

    return temp