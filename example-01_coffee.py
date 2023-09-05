#!/usr/bin/env python3

'''
Coffee problem in class.
'''

import numpy as np
import matplotlib.pyplot as plt

# Create a time array:
tfinal, tstep = 600, 1
time = np.arange(0, tfinal, tstep)

def solve_temp(time, k = 1./300, T_env = 25, T_init=90):
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
    temp = T_env + ((T_init - T_env) * np.exp(-k * time))

    return temp

def time_to_temp(T_targ, k=1/300., T_env=20, T_init=90):
    '''
    Given an initial temperature, `T_init`, an ambient temperature, `T_env`,
    and a cooling rate, return the time required to reach a target temperature, `T_targ`.

    '''
    return (-1/k) * np.log((T_targ-T_env)/(T_init-T_env))

#Solve our coffee question!
T_cream = solve_temp(time, T_init=85)
T_nocrm = solve_temp(time, T_init=90)

#Get time to drinkable temp
t_cream = time_to_temp(60, T_init=85) #Add cream right away.
t_nocrm = time_to_temp(60, T_init=90) #Add cream once at 60 deg.
t_smart = time_to_temp(65, T_init=90) #Put cream in at 65 deg.

#Create figure and axes objects:
fig, ax = plt.subplots(1,1)

#Plot lines and label
ax.plot(time, T_nocrm, label='No cream till cool.')
ax.plot(time, T_cream, label="Cream right away")

ax.axvline(t_nocrm, c='red',ls='--', label='No Cream: T=60')
ax.axvline(t_cream, c='blue',ls='--', label='Cream: T=60')
ax.axvline(t_smart, c='gray',ls='--', label='No Cream: T=65')

ax.legend(loc='best')

fig.tight_layout()