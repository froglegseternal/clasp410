#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def surface_temp(alpha=0.33, S_naught=1350, sigma = 5.67*(10**-8)):
    ''' 1-layer model equation.'''
    return (((1- alpha)*S_naught)/sigma) ** 1/4


year = np.array([1900, 1950, 2000])

s0 = np.array([1365, 1366.5, 1368])
t_anom = np.array([-.4,0,.4])
t_theoretical = np.array([0, 0, 0])


for i in range(len(year)):
    t_theoretical[i] = surface_temp(S_naught = s0[i])

sum = 0
for i in range(len(t_theoretical)):
    sum += t_theoretical[i]

average = sum/len(t_theoretical)

for i in range(len(t_anom)):
    t_anom[i] = t_anom[i] + average
fig, axes = plt.subplots(1, 1)
axes.plot(year, t_anom, label="Anomalous temperatures")
axes.plot(year, t_theoretical, label="Theoretical temperatures")
axes.set_xlabel("Time (years)")
axes.set_ylabel("Temperature (degrees Celsius)")
fig.show()