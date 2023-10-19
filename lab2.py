#!/usr/bin/env python3

# Standard imports as well as scipy:
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def Ndt_predprey(t, N, a=1.0, b=2.0, c=1.0, d=3.0):
    '''
    This function calculates the Lotka-Volterra predator-prey equations for
    two sepcies. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.

    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this function.

    Parameters
    ----------
    t: float
        The current time (not used here).
    N: two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d: float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt: floats
        The time derivatives of `N1` and `N2`. 
    '''
    # Here, N is a two element list such that N1=N[0] and N2=N[1]
    dN1dt = a * N[0] - b*N[0]*N[1]
    dN2dt = -1 * c * N[1] + d * N[0]*N[1]

    return dN1dt, dN2dt
def Ndt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.

    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this function.

    Parameters
    ----------
    t: float
        The current time (not used here).
    N: two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d: float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt: floats
        The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]
    
    return dN1dt, dN2dt

def euler_solve(func, N1_init=.5, N2_init=.5, dT=.1, t_final=100.0, a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Euler's Method. Given a python function, the initial values of N1 and N2,
    the time step, and the final time, return the time elapsed as an array and
    the normalized population density solutions.

    Parameters
    ----------
    func: function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init: float
        Initial conditions for `N1` and `N2`, ranging from (0, 1]
    dT: float, default=0.1
        Timestep in years.
    t_final: float, default=100
        Integrate until this value is reached, in years.
    a,b,c,d: floats, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    time: Numpy array
        Time elapsed in years.
    N1, N2: Numpy arrays
        Normalized population density solutions
    '''

    # Create time array.
    time = np.arange(0, t_final, dT)
    
    # Create containers for the solution, set initial condition.
    N1 = np.zeros(time.size)
    N2 = np.zeros(time.size)
    N1[0] = N1_init
    N2[0] = N2_init

    # Integrate forward:
    for i in range(1, time.size):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]])
        N1[i] = N1[i-1] + dT*dN1
        N2[i] = N2[i-1] + dT*dN2

    return time, N1, N2

def solve_rk8(func, N1_init=.5, N2_init=.5, dT=10, t_final=100.0, a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using Scipy's ODE
    class and the adaptive step 8th order solver.

    Parameters
    ----------
    func: function
        A python function that takes `time`, [`N1`, `N2`] as inputs and returns the time
        derivative of N1 and N2.
    N1_init, N2_init: float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT: float, default=10
        Largest timestep allowed in years.
    t_final: float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d: float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    
    Returns
    -------
    time: Numpy array
        Time elapsed in years
    N1, N2: Numpy arrays
        Normalized population density solutions.
    '''

    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init], args=[a, b, c, d],
                       method='DOP853', max_step=dT)
    
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]

    # Return values to caller.
    return time, N1, N2

def problem_one():
    '''
    This fuction solves the first problem in the assignment. It does so by first generating a plot with initially defined values, and then
    it iterates through a range of step sizes twice, once for the competition model and once for the predator/prey model.
    '''

    # Define our constants for this problem.
    a = 1
    b = 2
    c = 1
    d = 3
    max_T = 100
    init_N1 = 0.3
    init_N2 = 0.6

    # Define our changing values.
    comp_step = 1 # Euler time step for the competition model
    prey_step = 0.05 # Euler time step for the prey model

    # Run our models for the competition model.
    time_euler, N1_euler, N2_euler = euler_solve(Ndt_comp, N1_init=init_N1, N2_init=init_N2, dT=comp_step, t_final=max_T)
    time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T)

    # Plot our first graph.
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].plot(time_euler, N1_euler, ls='solid',color='C2',label='N1 Euler')
    axes[0].plot(time_euler, N2_euler, ls='solid',color='C3', label='N2 Euler')
    axes[0].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
    axes[0].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
    axes[0].set_title("Lotka-Volterra Competition Model")
    axes[0].legend()
    axes[0].set_xlabel('Time (years)')
    axes[0].set_ylabel('Population/Carrying Cap.')
    
    # Run our models for the predator-prey model.
    time_euler, N1_euler, N2_euler = euler_solve(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, dT=prey_step, t_final=max_T)
    time_rk, N1_rk, N2_rk = solve_rk8(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, t_final=max_T)

    # Plot our second graph.
    axes[1].plot(time_euler, N1_euler, ls='solid',color='C2',label='N1 (Prey) Euler')
    axes[1].plot(time_euler, N2_euler, ls='solid',color='C3', label='N2 (Predator) Euler')
    axes[1].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 (Prey) RKB')
    axes[1].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 (Predator) RKB')
    axes[1].set_title("Lotka-Volterra Predator-Prey Model")
    axes[1].legend()
    axes[1].set_xlabel('Time (years)')
    axes[1].set_ylabel('Population/Carrying Cap.')

    fig.suptitle("Coefficients: a=1, b=2, c=1, d=3")
 #   fig.show()
    fig.savefig(fname="question_one_init_lab2.png")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    k = 0
    i = 0
    for step in np.linspace(0.1, 2, 10):
         # Run our models for the competition model.
        time_euler, N1_euler, N2_euler = euler_solve(Ndt_comp, N1_init=init_N1, N2_init=init_N2, dT=step, t_final=max_T)
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T, dT=step)

        # Plot our first graphs.
        
        axes[k,i].plot(time_euler, N1_euler, ls='solid',color='C2',label='N1 Euler')
        axes[k,i].plot(time_euler, N2_euler, ls='solid',color='C3', label='N2 Euler')
        axes[k,i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes[k,i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes[k,i].set_title("Step: " + str(step))
        axes[k,i].legend()
        axes[k,i].set_xlabel('Time (years)')
        axes[k,i].set_ylabel('Population/Carrying Cap.')
        i += 1
        if i >= 5:
            i = 0
            k += 1
    fig.suptitle("Coefficients: a=1, b=2, c=1, d=3; Lotka-Volterra Competition Model")
#    fig.show()
    fig.savefig(fname="question_one_comp_lab2.png")
    
    k = 0
    i = 0
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for step in np.linspace(0.01, 0.1, 10):
        # Run our models for the predator-prey model.
        time_euler, N1_euler, N2_euler = euler_solve(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, dT=step, t_final=max_T)
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, dT=step,t_final=max_T)    

         # Plot our second graphs.
        axes[k,i].plot(time_euler, N1_euler, ls='solid',color='C2',label='N1 (Prey) Euler')
        axes[k,i].plot(time_euler, N2_euler, ls='solid',color='C3', label='N2 (Predator) Euler')
        axes[k,i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 (Prey) RKB')
        axes[k,i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 (Predator) RKB')
        axes[k,i].set_title("Step: "+str(step))
        axes[k,i].legend()
        axes[k,i].set_xlabel('Time (years)')
        axes[k,i].set_ylabel('Population/Carrying Cap.')
        axes[k,i].set_ylim(0,1)
        i+=1
        if i >= 5:
            i = 0
            k += 1
    fig.suptitle("Coefficients: a=1, b=2, c=1, d=3; Lotka-Volterra Predator-Prey Model")
#        fig.show()
    fig.savefig(fname="question_one_predprey_lab2.png")
def problem_two():
    '''
    This function solves the second problem in the assignment. It does so by iterating a range of values for the following variables
    in  the Lotka-Volterra competition equation; a, b, c, d, initial N_1, and initial N_2. After doing this, it plots hard-coded values for
    a specific scenario where the solution leads to an equilbrium. Optionally, this function will also loop through possible combinations of
    the previously mentioned values in an attempt to find a solution where there is an equilibrium.
    '''
     # Define our constants for this problem.
    max_T = 100
    comp_step = 1

    # Define our changing values.
    a = 1
    b = 2
    c = 1
    d = 3
    init_N1 = 0.3
    init_N2 = 0.6
    # Define our graph indices
    i = 0
    # Define our subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 10))
    for init_N1 in np.linspace(0.1, 0.5, 5): # Vary initial N1
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T, dT = comp_step)
  
        axes[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes[i].set_title("initial N1: "+str(init_N1))
        axes[i].legend()
        axes[i].set_xlabel('Time (years)')
        axes[i].set_ylabel('Population/Carrying Cap.')
        i+=1 # Increment our graph index

    fig.suptitle("Coefficients: a=1, b=2, c=1, d=3; Lotka-Volterra Competition Model")
#    fig.show()
    fig.savefig(fname="question_two_varied_N1_comp_lab2.png")
    init_N1 = 0.3 # Reset initial N1 back to original value.
    fig, axes = plt.subplots(1, 5, figsize=(20, 10)) # Define our subplots
    i = 0 # Reset our graph index
    for init_N2 in np.linspace(0.1,0.5,5): # Vary N2
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T, dT = comp_step)
        axes[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes[i].set_title("Initial N2: "+str(init_N2))
        axes[i].legend()
        axes[i].set_xlabel('Time (years)')
        axes[i].set_ylabel('Population/Carrying Cap.')
        i += 1 # Increment our graph index
    fig.suptitle("Coefficients: a=1, b=2, c=1, d=3; Lotka-Volterra Competition Model")
#    fig.show()
    fig.savefig(fname="question_two_varied_N2_comp_lab2.png")

    init_N2 = 0.6 # Reset initial N2 back to original value
    i = 0 # Reset our graph index
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    for a in np.linspace(0.1,2,4): # Vary a
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T, dT = comp_step, a=a)
        axes[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes[i].set_title("a: "+str(a))
        axes[i].legend()
        axes[i].set_xlabel('Time (years)')
        axes[i].set_ylabel('Population/Carrying Cap.')
        i += 1 # Increment our graph index
    
    fig.suptitle("Coefficients: b=2, c=1, d=3; Lotka-Volterra Competition Model")
#    fig.show()
    fig.savefig(fname="question_two_varied_a_comp_lab2.png")
    a = 1 # Reset a back to original value
    i = 0 # Reset our graph index
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    for b in np.linspace(0.1,2,4): # Vary b
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T, dT = comp_step, b=b)
        axes[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes[i].set_title("b: "+str(b))
        axes[i].legend()
        axes[i].set_xlabel('Time (years)')
        axes[i].set_ylabel('Population/Carrying Cap.')
        i += 1 # Increment our graph index
    fig.suptitle("Coefficients: a=1,c=1,d=3; Lotka-Volterra Competition Model")
#    fig.show()
    fig.savefig(fname="question_two_varied_b_comp_lab2.png")
    b = 2 # Reset b back to original value
    i = 0 # Reset our graphp index
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    for c in range(1,5): # Vary c
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T, dT = comp_step, c=c)
        axes[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes[i].set_title("c: "+ str(c))
        axes[i].legend()
        axes[i].set_xlabel('Time (years)')
        axes[i].set_ylabel('Population/Carrying Cap.')
        i += 1 # Increment our graph index
    fig.suptitle("Coefficients: a=1,b=2,d=3; Lotka-Volterra Competition Model")
#    fig.show()
    fig.savefig(fname="question_two_varied_c_comp_lab2.png")
    c = 1 # Reset c back to original value
    i = 0 # Reset our graph index
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    for d in range(1,5): # Vary d
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T, dT = comp_step, d=d)
        axes[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes[i].set_title("d: "+ str(d))
        axes[i].legend()
        axes[i].set_xlabel('Time (years)')
        axes[i].set_ylabel('Population/Carrying Cap.')
        i += 1 # Increment our graph index
    
    fig.suptitle("Coefficients: a=1,b=2,c=1; Lotka-Volterra Competition Model")
#    fig.show()
    fig.savefig(fname="question_two_varied_d_comp_lab2.png")
    d = 5
    b = 0.1
    c = 5
    a = 0.1
    init_N1 = 0.5
    init_N2 = 0.5
    max_T = 100000 # Increase max T so we can ensure that the destabilization point is not simply further along.
    time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T, dT = comp_step, a=a, b=b,c=c,d=d)
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    axes.plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
    axes.plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
    axes.set_title("Lotka-Volterra Competition Model")
    axes.legend()
    axes.set_xlabel('Time (years)')
    axes.set_ylabel('Population/Carrying Cap.')
    
    fig.suptitle("Coefficients: a=0.1,b=0.1,c=5,d=5; initial N1 = 0.5, initial N2 = 0.5.")
#    fig.show()
    fig.savefig(fname="question_two_final_comp_lab2.png")
'''    for a in np.linspace(0.1, 5, 9):
        for b in np.linspace(0.1, 5, 9):
            for c in np.linspace(0.1, 5, 9):
                for d in np.linspace(0.1, 5, 9):
                    time_rk, N1_rk, N2_rk = solve_rk8(Ndt_comp, N1_init=init_N1, N2_init=init_N2, t_final=max_T, dT = comp_step, a=a, b=b,c=c,d=d)
                    if(N1_rk[-1] > 0.2 and N2_rk[-1] > 0.2):
                        print("a: "+ str(a) + ",b:"+str(b)+",c:"+str(c)+",d:"+str(d)+",N1_init:"+str(init_N1)+",N2_init:"+str(init_N2))
'''


def problem_three():
    '''
    This function solves the third problem in the assignment. It does so by iterating through each of the variables a, b, c, d, initial N_1, and initial N_2
    in th Lotka-Volterra Predator=Prey equations. It plots both a phase diagram and a plot over time for each looped value.
    '''
     # Define our constants for this problem.
    max_T = 100

    # Define our changing values.
    a = 1
    b = 2
    c = 1
    d = 3
    init_N1 = 0.3
    init_N2 = 0.6

    # Run our model
    time_rk, N1_rk, N2_rk = solve_rk8(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, t_final=max_T, a=a, b=b,c=c,d=d)
    fig0, axes0 = plt.subplots(1, 5, figsize=(20, 10)) # Time plot
    fig1, axes1 = plt.subplots(1, 5, figsize=(20, 10)) # Phase plot

    i = 0 # Initialize plot index

    for init_N1 in np.linspace(0.1, 0.5, 5): # Vary initial N1
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, t_final=max_T, a=a, b=b, c=c, d=d)
        
        # Time plot
        axes0[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes0[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes0[i].set_title("Initial N1: " + str(init_N1))
        axes0[i].legend()
        axes0[i].set_xlabel('Time (years)')
        axes0[i].set_ylabel('Population/Carrying Cap.')
        axes0[i].set_ylim(0,1)

        # Phase plot
        axes1[i].plot(N1_rk, N2_rk)
        axes1[i].set_title("Initial N1: " + str(init_N1))
        axes1[i].set_xlabel("N1 Population/Carrying Cap.")
        axes1[i].set_ylabel("N2 Population/Carrying Cap.")    
        axes1[i].set_ylim(0,1)
        axes1[i].set_xlim(0,1)
        i += 1 # Increment plot index
    fig0.suptitle("Lotka-Volterra Predator-Prey Model Plot over Time. Coefficients: a=1, b=2, c=1, d=3")
    fig1.suptitle("Lotka-Volterra Predator-Prey Model Phase Plot. Coefficients: a=1, b=2, c=1, d=3")
 #   fig0.show()
 #   fig1.show()
    fig0.savefig(fname="question_three_varied_N1_time_predprey_lab2.png")
    fig1.savefig(fname="question_three_varied_N1_phase_predprey_lab2.png")

    init_N1 = 0.3 # Reset initial N1 back to original value.
    i = 0 # Reset plot index back to initial value.
    fig0, axes0 = plt.subplots(1, 5, figsize=(20, 10)) # Time plot
    fig1, axes1 = plt.subplots(1, 5, figsize=(20, 10)) # Phase plot
    for init_N2 in np.linspace(0.1,0.5,5): # Vary N2
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, t_final=max_T)
        # Time plot
        axes0[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes0[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes0[i].set_title("Initial N2: " + str(init_N2))
        axes0[i].legend()
        axes0[i].set_xlabel('Time (years)')
        axes0[i].set_ylabel('Population/Carrying Cap.')
        axes0[i].set_ylim(0,1)
        # Phase plot
        axes1[i].plot(N1_rk, N2_rk)
        axes1[i].set_title("Initial N2: " + str(init_N2))
        axes1[i].set_xlabel("N1 Population/Carrying Cap.")
        axes1[i].set_ylabel("N2 Population/Carrying Cap.")
        axes1[i].set_ylim(0,1)
        axes1[i].set_xlim(0,1)
        i += 1 # Increment plot index    

    fig0.suptitle("Lotka-Volterra Predator-Prey Model Plot over Time. Coefficients: a=1, b=2, c=1, d=3")
    fig1.suptitle("Lotka-Volterra Predator-Prey Model Phase Plot. Coefficients: a=1, b=2, c=1, d=3")
#    fig0.show()
#    fig1.show()
    fig0.savefig(fname="question_three_varied_N2_time_predprey_lab2.png")
    fig1.savefig(fname="question_three_varied_N2_phase_predprey_lab2.png")
   
    init_N2 = 0.6 # Reset initial N2 back to original value
    i = 0 # Reset plot index
    fig0, axes0 = plt.subplots(1, 4, figsize=(20, 10)) # Time plot
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 10)) # Phase plot
    for a in range(1,5): # Vary a
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, t_final=max_T, a=a,b=b,c=c,d=d)
        # Time plot
        axes0[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes0[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes0[i].set_title("a: " + str(a))
        axes0[i].legend()
        axes0[i].set_xlabel('Time (years)')
        axes0[i].set_ylabel('Population/Carrying Cap.')
        axes0[i].set_ylim(0,1)
        # Phase plot
        axes1[i].plot(N1_rk, N2_rk)
        axes1[i].set_title("a: "+str(a))
        axes1[i].set_xlabel("N1 Population/Carrying Cap.")
        axes1[i].set_ylabel("N2 Population/Carrying Cap.")
        axes1[i].set_ylim(0,1)
        axes1[i].set_xlim(0,1)
        
        i += 1 # Increment plot index
    
    fig0.suptitle("Lotka-Volterra Predator-Prey Model Plot over Time. Coefficients: b=2, c=1, d=3")
    fig1.suptitle("Lotka-Volterra Predator-Prey Model Phase Plot. Coefficients: b=2, c=1, d=3")
    
#    fig0.show()
#    fig1.show()
   
    fig0.savefig(fname="question_three_varied_a_time_predprey_lab2.png")
    fig1.savefig(fname="question_three_varied_a_phase_predprey_lab2.png")

    a = 1 # Reset a back to original value
    i = 0 # Reset plot index
    fig0, axes0 = plt.subplots(1, 4, figsize=(20, 10)) # Time plot
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 10)) # Phase plot
    for b in range(1,5): # Vary b
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, t_final=max_T, a=a,b=b,c=c,d=d)
        # Time plot
        axes0[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes0[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes0[i].set_title("b: " + str(b))
        axes0[i].legend()
        axes0[i].set_xlabel('Time (years)')
        axes0[i].set_ylabel('Population/Carrying Cap.')
        axes0[i].set_ylim(0,1)
        # Phase plot
        axes1[i].plot(N1_rk, N2_rk)
        axes1[i].set_title("b: "+str(b))
        axes1[i].set_xlabel("N1 Population/Carrying Cap.")
        axes1[i].set_ylabel("N2 Population/Carrying Cap.")
        axes1[i].set_ylim(0,1)
        axes1[i].set_xlim(0,1)
        
        i += 1 # Increment plot index
    
    fig0.suptitle("Lotka-Volterra Predator-Prey Model Plot over Time. Coefficients: a=1, c=1, d=3")
    fig1.suptitle("Lotka-Volterra Predator-Prey Model Phase Plot. Coefficients: a=1, c=1, d=3")
    
#   fig0.show()
#   fig1.show()
   
    fig0.savefig(fname="question_three_varied_b_time_predprey_lab2.png")
    fig1.savefig(fname="question_three_varied_b_phase_predprey_lab2.png")
    b = 2 # Reset b back to original value

    i = 0 # Reset plot index
    fig0, axes0 = plt.subplots(1, 4, figsize=(20, 10)) # Time plot
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 10)) # Phase plot
    for c in range(1,5): # Vary c
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, t_final=max_T, a=a,b=b,c=c,d=d)
        # Time plot
        axes0[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes0[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes0[i].set_title("c: " + str(c))
        axes0[i].legend()
        axes0[i].set_xlabel('Time (years)')
        axes0[i].set_ylabel('Population/Carrying Cap.')
        axes0[i].set_ylim(0,1)
        # Phase plot
        axes1[i].plot(N1_rk, N2_rk)
        axes1[i].set_title("c: "+str(c))
        axes1[i].set_xlabel("N1 Population/Carrying Cap.")
        axes1[i].set_ylabel("N2 Population/Carrying Cap.")
        axes1[i].set_ylim(0,1)
        axes1[i].set_xlim(0,1)
        
        i += 1 # Increment plot index
    
    fig0.suptitle("Lotka-Volterra Predator-Prey Model Plot over Time. Coefficients: a=1, b=2, d=3")
    fig1.suptitle("Lotka-Volterra Predator-Prey Model Phase Plot. Coefficients: a=1, b=2, d=3")
    
#   fig0.show()
#   fig1.show()
   
    fig0.savefig(fname="question_three_varied_c_time_predprey_lab2.png")
    fig1.savefig(fname="question_three_varied_c_phase_predprey_lab2.png")

    c = 1 # Reset c back to original value
    
    i = 0 # Reset plot index
    fig0, axes0 = plt.subplots(1, 4, figsize=(20, 10)) # Time plot
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 10)) # Phase plot
    for d in range(1,5): # Vary d
        time_rk, N1_rk, N2_rk = solve_rk8(Ndt_predprey, N1_init=init_N1, N2_init=init_N2, t_final=max_T, a=a,b=b,c=c,d=d)
        # Time plot
        axes0[i].plot(time_rk, N1_rk, ls='dashed',color='C2',label='N1 RKB')
        axes0[i].plot(time_rk, N2_rk, ls='dashed',color='C3',label='N2 RKB')
        axes0[i].set_title("d: " + str(d))
        axes0[i].legend()
        axes0[i].set_xlabel('Time (years)')
        axes0[i].set_ylabel('Population/Carrying Cap.')
        axes0[i].set_ylim(0,1)
        # Phase plot
        axes1[i].plot(N1_rk, N2_rk)
        axes1[i].set_title("d: "+str(d))
        axes1[i].set_xlabel("N1 Population/Carrying Cap.")
        axes1[i].set_ylabel("N2 Population/Carrying Cap.")
        axes1[i].set_ylim(0,1)
        axes1[i].set_xlim(0,1)
        
        i += 1 # Increment plot index
    
    fig0.suptitle("Lotka-Volterra Predator-Prey Model Plot over Time. Coefficients: a=1, b=2, c=1")
    fig1.suptitle("Lotka-Volterra Predator-Prey Model Phase Plot. Coefficients: a=1, b=2, c=1")
    
#   fig0.show()
#   fig1.show()
   
    fig0.savefig(fname="question_three_varied_d_time_predprey_lab2.png")
    fig1.savefig(fname="question_three_varied_d_phase_predprey_lab2.png")


problem_one() # Run problem one
problem_two() # Run problem two
problem_three() # Run problem three