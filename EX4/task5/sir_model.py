import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import solve_ivp


def mu(b, I, mu0, mu1):
    """Recovery rate.
    
    """
    # recovery rate, depends on mu0, mu1, b
    mu = mu0 + (mu1 - mu0) * (b/(I+b))
    return mu

def R0(beta, d, nu, mu1):
    """
    Basic reproduction number.
    """
    return beta / (d + nu + mu1)

def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.
    """
    c0 = b**2 * d * A
    c1 = b * ((mu0-mu1+2*d) * A + (beta-nu)*b*d)
    c2 = (mu1-mu0)*b*nu + 2*b*d*(beta-nu)+d*A
    c3 = d*(beta-nu)
    res = c0 + c1 * I + c2 * I**2 + c3 * I**3
    return res
    

def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    SIR model including hospitalization and natural death.
    
    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    d
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons
    """
    S,I,R = y[:]
    m = mu(b, I, mu0, mu1)
    
    # calculate the total number of population
    N = S+I+R 
    
    # Define the rate of change of each population
    dSdt = A-d*S-(beta*S*I)/N  #susceptible population
    dIdt = -(d+nu)*I-m*I+(beta*S*I)/N   #infected population
    dRdt = m*I-d*R    #recovered population
    
    return [dSdt, dIdt, dRdt]

def plot_SIR(sol, mu0, mu1, beta, A, d, nu, b):
    """
    Plots the results of a Susceptible-Infected-Removed (SIR) model simulation.

    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    d
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons

    """

    # Subplot 1: Susceptible, Infective, and Removed populations
    fig,ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].plot(sol.t, sol.y[0]-0*sol.y[0][0], label='1E0*susceptible');
    ax[0].plot(sol.t, 1e3*sol.y[1]-0*sol.y[1][0], label='1E3*infective');
    ax[0].plot(sol.t, 1e1*sol.y[2]-0*sol.y[2][0], label='1E1*removed');
    ax[0].set_xlim([0, 500])
    ax[0].legend();
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"$S,I,R$")
    
    # Subplot 2: Recovery rate and Infective population
    ax[1].plot(sol.t, mu(b, sol.y[1], mu0, mu1), label='recovery rate')
    ax[1].plot(sol.t, 1e2*sol.y[1], label='1E2*infective');
    ax[1].set_xlim([0, 500])
    ax[1].legend();
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$\mu,I$")

    # Subplot 3: Indicator function h(I)
    I_h = np.linspace(-0.,0.05,100)
    ax[2].plot(I_h, h(I_h, mu0, mu1, beta, A, d, nu, b));
    ax[2].plot(I_h, 0*I_h, 'r:')
    #ax[2].set_ylim([-0.1,0.05])
    ax[2].set_title("Indicator function h(I) with b="+ "{:.3f}".format(b))
    ax[2].set_xlabel("I")
    ax[2].set_ylabel("h(I)")

    # Adjust subplot spacing
    fig.tight_layout()

def plot_SIR_trajectory(b,time,sol_0,sol_1,sol_2,SIM0_0,SIM0_1,SIM0_2):
    """
    Plot SIR trajectories for different simulations.
    
    Parameters:
    -----------
    b
        hospital beds per 10,000 persons.
    time
        the time points.
    sol_0, sol_1, sol_2
        Solutions of the SIR model for different simulations.
        Each solution contains values for S, I, and R compartments.
    SIM0_0, SIM0_1, SIM0_2
        different initial conditions for the simulations.
    """
    # Add the first subplot with 3D projection
    fig=plt.figure(figsize=(15,6))
    ax1=fig.add_subplot(131,projection="3d")
    plot_SIR_tra(ax1,b,sol_0,SIM0_0,time)
    
    # Add the second subplot with 3D projection
    ax2=fig.add_subplot(132,projection="3d")
    plot_SIR_tra(ax2,b,sol_1,SIM0_1,time)
    
    # Add the third subplot with 3D projection
    ax3=fig.add_subplot(133,projection="3d")
    plot_SIR_tra(ax3,b,sol_2,SIM0_2,time)
    
    # Adjust the spacing between subplots
    fig.tight_layout()
    
def plot_SIR_tra(ax,b,sol,SIM0,time):
    """
    Plot an SIR trajectory in a 3D space.
    
    Parameters:
    -----------
    ax
        Axes3D object representing the subplot where the SIR trajectory will be plotted.
    b
        hospital beds per 10,000 persons.
    sol
        Solution of the SIR model for a specific simulation.
             The solution should contain values for S, I, and R compartments.
    SIM0
        initial condition for the simulation.
    time
        the time points.
    """
    ax.plot(sol.y[0], sol.y[1], sol.y[2], 'r-',label=SIM0);
    
    # Scatter plot of individual points with colors based on time
    ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap='bwr');

    ax.set_xlabel("S")
    ax.set_ylabel("I")
    ax.set_zlabel("R")
    
    # Add a legend to the plot
    ax.legend(bbox_to_anchor=(1.04,0.9), loc='upper right')
    
    # Set the title of the plot based on the value of 'b'
    ax.set_title("SIR trajectory with b="+ "{:.3f}".format(b)) 
    


# Define the SIR model ODE
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# Solve the ODE for different parameter values and plot the results
def plot_backward_bifurcation(beta_values):
    fig, ax = plt.subplots()

    for beta in beta_values:
        # Set the recovery rate
        gamma = 0.1

        # Solve the ODE for a given parameter value
        sol = solve_ivp(lambda t, y: sir_model(t, y, beta, gamma), [0, 100], [0.99, 0.01, 0], t_eval=np.linspace(0, 100, 1000))

        # Plot the solution
        ax.plot(sol.t, sol.y[1], label=f'beta={beta}')

    # Set plot properties
    ax.set_xlabel('Time')
    ax.set_ylabel('Infected')
    ax.set_title('SIR Model')
    ax.legend()
    plt.show()