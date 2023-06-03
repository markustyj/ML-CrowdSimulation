#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 18 16:23:39 2023
@author: Yongjian Tang
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def logistic_map (r, x):
    """
    _summary_      define the function of logistic map
    _parameters_   r, mutable parameter in this function
    _parameters_   x, parameter that is updated recursively
    _Returns_      x in the next step  
    """
    x_next = r * x * (1-x)
    return x_next

def logistic_map_vis(r, x0, n):
    """_summary_       visualize the logistic map and plot the zigzag trajectories based on the parameter r and fixed initial point x0

    Parameters
    ----------
    r : _type_         float
        _description_  one parameter in the logistic map
    x0 : _type_        float
        _description_  the initial point of the trajectory and will be updated recursively
    n : _type_         int
        _description_  the number of iterations we want the algorithm to run 
    """
    #plot the logistic map itself generally 
    t = np.linspace(0, 1)
    f = logistic_map(r,t)
    plt.plot(t, f, 'k', lw=3)
    plt.plot([0, 1], [0, 1], 'k', lw=3)

    # plot the descrete movement given an initial point
    x = x0
    for step in range(n):
        y = logistic_map(r, x)
        # Plot the zigzag line to make the trajectory more readable
        plt.plot([x, x], [x, y], color='blue', marker='o', lw=1)
        plt.plot([x, y], [y, y], color='blue', marker='o', lw=1)
        x = y

    #keep the same format and scale for all plots    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"$r={r:.1f}, \, x_0={x0:.1f}$")
    plt.show()


def logistic_map_bifurcation_vis(r_min, r_max, r_states):
    """ _summary_      visualize the bifurcation diagram of the logistic map

    Parameters    
    ---------- 
    r_min : _type_     float
        _description_  the minimun range for parameter r
    r_max : _type_     float
        _description_  the maximun range for parameter r
    r_states : _type_  int
        _description_  the number of different r between r_min and r_max
    """
    r = np.linspace(r_min,r_max,r_states)
    x0 = 1e-5 * np.ones(r_states) # choose 10^-5 as the initial points for all r_states

    x = x0
    # assuming that we can reach steady state after 1000 iterations
    for i in range(1000):
        x = logistic_map(r, x)
        # only plot the points (stable points) when iterations have been run sufficiently, i.e. points that have reached the steady state
        if i > 900: 
            plt.scatter(r,x, s =0.1)
    plt.title("The bifurcation diagram of logistic map")
    plt.xlabel("r")
    plt.ylabel("x")
    plt.show()        


def lorenz_attractor(xyz0 = [10,10,10], T_end = 1000, sigma=10, beta=8/3, rho=28):
    """_summary_       The function to get the trajectory of a lorenz attractor,
                       all necessary parameters for lorenz attractor are given in the input 
    Parameters
    ----------
    xyz0 : list,       
        _description_, the coordinate of the initial point in x, y, z axis
    T_end : int, 
        _description_, the end of the simulation time, not the iteration count
    sigma : int, 
        _description_, parameter in lorenz attractor
    beta : _type_, 
        _description_, parameter in lorenz attractor
    rho : int, 
        _description_, parameter in lorenz attractor

    Return
    ----------    
    output : list
        _description_, the output of solve_ivp based on the input lorenz system,
                       which includes the list of time series, list of coordinates in x, y, z axis along the whole simulation time
    """
    def lorenz_system( t, xyz0=xyz0 , sigma=sigma, beta=beta, rho=rho):
        """_summary_  construct lorenz_system based on the mathematical formular"""
        x, y, z = xyz0
        return [sigma*(y-x),  x*(rho-z)-y,  x*y-beta*z]

    t_span = [0,T_end]
    t_eval = np.linspace(0,T_end,100000)
    output = solve_ivp(fun=lorenz_system, t_span=t_span, y0=xyz0, t_eval=t_eval)

    return output

def plot_3d_traj(output):
    """_summary_      Function to plot the trajectory of a specific lorenz system

    Parameters
    ----------
    output : _type_   object
        _description_ the result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(output.y[0], output.y[1], output.y[2], label='lorenz attractor [10,10,10]', color = "red", linewidth = 0.3)
    ax.legend()


def plot_3d_traj_compare(output1,output2):
    """_summary_       Function to plot the trajectory of 2 specific lorenz systems, compare their differences

    Parameters
    ----------
    output1 : _type_   object
        _description_  the first result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    output2 : _type_   object
        _description_  the second result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    """
    #Visualization of the first trajectory
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(output1.y[0], output1.y[1], output1.y[2], label='lorenz attractor [10,10,10]', color = "red", linewidth = 0.2)
    ax.plot(output2.y[0], output2.y[1], output2.y[2], label='lorenz attractor [10+1e-8,10,10]', color = "yellow", linewidth = 0.2)
    ax.legend()
    #Visualization of the second trajectory
    ax2 = plt.figure().add_subplot(projection='3d')
    ax2.plot(output1.y[0][80000:], output1.y[1][80000:], output1.y[2][80000:], label='lorenz attractor [10,10,10]', color = "red", linewidth = 0.3)
    ax2.plot(output2.y[0][80000:], output2.y[1][80000:], output2.y[2][80000:], label='lorenz attractor [10+1e-8,10,10]', color = "yellow", linewidth = 0.3)
    ax2.legend()
    plt.show()     

def plot_3d_traj_rho5(output):
    """_summary_      Function to plot the trajectory of a specific lorenz system
                      almost same as the above function plot_3d_traj(). 
                      The only difference is linewidth, which can visualize our results better for the case that rho = 0.5
    Parameters
    ----------
    output : _type_   object
        _description_ the result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(output.y[0], output.y[1], output.y[2], label='lorenz attractor', color = "red", linewidth = 2)
    ax.legend()

def plot_3d_traj_compare_rho5(output1,output2):
    """_summary_       Function to plot the trajectory of 2 specific lorenz systems, compare their differences
                       almost same as the above function plot_3d_traj(). 
                       The only difference is linewidth, which can visualize our results better for the case that rho = 0.5
    Parameters
    ----------
    output1 : _type_   object
        _description_  the first result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    output2 : _type_   object
        _description_  the second result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    """
    #Visualization of the first trajectory
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(output1.y[0], output1.y[1], output1.y[2], label='lorenz attractor [10,10,10]', color = "red", linewidth = 2)
    ax.plot(output2.y[0], output2.y[1], output2.y[2], label='lorenz attractor [10+100,10+100,10]', color = "yellow", linewidth = 2)
    ax.legend()
    #Visualization of the second trajectory
    ax2 = plt.figure().add_subplot(projection='3d')
    ax2.plot(output1.y[0][80000:], output1.y[1][80000:], output1.y[2][80000:], label='lorenz attractor [10,10,10]', color = "red", linewidth = 0.3)
    ax2.plot(output2.y[0][80000:], output2.y[1][80000:], output2.y[2][80000:], label='lorenz attractor [10+100,10+100,10]', color = "yellow", linewidth = 0.3)
    ax2.legend()
    plt.show()    

def difference_vis(output1, output2):
    """_summary_      calculate the difference between 2 trajectories
                      plot the value of difference in a 2D figure
    Parameters
    ----------
    output1 : _type_   object
        _description_  the first result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    output2 : _type_   object
        _description_  the second result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    """
    diff = np.sqrt ( np.sum( (output1.y-output2.y)**2  , axis=0 ) ) 
    for index in range(len(diff)):
        if diff[index] > 1:
            print(f"\n \n The first time that difference is larger than 1 takes place at time point: {index/100} \n\n\n" )
            break
    # plot the change of difference during the whole simluation time
    plt.plot(output1.t, diff )
    plt.title("Difference between 2 lorenz attractors, T_end = 1000")
    plt.xlabel("time")
    plt.ylabel("difference")
    plt.show() 
    # plot the change of difference when t <= 50, 
    # in order to observe the "small perturbations in the initial condition will grow larger at an exponential rate"
    plt.plot(output1.t[:5000], diff[:5000] )
    plt.title("Difference between 2 lorenz attractors, T_end = 50")
    plt.xlabel("time")
    plt.ylabel("difference")
    plt.show() 

