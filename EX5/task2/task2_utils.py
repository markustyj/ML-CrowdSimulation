#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 20:26:59 2023

@author: jingzhang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def load_data():
    """
    This function loads the data for x0, x1 from txt files.
    
    :param x0: data for x0 from 'linear_vectorfield_data_x0.txt' file.
    :param x1: data for x1 from 'linear_vectorfield_data_x1.txt' file.
    
    Returns:
        the loaded data for x0 and x1.
    
    """
    
    x0 = np.loadtxt('linear_vectorfield_data_x0.txt')
    x1 = np.loadtxt('linear_vectorfield_data_x1.txt')
    
    return x0, x1


def estimate_vectors(delta_t):
    """
    This function uses the finite-difference formula to estimate the vectors v(k) 
    at all points x0(k), with a time step ∆t.
    
    :param x0, x1: loaded datasets.
    :param delta_t: time step ∆t.
    :param v: estimated vector.
    
    Returns:
        estimated vector.
    
    """    
    x0, x1 = load_data()
    v = (x1-x0)/delta_t
    
    return v


def approximate_matrix(x0, v):     
    """
    This function uses np.linalg.lstsq to approximate the matrix A.
    
    :param x0: loaded dataset x0.
    :param v: estimated vector.
    
    Returns:
        the matrix A.
    
    """   
    A = np.linalg.lstsq(x0, v, rcond=None)[0]
    
    return A


def plot_datasets(data_dict):    
    """
    This function plots datasets as scatterplot.
    
    :param data_dict: all datasets are saved as in a dictionary, e.g. {"x0":x0, "x1":x1}
    
    """    
    fig, ax = plt.subplots(1,1, figsize=(6,6)) 
    titles = []
    for title, data in data_dict.items():
        ax.scatter(data[:, 0], data[:, 1], s = 5, label = title)
        titles.append(title)
        
    ax.legend(loc ="upper right")
    plt.savefig('outputs/plot_datasets{}.png'.format(titles))


def compute_mse(linear_system, x0, x1, A, t_end):
    """
    This function computes the mean squared error (MSE) between the actual x1 and the estimated x1.
    
    :param x0_element: each element in the x0 dataset as initial point.
    :param x1_estimated: a list for storing all estimated x1, which are solved using the solve_ivp function
    :param mse: the mean squared error between actual error
    
    Returns:
        the mean squared error and the list x1_estimated.

    """
    x1_estimated = []
    for x0_element in x0:
        sol = solve_ivp(linear_system, [0, t_end], x0_element, t_eval=[t_end])
        x1_estimated.append(sol.y.flatten())

    x1_estimated = np.array(x1_estimated)
    mse = ((x1 - x1_estimated)**2).mean()
    print("Mean Squared Error:", mse)
    
    return mse, x1_estimated


def plot_approximate_vector_field(x0, x1, v):
    """
    This function plots the linear vector field using quiver.
    
    :param x0[:, 0], x0[:, 1]: the starting positions of the vectors.
    :param v[:, 0], v[:, 1]: the x and y components of the vectors.
    
    """
    
    fig, ax = plt.subplots(1,1, figsize=(6,6)) 
    ax.scatter(x0[:, 0], x0[:, 1], s = 5, label = '$x_0$')
    ax.scatter(x1[:, 0], x1[:, 1], s = 5, label = '$x_1$')
    ax.quiver(x0[:,0], x0[:,1], v[:,0], v[:,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc ="upper right") 
    plt.savefig('outputs/plot_approximate_vector_field.png')


def plot_each_dimension(x0, x1, x1_estimated):
    
    plt.scatter(x0[:,0], x1[:,0], c='blue', s=2, alpha=1, label='Actual')
    plt.scatter(x0[:,0], x1_estimated[:,0], c='orange', s=2, alpha=1, label='Estimated')
    plt.legend()
    plt.savefig('outputs/plot_first_dimension.png')
    plt.show()
    
    plt.scatter(x0[:,1], x1[:,1], c='blue', s=2, alpha=1, label='Actual')
    plt.scatter(x0[:,1], x1_estimated[:,1], c='orange', s=2, alpha=1, label='Estimated')
    plt.legend()
    plt.savefig('outputs/plot_second_dimension.png')
    plt.show()
    
    
    
def visualize_traj_phase_portrait(linear_system, A, t_end, initial_point):
    """
    This function visualizes the trajectory and the phase portrait with the initial point (10, 10).
    
    :param X, Y: set the range of the grid for plotting.
    :param U, V: calculate the velocity field for each point on the grid using the approximated matrix A.
    
    """
    w = 10
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]   
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)
    
    sol = solve_ivp(linear_system, [0, t_end], initial_point, t_eval=np.linspace(0, 100, 1000))
    
    plt.figure(figsize=(6, 6))
    plt.streamplot(X, Y, U, V, color="gray")
    plt.plot(sol.y[0,:],sol.y[1,:], c='red')
    plt.savefig('outputs/visualize_traj_phase_portrait.png')
    
    