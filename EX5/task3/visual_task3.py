#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 14 16:23:39 2023
@author: Yongjian Tang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist

def vis_nonlinear_vectorfield_data(x0, x1, v):
    """_summary_ visualize the two initial points x0 and x1
                 also plot the calculated vector field on the figure to have a general idea of the system dynamics 
    Parameters
    ----------
    x0 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the coordinates of inital points
    x1 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the same points advanced with an unknown (to you) evolution operator 
    v : _type_         numpy array
        _description_  shape(N,2), the calculated 2d vector field
    """
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x0[:, 0], x0[:, 1], s = 5, label='$data x_0$')
    plt.scatter(x1[:, 0], x1[:, 1], s = 5, label='$data x_1$')
    plt.title ("Initial plot")
    plt.legend()

    xd = v[:,0]
    yd = v[:,1]
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x0[:, 0], x0[:, 1], s = 5, label='$data x_0$')
    plt.scatter(x1[:, 0], x1[:, 1], s = 5, label='$data x_1$')
    plt.quiver(x0[:, 0], x0[:, 1], xd, yd)
    plt.title ("Initial plot with approximate vector field")
    plt.legend()

def vis_approximate_x1(x1, x1_hat):
    """_summary_       
                    visualize and compare the true points x1 and the estimated points x1_hat at t = ∆t 
                    here one visualization: t = ∆t
    Parameters
    ----------
    x1 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the same points advanced with an unknown (to you) evolution operator 
    x1_hat : _type_    numpy array
        _description_  estimated x1, shape (N, 2, t_eval), only use the case (N, 2, t_eval = ∆t) here
    """
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[:,0], x1[:,1], s = 8, label='data x1 original')
    plt.scatter(x1_hat[:,0, -1], x1_hat[:,1, -1], s = 8, label='data $\hat{x1}$ approximate')
    plt.title ("Original x1 data compared to approximate $\hat{x1}$")
    plt.legend()

def vis_approximate_x1_extend(x1, x1_hat):
    """_summary_       
                    almost same as the function vis_approximate_x1()
                    visualize and compare the true points x1 and the estimated points x1_hat at t = ∆t and t > ∆t
                    here 2 visualization: t = ∆t, t > ∆t
    Parameters
    ----------
    x1 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the same points advanced with an unknown (to you) evolution operator 
    x1_hat : _type_    numpy array
        _description_  estimated x1, shape (N, 2, t_eval), use the case (N, 2, t_eval = ∆t) and (N, 2, t_eval = T_end) here
    """
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[:,0], x1[:,1], s = 8, label='data x1 original')
    plt.scatter(x1_hat[:,0, 1], x1_hat[:,1, 1], s = 8, label='data $\hat{x1}$ approximate')
    plt.title ("Original x1 data compared to approximate $\hat{x1}$")
    plt.legend()

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[:,0], x1[:,1], s = 8, label='data x1 original')
    plt.scatter(x1_hat[:,0, -1], x1_hat[:,1, -1], s = 8, label='data $\hat{x1}$ approximate')
    plt.title ("Original x1 data compared to approximate $\hat{x1}$")
    plt.legend()


def plot_phase_portrait(A):
    """_summary_
                    Plots a linear vector field, defined with the matrix A.
                    same code given by Dr.Dietrich
    Parameters
    ----------
    A : _type_          numpy array
        _description_   shape(2,2), the calculated linear operator A
    """
    w = 4.5
    Y, X = np.mgrid[-w:w:500j, -w:w:500j]

    # calculate the vector U and V for streamline plot
    UV = A@ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Streamplot for linear vector field A*x')
    ax0.set_aspect(1)


def plot_phase_portrait_rbf(C_hat_T, xl, epsilon):
    """_summary_
                    Plots a nonlinear vector field for the nonlinear approximation using radial basis function
    Parameters
    ----------
    C_hat_T : _type_   numpy array
        _description_  shape (L,2), the list of coefficients cl  <—— phi(x) @ C_hat_T
    xl : _type_        numpy array
        _description_  shape (L,2), the list of centers for radial basis function
    epsilon : _type_   float
        _description_  the parameter in radial basis function
    """
    w = 4.5
    Y, X = np.mgrid[-w:w:300j, -w:w:300j]

    # define the nonlinear function used in approximation
    def nonlinear_fun(t, y, xl=xl, epsilon=epsilon, C_hat_T=C_hat_T ): 
        y = y.reshape(1, y.shape[-1])
        phi_x = np.exp(-cdist(y, xl) ** 2 / epsilon ** 2)
        return phi_x @ C_hat_T 

    # calculate the vector U and V for streamline plot
    U, V = [], []
    for x2 in X[0]:
        for x1 in Y[:, 0]: 
            res = nonlinear_fun(0, np.array([x1, x2]))
            U.append(res[0][0])
            V.append(res[0][1])
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[1, 1])
    ax0.set_title('Streamplot for nonlinear vector fields')
    ax0.set_aspect(1)    

def vis_steady_states(x1 ):
    """_summary_
                    visualize all steady states of a system
    Parameters
    ----------
    x1 : _type_        numpy array
        _description_  estimated x1, shape (N, 2, t_eval), use the cases (N, 2, t_eval = ∆t) and (N, 2, t_eval = T_end) here
    """
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x1[:, 0], x1[:, 1], s = 50, label='Steady states')
    plt.title ("Steady states obtained from the best L and $\epsilon$")
    plt.legend()    
    plt.xlim([-4.5, 4.5])
    plt.ylim([-4.5, 4.5])