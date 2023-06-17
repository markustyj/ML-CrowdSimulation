#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 14 16:23:39 2023
@author: Yongjian Tang
"""

import numpy as np
from numpy.linalg import lstsq
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist


def load_dataset_task3():
    """_summary_
                load the 2 original datasets x0 and x1 Returns
    _return_        
    -------
    _type_         numpy array
    _description_  shape(N,2), the loaded 2d dataset
    """
    data_x0 = np.loadtxt('data/nonlinear_vectorfield_data_x0.txt')
    data_x1 = np.loadtxt('data/nonlinear_vectorfield_data_x1.txt')
    return data_x0, data_x1



def get_finite_difference(x0,x1,dt):
    """_summary_
                    calculate the vector field given 2 initial point datasets and ∆t
    Parameters
    ----------
    x0 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the coordinates of inital points
    x1 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the same points advanced with an unknown (to you) evolution operator 
    dt : _type_        float
        _description_  the small time step to approximate the vector field

    Returns
    -------
    _type_             numpy array
        _description_  shape(N,2), the approximate vector foe;d
    """
    return (x1 - x0) / dt



def get_mean_square_error(x1, x1_hat, dt):
    """_summary_
                    calculate mse between true x1 and estimated x1
    Parameters
    ----------
    x1 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the same points advanced with an unknown (to you) evolution operator 
    x1_hat : _type_    numpy array
        _description_  shape (N, 2, t_eval), estimated x1 along a time series  
    dt : _type_        float
        _description_  the small time step to approximate the vector field

    Returns
    -------
    _type_             list
        _description_  a list of mse error along the time series 
    """
    # keep T_end same as in get_estimate_x1()
    T_end = dt *19
    mse = []
    for i in range (x1_hat.shape[-1]):
        mse.append(mean_squared_error(x1,x1_hat[:,:,i]))
        # output the mse error using print(), when t == ∆t
        if np.abs(i * T_end / (x1_hat.shape[-1]-1) - dt) < 0.0001:  # actually is equal here
            print(f"\n if ∆t = {dt}, the mean square error of all individual points is {mean_squared_error(x1,x1_hat[:,:,i])}\n")
    
    return mse 



def get_mse(x1, x1_hat):
    """_summary_
                 simple mse function, only output the mse at certain time point t, no textual output
    Parameters
    ----------
    x1 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the same points advanced with an unknown (to you) evolution operator 
    x1_hat : _type_    numpy array
        _description_  shape (N, 2, t_eval), estimated x1 along a time series 

    Returns
    -------
    _type_             float
        _description_  the calculated mse error at certain time point t = T_end
    """
    return mean_squared_error (x1,x1_hat[:,:,-1]) 




def get_estimated_x1(x0, x1, dt):
    """_summary_
                       the function to get estimated x1 using the linear approximation
    Parameters
    ----------
    x0 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the coordinates of inital points
    x1 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the same points advanced with an unknown (to you) evolution operator 
    dt : _type_        float
        _description_  the small time step to approximate the vector field

    Returns
    -------
    _type_             numpy array
        _description_  shape (N, 2, t_eval), estimated x1 along a time series
    """
    #calculate the approximate vector field
    v = get_finite_difference(x0,x1,dt)
    #based on the calculated vectorfield, get linear matrix A with least square minimizations
    result = lstsq(x0, v, rcond=None)
    A_hat_T = result[0]
    #define the linear ode with the computed linear matrix A_hat_T
    def linear_fun(t, x0 = x0, A_hat_T = A_hat_T):
        return [x0 @ A_hat_T ]
    #set parameter for solve_ivp function
    T_end = dt * 19
    t_span = [0,T_end]
    t_eval = np.linspace(0,T_end,20)
    # get the trajectory for all 2000 points respectively
    x1_hat = []
    for i in range(2000):
        trajectory = solve_ivp(fun=linear_fun, t_span=t_span, y0=x0[i,:], t_eval=t_eval)
        x1_hat.append(trajectory.y)
    # x1_hat is of size N x 2 x 10,  10 is the length of t_eval set with linspace    
    return np.asarray(x1_hat)



def get_estimated_x1_rbf(x0, x1, dt, epsilon = 0.5, num_center = 300):
    """_summary_
                        the function to get estimated x1 using the nonlinear approximation
    Parameters
    ----------
    x0 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the coordinates of inital points
    x1 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the same points advanced with an unknown (to you) evolution operator 
    dt : _type_        float
        _description_  the small time step to approximate the vector field
    epsilon :          float
        _description_  parameter for radial basis function
    num_center :       int
        _description_  parameter for radial basis function

    Returns
    -------
    x1_hat:  _type_    numpy array
        _description_  shape (N, 2, t_eval), estimated x1 along a time series
    C_hat_T : _type_   numpy array
        _description_  shape (L,2), the list of coefficients cl  <—— phi(x) @ C_hat_T
    xl : _type_        numpy array
        _description_  shape (L,2), the list of centers for radial basis function    
    """
    #calculate the approximate vector field
    v = get_finite_difference(x0,x1,dt)
    # get the centers for rbf
    xl = get_center(x0=x0, num_center=num_center)
    # calculate phi(x) with rbf function
    phi_x = rbf(x0, epsilon= epsilon, xl = xl)
    # calculate C_hat_T with lstsq function
    C_hat_T = lstsq(phi_x, v, rcond=None)[0]
    #define the nonlinear ode 
    def nonlinear_fun(t, y, xl=xl, epsilon=epsilon, C_hat_T=C_hat_T ): 
        y = y.reshape(1, y.shape[-1])
        phi_x = np.exp(-cdist(y, xl) ** 2 / epsilon ** 2)
        return [phi_x @ C_hat_T ]   
    #set parameter for solve_ivp function
    T_end = dt 
    t_span = [0,T_end]
    t_eval = np.linspace(0,T_end,10)
    # get the trajectory for all 2000 points respectively
    x1_hat = []
    for i in range(2000):
        trajectory = solve_ivp(fun=nonlinear_fun, t_span=t_span, y0=x0[i,:], t_eval=t_eval)
        x1_hat.append(trajectory.y)
    # x1_hat is of size N x 2 x 10,  10 is the length of t_eval set with linspace    
    return np.asarray(x1_hat), C_hat_T, xl

def get_center(x0, num_center):
    """_summary_
                    select a given number of initial points as centers for radial basis functions
    Parameters
    ----------
    x0 : _type_          numpy array
        _description_    shape(N,2), 2d numpy array containing the coordinates of inital points
    num_center : _type_  int
        _description_    the desired number of centers for radial basis functions

    Returns
    -------
    _type_               numpy array
        _description_    shape(L,2), selected centers from x0
    """
    xl = x0[np.random.choice(range(x0.shape[0]), replace=False, size=num_center)] 
    return xl

def rbf(x, epsilon, xl ):
    """_summary_
                       compute the result of rbf function following the fomular
    Parameters
    ----------
    x  : _type_        numpy array
        _description_  shape (N,2), the list of dataset points 
    xl : _type_        numpy array
        _description_  shape (L,2), the list of centers for radial basis function
    epsilon : _type_   float
        _description_  the parameter in radial basis function

    Returns
    -------
    _type_             numpy array
        _description_  phi(x) of shape (N,L)
    """
    return np.exp(-cdist(x, xl) ** 2 / epsilon ** 2)

def find_best_conf(epsilon_list, num_center_list, dt = 0.01,):    
    """_summary_
                    find the best epsilon and L for nonlinear approximation
    Parameters
    ----------
    epsilon_list : _type_    list
        _description_        a list of epsilon value to be checked with grid search
    num_center_list : _type_ list
        _description_        a list of L value to be checked with grid search
    dt : float, 
        _description_, small time step, by default 0.01

    Returns
    -------
    the best ε and L as epsilon best and L best as well as the resulting best xl, C_hat_T , and the estimated x_hat_1
    """
    x0_original, x1_original = load_dataset_task3()
    mse_min = 100000
    mse_max = 0
    #dt_best, epsiplon_best, num_center_best = None
    for epsilon in epsilon_list:
        for num_center in num_center_list:
            x0, x1 = x0_original, x1_original
            x1_hat,C_hat_T, xl = get_estimated_x1_rbf(x0, x1, dt, epsilon, num_center)
            mse = get_mse(x1, x1_hat)
            if mse < mse_min:
                mse_min = mse
                epsilon_best = epsilon
                num_center_best = num_center    
                C_hat_T_best = C_hat_T
                xl_best = xl
                x1_hat_best = x1_hat
            if mse > mse_max:
                mse_max = mse
                epsilon_worst = epsilon
                num_center_worst = num_center  
    print(f"\nThe best configuration parameters are epsiplon_best:{epsilon_best}, num_center_best:{num_center_best}. \n \
          and the mse is {mse_min} \n \
        The worst configuration parameters are epsiplon_worst:{epsilon_worst}, num_center_worst:{num_center_worst}. \n \
          and the mse is {mse_max} ")                 
    return x1_hat_best, C_hat_T_best, xl_best, epsilon_best, num_center_best, epsilon_worst, num_center_worst  

def find_steady_states(x0, x1, C_hat_T, xl, epsilon):
    """_summary_
                    extend the t_eval and find out the steady states of nonlinear approximation
    Parameters
    ----------
    x0 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the coordinates of inital points
    x1 : _type_        numpy array
        _description_  shape(N,2), 2d numpy array containing the same points advanced with an unknown (to you) evolution operator 
    dt : _type_        float
        _description_  the small time step to approximate the vector field
    C_hat_T : _type_   numpy array
        _description_  shape (L,2), the list of coefficients cl  <—— phi(x) @ C_hat_T
    xl : _type_        numpy array
        _description_  shape (L,2), the list of centers for radial basis function  
    epsilon :          float
        _description_  parameter for radial basis function  

    Returns
    -------
    x1_hat:  _type_    numpy array
        _description_  shape (N, 2, t_eval), estimated x1 along a time series
                       the T_end is much larger than before here
    """
    #define the nonlinear ode 
    def nonlinear_fun(t, y, xl=xl, epsilon=epsilon, C_hat_T=C_hat_T ): 
        y = y.reshape(1, y.shape[-1])
        phi_x = np.exp(-cdist(y, xl) ** 2 / epsilon ** 2)
        return [phi_x @ C_hat_T ]   
    #set parameter for solve_ivp function
    T_end = 10
    t_span = [0,T_end]
    t_eval = np.linspace(0,T_end,100)
    # get the trajectory for all 2000 points respectively
    x1_hat = []
    for i in range(2000):
        trajectory = solve_ivp(fun=nonlinear_fun, t_span=t_span, y0=x0[i,:], t_eval=t_eval)
        x1_hat.append(trajectory.y)
       
    return np.asarray(x1_hat)



