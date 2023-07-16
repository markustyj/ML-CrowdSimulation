#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 16:46:49 2023

@author: jingzhang, Mengyue Wang
"""
import torch
from tools.neural_network_integrator import *
import tools.neural_network_integrator as net


def runge_kutta_method(x, x_dot, dt):
    """based on the current coordinate, approximate the coordinate of x in the next step with dt

    Parameters
    ----------
    x : numpy array
        the current coordinate (x, y)
    x_dot : numpy array
        the output of the neural network in one iteration, which is used as x dot
    dt : numpy array
        the difference of time between current step and next step

    Returns
    -------
    x : numpy array
        updated coordinate using runge kutta method
    """
    k1 = x_dot
    k2 = x_dot + 0.5 * dt * k1
    k3 = x_dot + 0.5 * dt * k2
    k4 = x_dot + dt * k3
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulateRandomKutta(
    num_selected_peds, maximun_num_pedestrians, time, model, data_filepath
) -> np.ndarray:
    """simulate the trajectories of randomly selected pedestrains

    Parameters
    ----------
    num_selected_peds : int
        the number of pedestrians we want to simluate, e.g. 20 pedestrians from all 500 pedestrians
    maximun_num_pedestrians : int
        maximum number of pedestrians in a given scenario, e.g. 500 pedestrians
    time : float
        the end time of the simulation
    model : nn.Module
        our neural network model
    data_filepath : string
        the path to the dataset

    Returns
    -------
    tra : numpy array
        the full trajectory we get for the randomly selected pedestrians until T_ends
    """

    res = [
        random.randrange(1, maximun_num_pedestrians) for _ in range(num_selected_peds)
    ]
    traj = np.zeros([num_selected_peds * 2, time])

    for i, ped in enumerate(res):
        traj[2 * i : 2 * i + 2, :] = simulateKutta(model, ped, time, data_filepath)
    return traj


def simulateKutta(
    model: torch.nn.Module, pedestrainID: int, time: float, data_filepath: str
) -> np.ndarray:
    """simulate the trajectory of a pedstrain

    Parameters
    ----------
        model : nn.Module
            our neural network model
        pedestrainID : int
            the id number of pedestrain
        time : float
            the end time of the simulation
        data_filepath : string
            the path to the dataset

    Returns
    -------
        traj : numpy array
        A trajectory we get for the selected pedestrian until T_ends
    """

    traj = np.zeros([2, time])
    sim_data, _ = net.get_sim_data(data_filepath, pedestrainID)
    x0 = sim_data[0, :]

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #time = torch.tensor(time).to(device) 
    #x0 = torch.tensor(x0).to(device)
    model.to("cpu")

    sol = solve_ivp(
        net.neural_network_function,
        [0, time],
        x0,
        args=[model],
        t_eval=torch.tensor( np.arange(0, time) ),
        method="RK45",
    )
    traj[0, :] = sol.y[0]
    traj[1, :] = sol.y[1]
    return traj
