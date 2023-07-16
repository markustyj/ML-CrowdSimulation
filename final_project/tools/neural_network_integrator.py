#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 14 16:23:39 2023
@author: Yongjian Tang, Yun Di
"""
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch import mul
from torch.utils.data import DataLoader
from scipy.integrate import solve_ivp
from tools.eulers_method import *
from tools.runge_kutta_method import *


def set_seed(seed):
    """set the random seeds to rugulate the randomness.

    Parameters
    ----------
        seed: int
            the particuler number of seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    model,
    lr,
    epochs,
    train_data,
    val_data,
    batch_size,
    early_stopping_step=1,
    method="euler",
):
    """train the defined neural network, according to the given hyperparameters

    Parameters
    ----------
    model : nn.Module
        the defined neural network structure with Pytorch
    lr : float
        learning rate
    epochs : int
        epochs for the training
    train_data : numpy array
        split training data, size (N, 2, 3)
    val_data : numpy array
        split validation data, size (N, 2, 3)
    batch_size : int
        batch size during for training
    early_stopping_step : int, by default 1
        how many times is validation loss allowed to be larger than minimum loss. Otherwise, training stops.

    Returns
    -------
    training loss: list
        a list of training loss
    validation loss: list
        a list of validation loss
    """
    # use Dataloader in Pytorch to load the training and validation dataset
    train_set = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_set = DataLoader(val_data, shuffle=True)
    # 2 empty numpy array to record the loss
    train_loss_data = np.zeros(epochs * len(train_set))
    val_loss_data = np.zeros(epochs * len(train_set))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    optimizer = optim.Adam(model.parameters(), lr)
    model.weights_init()
    

    batch_nr = 0
    count = 0
    min_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for data in train_set:
            xk, xk1, time = (
                data[:, :, 0].float(),
                data[:, :, 1].float(),
                data[:, :, 2].float(),
            )
            optimizer.zero_grad()

            dt = time[:, 1] - time[:, 0]
            dt = dt.reshape(dt.size(dim=0), 1)
            # get xk dot as output

            xk = xk.to(device) 
            xk1 = xk1.to(device)
            dt = dt.to(device)
            output = model(xk)

            # Eulers_method(xk,output,dt) calculate the exactlly output of mdeol, which is th——e next step of simulation.
            if method == "euler":
                loss = func.mse_loss(xk1, eulers_method(xk, output, dt))
            elif method == "kutta":
                loss = func.mse_loss(xk1, runge_kutta_method(xk, output, dt))
            else:
                KeyError("The input method is invalid.")
            train_loss_data[batch_nr] = loss

            batch_nr += 1
            # Backpropagation of neural network.
            loss.backward()
            optimizer.step()

        model.eval()
        # val_loss = 0

        with torch.no_grad():
            current_loss, val_loss_data = cal_validation(
                val_set, val_loss_data, model, batch_nr, method
            )

        # Once the min_loss is smaller than the current loss, then trigger the early stopping mechanism.
        if early_stopping(current_loss, min_loss, early_stopping_step, count) == True:
            break
        else:
            min_loss, count = early_stopping(current_loss, min_loss, early_stopping_step, count)

    return train_loss_data, val_loss_data


def cal_validation(val_set, val_loss_data, model, batch_nr, method):
    """calculate the validation loss for each epoch to avoid overfitting

    Parameters
    ----------
    val_set : DataLoader format
        DataLoader format of validation dataset
    val_loss_data : list
        a list of validation loss in each epoch
    model : nn.Module
        the defined neural network structure with Pytorch
    batch_nr : int
        the current batch index in a epoch

    Returns
    -------
    current_loss : the latest validation loss
    val_loss_data : the list of validation loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_loss = 0
    for vdata in val_set:
        xk, xk1, time = (
            vdata[:, :, 0].float(),
            vdata[:, :, 1].float(),
            vdata[:, :, 2].float(),
        )
        dt = time[:, 1] - time[:, 0]
        dt = dt.reshape(dt.size(dim=0), 1)

        xk = xk.to(device) 
        xk1 = xk1.to(device)
        dt = dt.to(device)

        output = model(xk)
        # Eulers_method(xk,output,dt) calculate the exactlly output of mdeol, which is the next step of simulation.
        if method == "euler":
            loss = func.mse_loss(xk1, eulers_method(xk, output, dt))
        elif method == "kutta":
            loss = func.mse_loss(xk1, runge_kutta_method(xk, output, dt))
        else:
            KeyError("The input method is invalid.")
        val_loss += loss
    current_loss = val_loss / len(val_set)
    val_loss_data[batch_nr - 1] = current_loss

    return current_loss, val_loss_data


def early_stopping(current_loss, min_loss, early_stopping_step, count):
    """implementation of early stopping technique

    Parameters
    ----------
    current_loss : float
            the loss of one epoch
    min_loss : float
            the minimun loss until this epoch
    early_stopping_step : int
        how many times is validation loss allowed to be larger than minimum loss. Otherwise, training stops.
    count : int
        count how many times validation loss is larger than minimum loss.

    Returns
    -------
    updated min_loss, updated count
    """
    if current_loss < min_loss:
        min_loss = current_loss
        count = 0
    else:
        count = count + 1

    if count > early_stopping_step:
        return True
    else:
        return min_loss, count


def get_sim_data(filepath: str, ped_number: int):
    """extract the trajectory path of the pedestrain.

    Parameters
    ----------
        filepath:string
            the path to the datasets
        ped_number: int
            the number id of pedestrain

    Returns
    -------
        traj.T: np.array
            the trajectory of pesdestrian which are consist with x-y position.
        sim_steps: int
            the number of steps for the pedestrain.
    """
    df = pd.read_csv(filepath, delimiter=" ")
    org_df = df.groupby("pedestrianId")
    sim_steps = org_df.size()

    traj = np.zeros([2, sim_steps[ped_number]])
    traj[0, :] = org_df.get_group(ped_number)["startX-PID1"].to_numpy()
    traj[1, :] = org_df.get_group(ped_number)["startY-PID1"].to_numpy()
    return traj.T, sim_steps


def neural_network_function(
    t: np.array, x: torch.Tensor, model: torch.nn.Module
) -> np.ndarray:
    """represent the differential function of neural network.

    Parameters
    ----------
        t: np.array
            the vector of time
        x : torch.Tensor
            one of positions of trajectory
        model: nn.Module
            the trained neural network

    Returns
    -------
        diff: np.ndarray

    """
    ## the neural network need tensor input, solve_ivp is implemented using numpy
    
    x = torch.from_numpy(x)
    diff = model(x.float()).detach()
    return diff
