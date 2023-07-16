#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 14 16:23:39 2023
@author: Yongjian Tang, Yun Di
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from tools.neural_network_integrator import neural_network_function
import matplotlib.image as mpimg


def plot_train_val_loss(train_loss_data, val_loss_data):
    """Plot the traing and validation loss

    Parameters
    ----------
    train_loss_data : list, float
        _description_ the loss of training set.
    val_loss_data :   list, float
        _description_ the loss of validation set.
    """
    # fix the scalar of the diagram
    #plt.axis([0, 10000, 0, 0.02])
    # plt.plot(train_loss_data, label="Training loss")

    val_loss_data_eva = np.asarray(
        [
            #[i, val_loss_data[i]]
            val_loss_data[i]
            for i in range(0, len(train_loss_data))
            if val_loss_data[i] > 1e-8
        ]
    )
    train_loss_data_eva = np.asarray(
        [
            #[i, train_loss_data[i]]
            train_loss_data[i]
            for i in range(0, len(train_loss_data))
            if val_loss_data[i] > 1e-8
        ]
    )
    # plot the training loss and validation loss in the same diagram
    plt.plot(
        #train_loss_data_eva[:, 0],
        #train_loss_data_eva[:, 1],
        train_loss_data_eva,
        label="train loss",
    )
    plt.plot(
        #val_loss_data_eva[:, 0],
        #val_loss_data_eva[:, 1],
        val_loss_data_eva,
        color="r",
        label="Validation loss",
    )
    plt.legend(loc="upper right")


def plot_traj(num_selected_peds, traj, background="corner"):
    """plot the trajectory of randomly selected trajectories

    Parameters
    ----------
    num_selected_peds : int
            the number of pedestrians, whose trajectories are supposed to be plotted
    traj : np.ndarray
            full trajectories of all pedestrians
    background: string
            choose the background of the figure
    """
    fig = plt.figure(figsize=(8, 8))

    for i in range(num_selected_peds):
        plt.plot(traj[2 * i, :], traj[2 * i + 1, :], label="Trajectory" + str(i))

    plt.title("Trajectory of selected pedestrians")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")

    if background == "corner":
        plot_scenario_corner()
    elif background == "4directions":
        plot_scenario_surrounding()
    else:
        pass


def plot_traj_supermarket(num, traj, img_name):
    """plot the trajectory of randomly selected trajectories in the supermarket scenario.

    Parameters
    ----------
    num : int
            the number of pedestrians, whose trajectories are supposed to be plotted
    traj : np.ndarray
            trajectories of all pedestrians
    
    """
    
    background_img = mpimg.imread("data/supermarket.jpg")
    
    fig = plt.figure(figsize=(8, 8))
    
    plt.imshow(background_img, extent=[0,40,0,40])

    for i in range(num):
        plt.plot(traj[2 * i, :], traj[2 * i + 1, :], label="Trajectory" + str(i))

    plt.title("Trajectory of selected pedestrians")
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.legend(loc="upper left")    
    plt.savefig("outputs/{}".format(img_name))


def plot_scenario_corner():
    """
    plot the background of the corner case scenario

    Parameters
    ---------- no parameters needed
    """

    # define the width and height of the scenario
    w = 40
    h = 40
    # Visualizing scenario
    plt.xlim([0, w])
    plt.ylim([0, h])
    # draw the position of the source
    plt.vlines(0.6, 4.6, 10.5, colors="green", linestyles="dashed")
    plt.vlines(13.9, 4.6, 10.5, colors="green", linestyles="dashed")
    plt.hlines(4.6, 0.6, 13.9, colors="green", linestyles="dashed")
    plt.hlines(10.5, 0.6, 13.9, colors="green", linestyles="dashed")
    # draw the target of the pedestrians
    plt.vlines(33, 31.5, 34.4, colors="orange", linestyles="dashed")
    plt.vlines(35, 31.5, 34.4, colors="orange", linestyles="dashed")
    plt.hlines(31.5, 33, 35, colors="orange", linestyles="dashed")
    plt.hlines(34.4, 33, 35, colors="orange", linestyles="dashed")
    # draw the obstacles 1
    plt.vlines(28.9, 13.2, 40, colors="gray")
    plt.vlines(31, 10.6, 40, colors="gray")
    plt.hlines(10.6, 0, 31, colors="gray")
    plt.hlines(13.2, 0, 28.9, colors="gray")
    # draw the obstacles 2
    plt.vlines(37.8, 4.4, 40, colors="gray")
    plt.vlines(39.5, 2.8, 40, colors="gray")
    plt.hlines(2.8, 0, 39.5, colors="gray")
    plt.hlines(4.4, 0, 37.8, colors="gray")


def plot_scenario_surrounding():
    """
    plot the background of the surrouding case scenario(one target in the middle, and four source surrounds it)

    Parameters
    ---------- no parameters needed
    """

    # define the width and height of the scenario
    w = 40
    h = 40
    # Visualizing scenario
    plt.xlim([0, w])
    plt.ylim([0, h])
    # draw the position of the source
    plt.vlines(19, 0.2, 2.2, colors="green", linestyles="dashed")
    plt.vlines(21, 0.2, 2.2, colors="green", linestyles="dashed")
    plt.hlines(0.2, 19, 21, colors="green", linestyles="dashed")
    plt.hlines(2.2, 19, 21, colors="green", linestyles="dashed")

    plt.vlines(0.2, 21, 23, colors="green", linestyles="dashed")
    plt.vlines(2.2, 21, 23, colors="green", linestyles="dashed")
    plt.hlines(21, 0.2, 2.2, colors="green", linestyles="dashed")
    plt.hlines(23, 0.2, 2.2, colors="green", linestyles="dashed")

    plt.vlines(19, 37.8, 39.8, colors="green", linestyles="dashed")
    plt.vlines(21, 37.8, 39.8, colors="green", linestyles="dashed")
    plt.hlines(37.8, 19, 21, colors="green", linestyles="dashed")
    plt.hlines(39.8, 19, 21, colors="green", linestyles="dashed")

    plt.vlines(37.8, 21, 23, colors="green", linestyles="dashed")
    plt.vlines(39.8, 21, 23, colors="green", linestyles="dashed")
    plt.hlines(21, 37.8, 39.8, colors="green", linestyles="dashed")
    plt.hlines(23, 37.8, 39.8, colors="green", linestyles="dashed")

    # draw the target of the pedestrians
    plt.vlines(21, 21, 23, colors="orange", linestyles="dashed")
    plt.vlines(19, 21, 23, colors="orange", linestyles="dashed")
    plt.hlines(23, 19, 21, colors="orange", linestyles="dashed")
    plt.hlines(21, 19, 21, colors="orange", linestyles="dashed")


def phase_portrait(
    model,
):
    """compute the derivations of each position in the scenario,then use the output to plot phase portrait

    Parameters
    ----------
        model : nn.Module
            our neural network model
    Returns
    -------
        X: np.meshgrid
            x position of each position in the scenario
        Y: np.meshgrid
            y position of each position in the scenario
        U: np.meshgrid
            the vectorize x position of a point in the scenario
        V: np.meshgrid
            the vectorize y position of a point in the scenario
    """
    WIDTH = 40
    HEIGHT = 40
    x = np.linspace(0, WIDTH)
    y = np.linspace(0, HEIGHT)
    X, Y = np.meshgrid(x, y)
    U, V = np.meshgrid(x, y)

    for j in range(len(Y)):
        for i in range(len(X)):
            position = np.array([X[j, i], Y[j, i]])
            # diffenrential of each position in the grid
            diffenrential_position = neural_network_function(None, position, model)
            # U V both are vector,which store the direction vector of each arrow in the grid
            U[j, i] = diffenrential_position[0]
            V[j, i] = diffenrential_position[1]
    return X, Y, U, V


def plot_phase_portrait(
    X: np.meshgrid,
    Y: np.meshgrid,
    U: np.meshgrid,
    V: np.meshgrid,
    background,
    save_filepath: str = None
):
    """plot the phase portrait of model.

     Parameters
    ----------
        X: np.meshgrid
            x position of each position in the scenario
        Y: np.meshgrid
            y position of each position in the scenario
        U: np.meshgrid
            the vectorize x position of a point in the scenario
        V: np.meshgrid
            the vectorize y position of a point in the scenario
        background: string
            Defaults to "4directions".could change to other background based on the datasets
    """
    fig = plt.figure(figsize=(12, 8))
    plt.title("Phase Portrait")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.streamplot(X, Y, U, V, density=[0.5, 1])

    if background == "corner":
        plot_scenario_corner()
    elif background == "4directions":
        plot_scenario_surrounding()
    else:
        pass

    if save_filepath:
        fig.savefig(save_filepath)
