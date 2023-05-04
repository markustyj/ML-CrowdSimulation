#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 18 16:23:39 2023

@author: Yongjian Tang
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation


state_space = {
    "E": 0,
    "P":1,
    "O":2,
    "T":3,
    }
    

def get_color_map():    
    """_summary_
        The method that makes the grid colourful. 
        The colors represent empty, pedestrian, obstacle, target respectively
    Returns
    -------
        _description_  A colorful map that assign 4 different cell states with 4 colours
    """
    cmap = colors.ListedColormap(['white', 'blue', 'red', 'green']) 
    return cmap

def get_mark(states, ax):
    """_summary_
        mark the cell and add a dot in the middle if the cell is the path of pedestrians
    Parameters
    ----------
    states : _type_      2d numpy array
        _description_    states of all cells on the grid
    ax : _type_          plot figure
        _description_    figure ax in plt.subplot
    """
    for (i,j), state in np.ndenumerate(states): 
        if state == state_space["P"]:
            ax.text(j, i, ".", fontsize=18, ha='center', va='center')
            
def get_text(states, ax):
    """_summary_
         write the text "E", "P", "O", "T" in the middle of the cell
    Parameters
    ----------
    states : _type_      2d numpy array
        _description_    states of all cells on the grid
    ax : _type_          plot figure
        _description_    figure ax in plt.subplot
    """
    for (i,j), state in np.ndenumerate(states): 
        if state == state_space["P"]:
            ax.text(j, i, "P", ha='center', va='center')
        if state == state_space["O"]:
            ax.text(j, i, "O", ha='center', va='center')
        if state == state_space["T"]:
            ax.text(j, i, "T", ha='center', va='center')  
            
def get_grid(states, ax):
    """_summary_
        replace axis, labels with grid, beautify the outlook of the grid output
    Parameters
    ---------- 
    same as above 2 methods
    """

    ax.tick_params(bottom=False, top=False, left=False, right=False, 
                   labelbottom=False, labelleft=False, labelright=False, labeltop=False )
    ax.set_xticks(np.arange(-.5, states.shape[1], 1))
    ax.set_yticks(np.arange(-.5, states.shape[0], 1))
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.28)  
    
    
def get_plot_states(states):
    """_summary_
        The method that combines get_color_map(), get_text(), get_grid() together and visualize the states
    Parameters
    ----------
    states : _type_      2d numpy array
     _description_       states of all cells on the grid
    Returns
    -------
    _type_               plt.subplot figure
        _description_    visualized results of states on grid
    """
    #add basic figure
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    #add color
    ax.matshow(states, cmap=get_color_map())
    #add text, e.g. 
    get_text(states, ax)
    #remove labels and add grid
    get_grid(states, ax)
    
    return fig 
    
def get_plot_all_simulation_states(simulation_scenario_states, scenario):
    """_summary_
        get the animation of moveing pedestrians to the target
    Parameters
    ----------
    simulation_scenario_states : _type_   3-d numpy array, i.e. 2-d states with time series
        _description_                     a series of all grid states after running automaton, from the initial state to the final target
    scenario : _type_                     scenario object
        _description_                     the scenario created based on the options in the json file

    Returns
    -------      
        _description_                     a gif or a hdml video of the whole simulation process
    """
    #create basic figure
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    plot = plt.matshow(simulation_scenario_states[0], fignum=0, cmap=get_color_map())
    
    #remove labels and add grid
    get_grid(simulation_scenario_states[0], ax)
    get_text(simulation_scenario_states[0], ax)

    def update(i):
        """_summary_
            Updated state of the animation in each time step
        """
        if i<len(simulation_scenario_states)-1:
            get_mark(simulation_scenario_states[i+1], ax)
        plot.set_data(simulation_scenario_states[i])
        return [plot]
    
    def init():
        """_summary_
            Initial state of the animation
        """
        plot.set_data(simulation_scenario_states[0])
        return plot        
        

    return animation.FuncAnimation(fig, update, init_func=init, frames=len(simulation_scenario_states), interval = 300, repeat=False)    
    
    