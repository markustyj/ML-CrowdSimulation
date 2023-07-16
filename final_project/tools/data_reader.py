#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 14 16:23:39 2023
@author: Yongjian Tang, Yun Di
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def data_reader(file_path, train_percentage = 0.8, test_percentage = 0.2):
    """ Read the pedestrian dataset, and change the format to (N, 2, 3),
        the second dimension stands for (x, y) position of pedestrians.
        
        Split the dataset into training and validation dataset 

    Parameters
    ----------
    file_path : string
             path to the dataset of pedestrain trajectories
    train_percentage : float, by default 0.8
             the percentage of the training dataset
    test_percentage : float, by default 0.2
             the percentage of the validation dataset
    Returns
    -------
    df.columns : list
             the list of the useful column names
    df_selected : pandas dataframe
             the dataframe of the selected columns
    train_data : numpy array
             data for training the neural network
    train_data : numpy array
             data for validation during the training
    """
    df = pd.read_csv(file_path, delimiter=" ")
    columns_selected = ['startX-PID1', 'endX-PID1', 'simTime', 'startY-PID1', 'endY-PID1', 'endTime-PID1'] 
    df_selected = df[columns_selected]
    #split the dataset into training and validation
    train_split, val_split = train_test_split( df_selected, test_size=test_percentage, train_size=train_percentage, random_state=42, shuffle = True)
    print(train_split.iloc[0])

    # change pandas dataframe to numpy array
    train_data = train_split.to_numpy().reshape( (len(train_split),2,3) )
    val_data = val_split.to_numpy().reshape( (len(val_split),2,3) )
    return df.columns, df_selected, train_data, val_data


    