#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:27:30 2023

@author: jingzhang
"""

import pandas as pd

# read the output file that contains infection information
input_path = "/Users/jingzhang/Downloads/TUM_Studium/Praktikum/Ex02/task4_vadere_project/output/Task4_corridor_2023-05-14_16-05-03.394/SIRInformation.csv"

with open(input_path) as f:
    df = pd.read_csv(f, sep=" ")
       
# get Infected ID
ID_INFECTED = 0

# get the total number of infected pedestrians
infected_ped = 0
for ped in df['pedestrianId'].unique():
    if ID_INFECTED in df[df['pedestrianId'] == ped]['groupId-PID5'].values:
        infected_ped += 1 

# print the result        
print("Pedestrians get infected in this counter-flow: " + str(infected_ped))