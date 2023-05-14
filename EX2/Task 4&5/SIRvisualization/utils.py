# -*- coding: utf-8 -*-

import os
import pandas as pd
import plotly.graph_objects as go


def file_df_to_count_df(df,
                        ID_SUSCEPTIBLE=1,
                        ID_INFECTED=0,
                        ID_RECOVERED=2):
    """
    Converts the file DataFrame to a group count DataFrame that can be plotted.
    The ID_SUSCEPTIBLE and ID_INFECTED specify which ids the groups have in the Vadere processor file.
    """
    # Get unique pedestrian ids and simulation times
    pedestrian_ids = df['pedestrianId'].unique()
    sim_times = df['simTime'].unique()
    # Initialize group counts DataFrame
    group_counts = pd.DataFrame(columns=['simTime', 'group-s', 'group-i', 'group-r'])
    group_counts['simTime'] = sim_times
    group_counts['group-s'] = 0  # "susceptible" group
    group_counts['group-i'] = 0  # "infected" group
    group_counts['group-r'] = 0  # "recovered" group

    # loop over all the unique pedestrian IDs (pedestrians_ids)
    for pid in pedestrian_ids:
        # Get simulation times and group ids for current pedestrian
        simtime_group = df[df['pedestrianId'] == pid][['simTime', 'groupId-PID5']].values
        # Set current state of the pedestrian as "susceptible"
        current_state = ID_SUSCEPTIBLE
        # Loop through simulation times and update group counts
        group_counts.loc[group_counts['simTime'] >= 0, 'group-s'] += 1
        for (st, g) in simtime_group:
            if g != current_state and g == ID_INFECTED and current_state == ID_SUSCEPTIBLE:
                # Update group counts if pedestrian becomes infected
                # Reduce counts in the "susceptible" group and add counts in the "infected" group
                current_state = g
                group_counts.loc[group_counts['simTime'] > st, 'group-s'] -= 1
                group_counts.loc[group_counts['simTime'] > st, 'group-i'] += 1

            if g != current_state and g == ID_RECOVERED and current_state == ID_INFECTED:
                # Update group counts if pedestrian becomes recovered
                # Reduce counts in the "infected" group and add counts in the "recovered" group
                current_state = g
                group_counts.loc[group_counts['simTime'] > st, 'group-i'] -= 1
                group_counts.loc[group_counts['simTime'] > st, 'group-r'] += 1
                break
    return group_counts


def create_folder_data_scatter(folder):
    """
    Create scatter plot from folder data.
    :param folder:
    :return:
    """
    # Construct the file path for the SIRinformation.csv file in the folder
    file_path = os.path.join(folder, "SIRinformation.csv")
    # Check if the file exists
    if not os.path.exists(file_path):
        return None
    data = pd.read_csv(file_path, delimiter=" ")

    print(data)

    # Set the state values for the "susceptible", "infected" and "recovered" groups
    # Corresponding to the group IDs used in the simulation data
    ID_SUSCEPTIBLE = 1
    ID_INFECTED = 0
    ID_RECOVERED = 2

    # Convert the simulation data into a DataFrame of group counts over time
    group_counts = file_df_to_count_df(data, ID_INFECTED=ID_INFECTED, ID_SUSCEPTIBLE=ID_SUSCEPTIBLE,
                                       ID_RECOVERED=ID_RECOVERED)
    # group_counts.plot()
    # Create scatter plots for the number of individuals in each group over time
    scatter_s = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-s'],
                           name='susceptible ' + os.path.basename(folder),
                           mode='lines')
    scatter_i = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-i'],
                           name='infected ' + os.path.basename(folder),
                           mode='lines')
    # Create a scatter plot for the "recovered" group
    # Set the x-axis data as the simulation time values from the group_counts dataframe
    # Set the y-axis data as the group-r values from the group_counts dataframe
    scatter_r = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-r'],
                           name='recovered' + os.path.basename(folder),
                           mode='lines')
    return [scatter_s, scatter_i, scatter_r], group_counts
