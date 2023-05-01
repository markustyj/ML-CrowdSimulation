#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:57:14 2023

@author: jingzhang
"""

import numpy as np


class Pedestrian:
    """
    Defines a single pedestrian.
    
    """

    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed

    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        
        :param scenario: The scenario instance.
        :return: a list of neighbor cell indices (x,y) around the current position.
        
        """
        
        return [
            (int(x + self._position[0]), int(y + self._position[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + self._position[0] < scenario.cell_width and 0 <= y + self._position[1] < scenario.cell_height and np.abs(x) + np.abs(y) > 0
        ]


    def update_step(self, scenario):
        """
        Moves to the cell with the lowest distance to the target.
        This takes obstacles and other pedestrians into account.
        Pedestrians will avoid cells already occupied by obstacles and other pedestrians when moving

        :param scenario: The current scenario instance.
        :param self._position: Pedestrian current position.
        :param neighbors_all: all neighbor cells.
        :param obstacles: get all positions of obstacles.
        :param pedes: get all positions of Pedestrian.
        :param targets: get all positions of targets.
        :param neighbors_without_obs: neighbor cells that are not occupied by obstacles and other pedestrians.
        :param next_pos: new position that Pedestrian will move to.
        
        :return: a list of Pedestrians positions.
        
        """
        
        neighbors_all = self.get_neighbors(scenario)
        print("-------------------------")
        print('Pedestrian current position: ')
        print(self._position)
        print('all neighbor cells')
        print(neighbors_all)
        
        
        obstacles = []          
        for obs_dict in scenario.obs_coordinates:
            obstacles.append((obs_dict['x'], obs_dict['y']))
        print("obstacles: ")
        print(obstacles)    
        
        pedes = []          
        for ped_dict in scenario.ped_coordinates:
            pedes.append((ped_dict['x'], ped_dict['y']))
            
            
        targets = []          
        for tar_dict in scenario.tar_coordinates:
            targets.append((tar_dict['x'], tar_dict['y']))
                      
        print("targets: ")
        print(targets) 
          
          
        neighbors_without_obs = [(x,y) for (x,y) in neighbors_all if (x,y) not in obstacles]
        neighbors = [(x,y) for (x,y) in neighbors_without_obs if (x,y) not in pedes]
        print("neighbors without obs and pedes: ")
        print(neighbors)
        
        
        next_cell_distance = scenario.target_distance_grids[self._position[0]][self._position[1]]

        next_pos = self._position
        
        for (n_x, n_y) in neighbors:           
            if(n_x, n_y) in targets:
                next_pos = (n_x, n_y)
                
            elif next_cell_distance >= scenario.target_distance_grids[n_x, n_y]:
                next_pos = (n_x, n_y)
                next_cell_distance = scenario.target_distance_grids[n_x, n_y]           
        
        
        self._position = next_pos
        
        print('next position: ')
        print(next_pos)
        
        return pedes
        
        