#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 18 16:23:39 2023

@author: Yongjian Tang
"""

import math
import numpy as np
import random
import scipy.spatial.distance

state_space = {
    "E":0,
    "P":1,
    "O":2,
    "T":3,
    }


class Scenario: 
    """
    class of scenario, used to create object of different scenario object based on the given json scenario file
    """
  
    def __init__(self, scenario_setting):
        """_summary_
            preprocess the json data to create basic information for grid visualization and simulation
        Parameters
        ----------
        scenario_setting : _type_    dictionary
            _description_            json file that contains the basic information of a scenario
        """
        self.cell_size = scenario_setting["cell_size"]
        self.number_of_cells_x = math.floor(scenario_setting["width"]/scenario_setting["cell_size"])
        self.number_of_cells_y = math.floor(scenario_setting["height"]/scenario_setting["cell_size"])
        self.ped_coordinates = scenario_setting["pedestrian_coordinates"]
        self.obs_coordinates = scenario_setting["obstacle_coordinates"]
        self.tar_coordinates = scenario_setting["target_coordinates"]
        self.blo_coordinates = scenario_setting["block_coordinates"]
        self.density_info = scenario_setting["pedestrian_for_density"]
        self.use_dijkstra = scenario_setting["use_dijkstra"]
        self.use_speed = scenario_setting["use_speed"]
        self.speed = scenario_setting["speed"]
        self.follow = scenario_setting["follow"]
        self.use_utility= scenario_setting["use_utility"]
        self.r_max = scenario_setting["r_max"]
    
    
        
        #Basic attribute of a scenario, which stores the state of each cell in each step
        #The states value are 0, 1, 2, 3, representing "E", "P", "O", "T" respectively
        self.states = np.zeros((self.number_of_cells_y, self.number_of_cells_x))
        self.distances = np.zeros((self.number_of_cells_y, self.number_of_cells_x))
        self.utilities = np.zeros((self.number_of_cells_y, self.number_of_cells_x))
        
        #empty list of pedestrian, target, obstacle position
        self.ped_coordinates_list = []
        self.obs_coordinates_list = []
        self.tar_coordinates_list = []
        
        #calculate the width of each cell
        self.cell_width = int(500 / self.number_of_cells_x)
        #calculate the height of each cell
        self.cell_height = int(500 / self.number_of_cells_y)
        
        
        self.pedestrians = []
        self.target_distance_grids = self.recompute_target_distances()
        
        
    def add_pedestrian(self):
        """_summary_
            get the list of pedestrain positions from the scenario_setting file (json)
        """
        for ped in self.ped_coordinates:
            if  (ped["y"]-1, ped["x"]-1) not in self.ped_coordinates_list:
                # from dict in json file to ped_coordinates_list
                self.ped_coordinates_list.append(  (ped["y"]-1,ped["x"]-1) )
                # update the scenario state according to the given coordinates of pedestrian.
                self.states[ped["y"]-1, ped["x"]-1] = state_space["P"]
                
            
    def add_obstacle(self):
        """_summary_
            get the list of obstacle positions from the scenario_setting file (json)
        """
        for obs in self.obs_coordinates:
            if obs['x'] == None:
                break
            elif (obs["y"]-1, obs["x"]-1) not in self.obs_coordinates_list:
                self.obs_coordinates_list.append(  (obs["y"]-1, obs["x"]-1) )
                self.states[obs["y"]-1, obs["x"]-1] = state_space["O"]
                
            
    def add_target(self):    
        """
            get the list of target positions from the scenario_setting file (json)
        """
        for tar in self.tar_coordinates:
            if  (tar["y"]-1, tar["x"]-1) not in self.tar_coordinates_list:
                self.tar_coordinates_list.append(  (tar["y"]-1, tar["x"]-1) )
                self.states[tar["y"]-1, tar["x"]-1] = 3
                
    
    def add_block_obstacle(self):
        """_summary_
            add a block of obstacles in certain area to accelerate the scenario setting
        """
        for blo in self.blo_coordinates:
            for x_offset in range(blo["lower_right_x"] - blo["upper_left_x"] + 1):
                for y_offset in range(blo["lower_right_y"] - blo["upper_left_y"] + 1):
                    self.obs_coordinates_list.append(  ( blo["upper_left_y"]+y_offset-1, blo["upper_left_x"]+x_offset-1) )
                    self.states[ blo["upper_left_y"]+y_offset-1, blo["upper_left_x"]+x_offset-1] = 2 #state_space["O"]
                    
                    
    def add_ped_with_density(self):
        """_summary_
            add a certain number of pedestrians with density
            The coordinates of the added pedestrians are random
        """
        for density_info in self.density_info:
            density = density_info["density"]
            total_cells = self.number_of_cells_x * self.number_of_cells_y
            total_peds = math.floor(total_cells * (self.cell_size**2) * density)
            
            if total_peds > total_cells:
                raise ValueError('Failed to add Pedestrians. Density to high.')
            else:
                for add_ped in range(total_peds):
                    rand_y = random.randrange(density_info["upper_left_y"], density_info["lower_right_y"]+1 )
                    rand_x = random.randrange(density_info["upper_left_x"], density_info["lower_right_x"]+1 )
                    #if the newly generated ped coordinate does not exist yet, add it.
                    if (rand_y-1, rand_x-1) not in self.ped_coordinates_list: 
                        self.ped_coordinates_list.append(  (rand_y-1, rand_x-1)  )
                        self.states[rand_y-1, rand_x-1] = state_space["P"]
                #start from the states of left cells       
                self.ped_coordinates_list.sort( key=lambda x: x[0])        
                self.ped_coordinates_list.sort(reverse= True, key=lambda x: x[1])
  
    
    def euclidean_distance(self,i,j,k,l):
        return np.sqrt(   (i-k)**2 + (j-l)**2  ) 


    def recompute_target_distances(self):
        self.target_distance_grids = self.update_target_grid()
        return self.target_distance_grids
    
    
    def update_target_grid(self):
        """_summary_
            Computes the shortest distance from every grid point to the nearest target cell.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        self.add_pedestrian()
        self.add_obstacle()
        self.add_target()
        self.add_block_obstacle()
        self.add_ped_with_density()
        curr_states = self.states
        
        
        targets = []              
        for tar_dict in self.tar_coordinates:
            targets.append((tar_dict['x'], tar_dict['y']))
           
        if len(targets) == 0:
            print("There is no targets!!!")
            return np.zeros((self.number_of_cells_x, self.number_of_cells_y))          
            
        targets = np.row_stack(targets)
        x_space = np.arange(0, self.cell_width)
        y_space = np.arange(0, self.cell_height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)

        return distances.reshape((self.cell_width, self.cell_height))
    
    
    def update_step(self):
        """
        Updates the position of all pedestrians.
        This takes obstacles and other pedestrians into account.
        Pedestrians will avoid cells already occupied by obstacles when moving
        
        :returns: the original positions before moving.
        """
               
        for pedestrian in self.pedestrians:
            old_positions = pedestrian.update_step(self)
            
        return old_positions


    def to_image(self, canvas, old_positions):
        """
        Get new positions of Pedestrians after moving, then create Pedestrian instances with color "blue"
        Get original positions of Pedestrians before moving, then make the cells as empty, i.e. change the color to "white"      
        If the new position is ocupied by target, keep the color "green"
        
        :param canvas: the canvas that holds the image.
        :param old_positions: the original positions of Pedestrians before moving.
        
        """       
        
        rect_size_x = 500 / self.number_of_cells_x
        rect_size_y = 500 / self.number_of_cells_y
        
        
        targets = []              
        for tar_dict in self.tar_coordinates:
            targets.append((tar_dict['x'], tar_dict['y']))
            
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            
            x1 = (x - 1) * rect_size_x
            y1 = (y - 1) * rect_size_y
            x2 = x1 + rect_size_x
            y2 = y1 + rect_size_y
            
            if (x, y) in targets:
                canvas.create_rectangle(x1, y1, x2, y2, fill="green", outline="black")
                
            else:
                canvas.create_rectangle(x1, y1, x2, y2, fill="blue", outline="black")
                
            self.ped_coordinates.append({"x":x, "y":y})
        
        
        for (x_pos, y_pos) in old_positions:
            x1 = (x_pos - 1) * rect_size_x
            y1 = (y_pos - 1) * rect_size_y
            x2 = x1 + rect_size_x
            y2 = y1 + rect_size_y
            if (x_pos, y_pos) not in targets:
                canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
                
            self.ped_coordinates.remove({"x":x_pos, "y":y_pos})
        
        
def set_scenario(scenario):
    """_summary_
        assign corresponding cells of a certain scenario with "E", "P", "O", "T"
        
    Parameters
    ----------
    scenario : scenario object

    Returns
    -------
        _description_  the initial states with "E", "P", "O", "T"
    """
    scenario.add_pedestrian()
    scenario.add_obstacle()
    scenario.add_target()
    scenario.add_block_obstacle()
    scenario.add_ped_with_density()
    return scenario.states          



