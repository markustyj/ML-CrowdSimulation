#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 18 16:23:39 2023

@author: Yongjian Tang
"""
import heapq
import random
import math
import numpy as np

from graph_on_grid import *


state_space = {
    "E": 0,
    "P":1,
    "O":2,
    "T":3,
    }
    

class Automaton:
    """_summary_
        class to calculate distances, utilities and move the pedestrian in simulation
    """
    def __init__ (self, scenario, measure_speed = False):
        self.scenario = scenario
        self.measure_speed = measure_speed  #boolean
        self.remaining_distance = np.zeros(len(self.scenario.ped_coordinates_list))
        self.remaining_step = np.zeros(len(self.scenario.ped_coordinates_list))
        self.speed = np.zeros(len(self.scenario.ped_coordinates_list))
        for i in range(len(self.scenario.ped_coordinates_list)):
            self.speed[i] = self.scenario.speed - 0.3 + random.random() * 0.6
        self.utility = np.zeros((self.scenario.number_of_cells_y, self.scenario.number_of_cells_x))
        self.accumulated_dis = np.zeros(len(self.scenario.ped_coordinates_list))
        self.remaining_dis = np.zeros(len(self.scenario.ped_coordinates_list))
    
    def euclidean_distance1(self, i, j, k, l):
        """_summary_
            calculate euclidean distance between 2 cells
        i,j,k,l: _type_        int
                _description_  coordinates of the 2 cells    
        """
        return np.sqrt(   (i-k)**2 + (j-l)**2  ) 
    
    def cost_func(self, r, rmax):
        """_summary_
            Calculate the cost function for a given distance r and a maximum distance rmax.
        """
        if r < rmax:
            return np.exp(1 / (r**2-rmax**2))
        else:
            return 0
    
    def pedestrians_utilities(self, i, j):
       #calculates the utilities between pedestrians and surrounding neighbor cells according to cost function
        tar_x, tar_y = self.scenario.tar_coordinates_list[0]
        for x in range(self.scenario.number_of_cells_x):
            for y in range(self.scenario.number_of_cells_y):
                if (x, y) not in self.scenario.obs_coordinates_list:
                    #calculates the distance between the surrounding neighbor cells and the target
                    tar_d = self.euclidean_distance1(x, y, tar_x, tar_y)
                    #calculates the distance between the surrounding neighbor cells and the pedestrian
                    ped_d = self.euclidean_distance1(x, y, i, j)
                    r = math.sqrt(tar_d ** 2 + ped_d ** 2)
                    self.utility[x, y] += self.cost_func(r, self.scenario.r_max)
              
    def get_neighbours(self, i, j):
        #Compute all neighbors in a 9 cell neighborhood of the current position.
        return [
            (int(x + i),int(y + j))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + i < self.scenario.number_of_cells_x and 0 <= y + 1 < self.scenario.number_of_cells_y and np.abs(x) + np.abs(y) > 0
        ]
    
    def reset_pedestrians_positions(self, ped_next_step):
        """"""
        #set the current P value to E on the grid, i.e. empty
        for ped in self.scenario.ped_coordinates_list:
            if (ped[0],ped[1]) not in self.scenario.tar_coordinates_list:
                #only clear the cell when it is not in the target cell
                self.scenario.states[ped[0], ped[1]] = state_space["E"]   
                #besides the value on grid, also clear the old ped_coordinates_list
                self.scenario.ped_coordinates_list = []
                
        #set the value of the next step    
        for ped in ped_next_step:
            if (ped[0],ped[1]) not in self.scenario.tar_coordinates_list: 
            #if the new ped coordinate is not in target, set it to P, otherwise keep the target state_space["T"] 
                self.scenario.states[ped[0], ped[1]] = state_space["P"]   
                self.scenario.ped_coordinates_list.append( (ped[0], ped[1]) )
            else:
            #if the new ped coordinate is in target, do not set it to P.     
                self.scenario.ped_coordinates_list.append( (ped[0], ped[1]) )

        #if we hope a cell can strictly follow each other
        if self.scenario.follow :
            self.scenario.ped_coordinates_list.sort( key=lambda x: x[0])        
            self.scenario.ped_coordinates_list.sort(reverse= True, key=lambda x: x[1])       
    
    
    def move_pedestrians(self,diagnal):
        neighbour_offsets_diagnal = [ [-1,0], [0,1], [1,0], [0,-1], [1, 1], [-1, -1], [-1, 1], [1, -1], [0,0] ]
        #     here diagonal is probabily needed ?
        neighbour_offsets = [ [-1,0], [0,1], [1,0], [0,-1], [0,0] ]
        cell_size=self.scenario.cell_size
        #based on the input of method, decide whether we should abandom diagnal neighbour or not.
        if diagnal :
            offsets = neighbour_offsets_diagnal
        else:
            offsets = neighbour_offsets

        #calculate the distance of each cell to the target. There are 2 cases: euclidean and dijkstra
        if self.scenario.use_dijkstra:
            self.dijkstra_distance()
        else:
            self.euclidean_distance()    
        
        # list of ped coordinates for next step 
        ped_next_step = []      
        #consider each pedestrian respectively
        for i, ped in enumerate(self.scenario.ped_coordinates_list):
            #set a large min_distance at very beginning
            min_distance = sys.maxsize
            if not self.scenario.use_utility:            
                if self.scenario.use_speed :
                    # in case of accumulated ditance larger than 1, double the offset
                    self.remaining_step[i] = math.floor(self.remaining_distance[i] / self.scenario.cell_size)
                    offsets = []
                    for j in range(1,int(self.remaining_step[i])+1):
                        offsets.append([0,j])
                    #offsets.reverse()
                    
                #first, set the current position as the position in the next step, in case that no pedestrian has taken this chosen cell in the next step
                neighbour_selected = (ped[0],ped[1]) 
                #consider each neighbour offset respectively
                for neighbour_offset in offsets:
                    #get the true value of the neighbour
                    neighbour = np.add ([ ped[0],ped[1] ], neighbour_offset)

                    #consider the case of obstacle, set the distance to infinite large
                    if (neighbour[0], neighbour[1]) in self.scenario.obs_coordinates_list:
                        distance = sys.maxsize
                        continue    

                    if not self.scenario.use_speed:
                        #self.reset_pedestrians_positions(ped_next_step)#make sure that cell is on grid and pick the minimun distance 
                        if  self.on_grid(neighbour[0], neighbour[1]) :
                            #calculate euclidean distance
                            distance = self.scenario.distances[neighbour[0],neighbour[1]]
                            if  (distance<min_distance) :#& ((neighbour[0],neighbour[1]) not in self.scenario.ped_coordinates_list)  ).all(): 
                                if ((neighbour[0],neighbour[1]) not in ped_next_step) :#self.scenario.ped_coordinates_list):

                                    neighbour_selected = (neighbour[0],neighbour[1])
                                    #update min_distance from the neighbour of a cell to the target
                                    min_distance = distance            

                    # use_speed == True                       
                    else: 
                        if self.on_grid(neighbour[0],neighbour[1]):
                            distance = self.scenario.distances[neighbour[0],neighbour[1]]
                            if  (distance<min_distance) :
                                min_distance = distance
                                if ((neighbour[0],neighbour[1]) not in self.scenario.ped_coordinates_list) and ((neighbour[0],neighbour[1]) not in ped_next_step):
                                    neighbour_selected = (neighbour[0],neighbour[1])
                                else:
                                    break    
                        else:
                            break  
            #check if the utility needs to be used 
            elif self.scenario.use_utility:
                # Reset the utility values to zero
                self.utility.fill(0)
                neighbour_selected = None
                speed=self.scenario.speed
                next_pos = ped
                if not self.scenario.use_speed:
                    # Calculate the utility values for each neighbouring cell
                    self.pedestrians_utilities(ped[0], ped[1])
                    max_utility = 0
                    #Iterate over each neighbouring cell to find the one with the highest utility value
                    for neighbour_offset in offsets:
                        #get the true coordinate of the neighbour 
                        neighbour = np.add ([ ped[0],ped[1] ], neighbour_offset)
                        # Check if the neighbour is an obstacle or outside the grid
                        if (neighbour[0], neighbour[1]) in self.scenario.obs_coordinates_list or not self.on_grid(neighbour[0], neighbour[1]):
                        
                            continue
                        # Check if the neighbour is the target cell
                        elif (neighbour[0], neighbour[1]) in self.scenario.tar_coordinates_list:
                            neighbour_selected = neighbour
                            max_utility=0
                            break
                        #if the neighbour is already a planned next step to avoid the situation that a cell including more than one pedestrian
                        elif any([np.array_equal(neighbour, next_step) for next_step in ped_next_step]) : 
                            continue

                        else:
                            # Select the neighbour with the highest utility
                            if self.utility[neighbour[0],neighbour[1]] > max_utility:
                                neighbour_selected = neighbour
                                max_utility = self.utility[neighbour[0],neighbour[1]] 
                #use_speed = true
                else:
                    # Calculate the remaining distance the pedestrian has to move
                    remaining = self.remaining_dis[i]
                    available_dis =  speed + remaining
                    neighbour_sel = None
                    n=0
                    t=0
                    visited = set()
                     # while the pedestrian has available distance to move and has not reached the target
                    while available_dis > 0:
                        # set the initial min_distance to a large value
                        min_distance = 1000
                        # iterate over each neighbouring cell to find the cell closest to the target
                        for neighbour_offset in offsets:
                            x,y=self.scenario.tar_coordinates_list[0]
                            neighbour = np.add([next_pos[0], next_pos[1]], neighbour_offset)
                            tar_distance = self.scenario.distances[next_pos[0],next_pos[1]]
                            if (neighbour[0], neighbour[1]) in self.scenario.obs_coordinates_list or not self.on_grid(neighbour[0], neighbour[1]):
                                continue
                            # if the neighbour is one of the target's neighbours, stop the loop
                            elif (next_pos[0], next_pos[1]) in self.get_neighbours(x,y):
                                break
                            else:
                                # Calculate the distance to the target from the neighbour
                                distance_to_target = self.scenario.distances[neighbour[0], neighbour[1]]
                                if any([np.array_equal(neighbour, next_step) for next_step in ped_next_step]):
                                    continue
                                else:
                                    # set the minimum distance to this distance and select this neighbour as the next position
                                    if distance_to_target < min_distance:
                                        min_distance = distance_to_target
                                        neighbour_sel = neighbour
                                        dis = self.euclidean_distance1(neighbour[0],neighbour[1],next_pos[0],next_pos[1]) 
                        # ff no suitable neighbour was found or the neighbour was already visited, stop the loop
                        if neighbour_sel is None or tuple(neighbour_sel) in visited:
                            break
                        else:
                            # add the current position to the set of visited positions
                            visited.add(tuple(next_pos))
                            #if the pedestrian can move at its maximum speed, and there is enough distance left to reach the neighbour
                            if speed>=math.sqrt(2*(cell_size**2)):   
                                if available_dis >= dis:
                                 # set the neighbour as the next position and subtract the distance travelled from the available distance
                                    next_pos = neighbour_sel
                                    available_dis -= dis
                                    self.remaining_dis[i] = available_dis
                                else:
                                    self.remaining_dis[i] = available_dis
                                    break
                            elif speed<math.sqrt(2*(cell_size**2)):        
                                if available_dis >= dis:
                                    next_pos = neighbour_sel
                                    available_dis -= dis
                                    self.remaining_dis[i] = available_dis
                                    t=t+1
                                else:
                                    if available_dis+speed >= dis and n==0 and t==0:
                                        # add the speed to the available distance and increment n
                                        available_dis+=speed
                                        n=n+1
                                    else:                                                            
                                        break
                            else:
                                #updates the remaining distance that the pedestrian can accumulate to move in the next iteration 
                                self.remaining_dis[i] = available_dis
                                break
                    neighbour_selected = next_pos


            #speed measure for newly selected neighbour, if it is in the measuring point
            if self.measure_speed:
                if (neighbour_selected[1] >= self.measuring_points_boundaries[0] and neighbour_selected[1] <= self.measuring_points_boundaries[1]):            
                    #print(ped, neighbour_selected, "adfa")
                    if ped[0] == neighbour_selected[0] and ped [1] == neighbour_selected[1] :
                        #print("adsfasd")       
                        #as long as measure_speed == True && a ped is in measuring point && he doesn't move, we should add the time step for him                    
                        self.pedestrians_time_in_measuring_points[i] = self.pedestrians_time_in_measuring_points[i] + 1 #accumulate time steps for each pedestrian        
                    else: 
                        #print("spppeeeeed")
                        self.pedestrians_speed_sum[i] += self.scenario.speed - 0.1 + random.random() * 0.2 #accumulate speed for each pedestrian
                        self.pedestrians_time_in_measuring_points[i] = self.pedestrians_time_in_measuring_points[i] + 1 #accumulate time steps for each pedestrian     
                                

            # a list of new ped coordinates for each old ped coordinates respectively        
            ped_next_step.append(neighbour_selected)
            #print(ped_next_step)
        

        #update the grid layout
        self.reset_pedestrians_positions(ped_next_step)
    

    def set_measuring_point(self, x_left, x_right):
        """_summary_
            set the measuring points for the speed
        Parameters
        ----------
        x_left : _type_      int
            _description_    left bounndry of the neasuring point
        x_right : _type_     int
            _description_    right boundry of the neasuring point
        moving_speed : _type_float
            _description_    the speed of pedestrians
        """
        self.measuring_points_boundaries = [x_left, x_right]
        self.pedestrians_speed_sum = np.zeros(len(self.scenario.ped_coordinates_list))
        self.pedestrians_time_in_measuring_points = np.zeros(len(self.scenario.ped_coordinates_list))

    

    def euclidean_distance(self):
        """_summary_
                calculate the euclidean distance of each cell on a grid to the target
           _return_
                2 dimensional numpy array of euclidean distance (same shape as cell states)
        """
        for target in self.scenario.tar_coordinates_list:
            k, l = target[0], target[1]
            for i in range(self.scenario.distances.shape[0]):
                for j in range(self.scenario.distances.shape[1]):
                    #compute euclidean distance of each cell to target cell
                    if self.scenario.states[i, j] != state_space["O"]:
                        self.scenario.distances[i, j] += np.sqrt((i - k) ** 2 + (j - l) ** 2) 
                        self.scenario.utilities[i, j] += np.sqrt((i - k) ** 2 + (j - l) ** 2) 

    def dijkstra_distance(self):
        """_summary_
                calculate the dijkstra distance of each cell on a grid to the target
           _return_
                2 dimensional numpy array of dijkstra distance (same shape as cell states)
        """
        states = self.scenario.states
        target = tuple(self.scenario.tar_coordinates_list[0]) 
        for x in range(states.shape[1]):
            for y in range(states.shape[0]):
                #do not consider obstacle and target, 
                if (states[y,x] == 2) or (states[y,x] == 3) or (self.scenario.distances[y,x] != 0):
                    continue 
                # for each non-obstacle cell, regenerate the graph once. The distance in each cell will be set to sys.max
                graph = generate_graph(states)
                # choose this cell as the starting vertice, then extend to all cells in grid to find their dijkstra distance to the starting vertice.
                self.dijkstra(graph, graph.get_vertex((y,x)), graph.get_vertex(target))
                # starting from the target cell or vertex, find the shortest path to the starting vertex (y,x) in a recursive way
                path = [target]
                self.shortest(graph.get_vertex(target), path)

                path = np.array(path)
                path_length = (path.shape[0] - 1) 
                # inverse the order of all nodes in the path, namely from the starting vertex (x,y) to the target cell
                for node in path[::-1]:
                    node = np.array(node)
                    # if any cell in the grid has not been assigned any distance value yet, assign the computed dijkstra distance to it
                    if self.scenario.distances[node[0], node[1]] == 0:
                        self.scenario.distances[node[0],node[1]] += path_length
                        self.scenario.utilities[node[0],node[1]] += path_length
                    path_length -= 1  
                    
    
    def on_grid(self, i, j):
        """_summary_
                Given 2 coordinates i, j 
                determine whether they are still on the grid
           _return_
                True, if the given coordinates is on the grid 
        """
        return -1 < i < self.scenario.number_of_cells_y and -1 < j < self.scenario.number_of_cells_x    

    

    def dijkstra(self, aGraph, start, target):
        """[summary]
        :param aGraph: Search Graph
        :type aGraph: Graph
        :param start: Start node, which is iterating over all grid cells in dijkstra_distance
        :type start: Vertex
        :param target: target
        :type target: Vertex
        """
        # Set the distance for the start node to the value in ( ), i.e. 0
        start.set_distance(0)

        # get a tupel of the distance between current vertex and initial vertex, and vertex id
        unvisited_queue = [(v.get_distance(),v.get_id(),v) for v in aGraph]
        # Put tuple pair into the priority queue
        heapq.heapify(unvisited_queue)

        # As long as there are still vertices in the graph and they have not been all popped out.
        while len(unvisited_queue):
            # Pops a vertex with the smallest distance 
            popped_tupel = heapq.heappop(unvisited_queue)
            #assign this vertex as the current vertex
            current = popped_tupel[2] 
            #make this vertex as visitedS
            current.set_visited()

            #use for loop to walk around all neighbours of this vertex:
            for next in current.adjacent:
                # if visited, skip
                if next.visited:
                    continue
                # calculate the distance of each neighbour of this vertex
                new_distance = current.get_distance() + current.get_weight(next)
                # compare it with the current distance of this vertex 
                if new_distance < next.get_distance():
                    #choose the smaller one and set larger one to previous
                    next.set_distance(new_distance)
                    next.set_previous(current)

            # Rebuild heap: Process the unvisited_queue for the next iteration in the larger while loop
            # Pop every item to make the unvisited_queue empty
            while len(unvisited_queue):
                heapq.heappop(unvisited_queue)
            # Put all vertices not visited into the queue
            unvisited_queue = [(v.get_distance(),v.get_id(),v) for v in aGraph if not v.visited]
            heapq.heapify(unvisited_queue)
            

    def shortest(self, v, path):
            '''find the shortest path to initial node, given a vertex'''
            if v.previous:
                path.append(v.previous.get_id())
                self.shortest(v.previous, path)
            return

        
    def simulation_multiple_steps(self, timesteps = None, diagnal = False):
        """_summary_
            use the written automaton and simulate the crowd dynamics until all pedestrians reach the target. 
        Parameters
        ----------
        timesteps : int, optional 
             _description_,    the maximum simulation time steps. Stop here, even when all pedestrians have not reached the target
        diagnal : bool, optional
             _description_,    allow pedestians to move diagonally, when True

        Returns
        -------
        _type_
            _description_      a series of states starting from the initial state to the final stopping state
        """
        data_scenario_states = []

        if timesteps == None:
            timesteps = self.scenario.number_of_cells_y * self.scenario.number_of_cells_x
        for k in range(timesteps):
            data_scenario_states.append(np.copy(self.scenario.states))
            if self.scenario.use_utility:
                if k == 0:
                #The first step has to be executed anyway, but the elif condition can't be tested. States[-2] is not existing yet.
                    self.move_pedestrians(diagnal)
                elif ((self.scenario.states == data_scenario_states[-2]) & (k>=2)).all():
                #The last step is k, not k+1 steps, because k+1 step is used to check the above condition, i.e.2 states are equal.
                    print("Simulation stopped after", k-1 ,"simulation steps, since the pedestrians have reached the target.")
                    break     
                else:
                    self.move_pedestrians(diagnal)

            else:
                #self.move_pedestrians(diagnal)
                # check whether all pedestrians reach targets
                if k == 0:
                #The first step has to be executed anyway, but the elif condition can't be tested. States[-2] is not existing yet.
                    self.move_pedestrians(diagnal)
                elif ((self.scenario.states == data_scenario_states[-2]) & (k>=2)).all():
                #The last step is k, not k+1 steps, because k+1 step is used to check the above condition, i.e.2 states are equal.
                    print("Simulation stopped after", k-1 ,"simulation steps, since the pedestrians have reached the target.")
                    break     

                elif self.scenario.use_speed==True and (self.scenario.states[1:19,0:190] == 0).all(): 
                    print("Simulation stopped after", k-1 ,"simulation steps, since the pedestrians have reached the target.")
                    break 

                else:
                    self.move_pedestrians(diagnal)

                if self.scenario.use_speed==True:
                    for i in range(len(self.scenario.ped_coordinates_list)):
                        self.remaining_distance[i] = self.remaining_distance[i] - self.remaining_step[i] * self.scenario.cell_size + self.speed[i]
                    

        print("Having reached the preset maximum simulation steps.")

        #handling the speed measuring 
        if self.measure_speed:
            #calculate the average speed of a single pedestrian during his stay in measuring area
            self.pedestrians_average_speed = self.pedestrians_speed_sum / self.pedestrians_time_in_measuring_points
            #calculate the average speed of all pedestrians that have gone through the measuring area
            self.average_speed = np.sum(self.pedestrians_average_speed) / len(self.pedestrians_average_speed)
            
        return data_scenario_states        

   