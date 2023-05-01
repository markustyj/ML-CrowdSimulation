#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Some codes of vertices and graph generation are found online.
"""
Created on Apr 18 16:23:39 2023

@author: Yongjian Tang
"""

import sys
import numpy as np


state_space = {
    "E":0,
    "P":1,
    "O":2,
    "T":3,
    }

def generate_graph(states):
    """_summary_       generate graph based on the cell grid layout
     
    Parameters
    ----------
    states : _type_    2d numpy array
        _description_  store the states of all cells in a grid

    Returns
    -------
    g : _type_         graph 
        _description_  newly generated graph object
    """
    g = Graph()
    y_shape = states.shape[0]
    x_shape = states.shape[1]

    #add vertecies, as long as the cell is not obstacle. In other words, targets and pedestrian will be added. 
    for x in range(x_shape):
        for y in range(y_shape):
            if states[y,x] == state_space["O"]:
                continue
            g.add_vertex((y, x))

    #add edges    
    offset = [[1,0],[0,1],[-1,0],[0,-1]]
    all_nodes = g.vert_dict.keys()
    # integrate over all added vertices (pedestrians, target, empty cell)
    for node in all_nodes:
        neighbors = np.add(node,offset)
        for neighbor in neighbors:
            if tuple(neighbor) in all_nodes:
                # add edge between a cell and its neighbour with cost 0.5
                g.add_edge(node,tuple(neighbor), 1) 
    return g



class Graph:
    """_summary_
    Class of graph to create object graph
    """

    def __init__(self):
        """ Parameters
        ----------
        vert_dict:     dictionary to store all vertices 
        num_vertices:  amount of Vertices in the graph
        """
        self.vert_dict = {}
        self.num_vertices = 0

    def add_vertex(self, node):
        """Generate a Vertex object with id node and adds it to the dictionary of the Graph's Vertices
        """
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, node):
        """ get the vertex of a graph given its vertex(node)_id 
        """
        return self.vert_dict[node]

    def get_vertices(self):
        """Returns the vertex of a graph given the vertex(node)_id
        """
        return self.vert_dict.keys()
    
    def add_edge(self, frm, to, cost = 0):
        """_summary_
        add an edge between 2 vertices of a graph, also add the weight to the edges
        Parameters
        ----------
        frm : _type_        vertex
            _description_   the start vertex of an edge
        to : _type_         vertex
            _description_   the end vertex of an edge
        cost : int
            _description_   weight on the edge, by default 1
        """
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def set_previous(self, current):
        """_summary_       set the given current vertex to previous
                           ease the work to trace back from target to the starting node
        """
        self.previous = current

    def get_previous(self):
        """_summary_       get the previous vertex given the current vertex
                           ease the work to trace back from target to the starting node
        """
        return self.previous

    def __iter__(self):
        """Iterates through the Vertices of Graph an returns one random Vertex
        """
        return iter(self.vert_dict.values())
    

class Vertex:
    """_summary_
    The Vertex class generates the nodes of the search  graph as objects 
    """
    def __init__(self, nodes):
        """ Parameters
        ----------
        id:        coordinates of a node/cell in the form of tuple
        adjacent:  Dictionary to store the neighbors of a cell
        distance:  distance to the initial point, is set to infinite at the beginning
        visited:   All nodes are unvisited at the very beginning
        previous:  Stores Predecessor
        """
        self.id = nodes
        self.adjacent = {}
        self.distance = sys.maxsize     
        self.visited = False  
        self.previous = None

    def get_connections(self):
        """
        :return: the vertice_id of a vertex's neighbours 
        :rtype: tuple, (y, x)
        """
        return self.adjacent.keys()  

    def get_id(self):
        """
        :return: vertice_id
        :rtype: tuple, (y, x)
        """
        return self.id

    def get_weight(self, neighbor):
        """_summary_   Get weight of neighbour object
        :param neighbor: Neighbour of Vertex
        :type neighbor: Vertex
        :return: weight of edge to neighbour
        :rtype: float
        """
        return self.adjacent[neighbor]

    def set_distance(self, distance_value):
        """
        :param dist: Distance from initial vertice to reach the Vertex
        :type dist: int 
        """
        self.distance = distance_value

    def get_distance(self):
        """
        :return: Distance from initial vertice to reach the Vertex
        :rtype: int
        """
        return self.distance
    
    def __str__(self):
        """Returns ID and all IDs of the neighbour nodes
        """
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])
    
    def set_previous(self, prev):
        """ set the given current vertex to previous, ease the work to trace back from target to the starting node
        :param prev: Previous node
        :type prev: Vertex
        """
        self.previous = prev

    def set_visited(self):
        """set visited vertice to visited state
        """
        self.visited = True

    def add_neighbor(self, neighbor, weight=0):
        """_summary_   Adds neighbour to the Vertex

        Parameters
        ----------
        neighbor : _type_    tuple
            _description_    vertices_id
        weight : int
            _description_    weight on an edge, by default 0
        """
        self.adjacent[neighbor] = weight


        

