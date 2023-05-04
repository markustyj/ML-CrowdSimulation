#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:55:39 2023

@author: jingzhang
"""

import sys
import tkinter as tk
from tkinter import Button, Canvas, Menu
import math
from pedestrian import Pedestrian

state_space = {
    "E":0,
    "P":1,
    "O":2,
    "T":3,
    }

class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """
    def __init__ (self, scenario):
        self.scenario = scenario
        self.win = tk.Tk()
        self.canvas = tk.Canvas(self.win, width = 500, height = 500)
        self.rect_size = 500 / self.scenario.number_of_cells_x


    def create_scenario(self, ):
        print('create not implemented yet')
        
        
    def add_ped_by_click(self, event):
        """
        Adds a point to the canvas at the location of the mouse click.

        Args:
            event (tkinter.Event): Add _description_
        """
        x = math.floor(event.x / self.rect_size)
        y = math.floor(event.y / self.rect_size)
        ped = {'x' : x + 1, 'y' : y + 1}
        
        if  ped not in self.scenario.ped_coordinates + self.scenario.obs_coordinates + self.scenario.tar_coordinates + self.scenario.ped_coordinates_list:
            self.scenario.ped_coordinates_list.append((x + 1, y + 1))           
            x1 = (ped['x'] - 1) * self.rect_size
            y1 = (ped['y'] - 1) * self.rect_size
            x2 = x1 + self.rect_size
            y2 = y1 + self.rect_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue", outline="black")
            print("added" + str(ped))
            self.scenario.ped_coordinates.append(ped)
            
            # update the scenario state according to the given coordinates of pedestrian.
            self.scenario.states[ped["y"]-1, ped["x"]-1] = state_space["P"]
        else : print("existing cell")


    def restart_scenario(self, ):
        print('restart not implemented yet')


    def step_scenario(self, scenario, canvas):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        old_positions = self.scenario.update_step()
        self.scenario.to_image(canvas, old_positions)
        
        

    def exit_gui(self, ):
        """
        Close the GUI.
        """
        sys.exit()



    def start_gui(self):    
        """
        Creates and shows a simple user interface with a menu and button "step simulation".       
        Also creates a rudimentary, fixed Scenario instance with two Pedestrian instances, three obstacles, and one target.
        
        """

        # setting the size of the window
        self.win.geometry('500x600') 
        
        # setting the title of the window
        self.win.title('Cellular Automata GUI')

        menu = Menu(self.win)
        self.win.config(menu=menu)
        file_menu = Menu(menu)
        # Creates a new hierarchical menu by associating a given menu to a parent menu
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=self.create_scenario)
        # Adds a menu item to the menu.
        file_menu.add_command(label='Restart', command=self.restart_scenario) 
        file_menu.add_command(label='Close', command=self.exit_gui)
        
        
        # bind the canvas to the click event
        self.canvas.bind('<Button-1>', self.add_ped_by_click)
        self.canvas.pack()
        
        
        # Create empty cells with color "white"
        for i in range(self.scenario.number_of_cells_x):
            for j in range(self.scenario.number_of_cells_y):
                x1 = j * self.rect_size
                y1 = i * self.rect_size
                x2 = x1 + self.rect_size
                y2 = y1 + self.rect_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")


        # Create Pedestrian instances with color "blue"
        for cell in self.scenario.ped_coordinates:
            x1 = (cell['x'] - 1) * self.rect_size
            y1 = (cell['y'] - 1) * self.rect_size
            x2 = x1 + self.rect_size
            y2 = y1 + self.rect_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue", outline="black")
        
        
        # Create obstacles instances with color "red"
        for cell in self.scenario.obs_coordinates:
            x1 = (cell['x'] - 1) * self.rect_size
            y1 = (cell['y'] - 1) * self.rect_size
            x2 = x1 + self.rect_size
            y2 = y1 + self.rect_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="red", outline="black")
        
            
        # Create target instances with color "green"
        for cell in self.scenario.tar_coordinates:
            x1 = (cell['x'] - 1) * self.rect_size
            y1 = (cell['y'] - 1) * self.rect_size
            x2 = x1 + self.rect_size
            y2 = y1 + self.rect_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="green", outline="black")

        
        for ped_dict in self.scenario.ped_coordinates:
            self.scenario.pedestrians.append(Pedestrian((ped_dict['x'], ped_dict['y']), self.scenario.speed))
     
       
        btn = Button(self.win, text='Step simulation', command=lambda: self.step_scenario(self.scenario, self.canvas))
        btn.place(x=180, y=525)
    
        self.win.mainloop()
        
        
      