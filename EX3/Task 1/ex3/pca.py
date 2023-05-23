#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 18 16:23:39 2023
@author: Yongjian Tang
"""

import numpy as np
import pandas as pd
import scipy
import cv2
import matplotlib.pyplot as plt


def read_file(path):
    """_summary_        Read the given data in txt format

    Parameters
    ----------
    path : _type_       string
        _description_   The path to the file location

    Returns
    -------
    _type_              2-dimensional numpy array
        _description_   The original data is transformed to a numpy array
    """
    data = pd.read_csv(path, delimiter=" ", header = None)
    return data.to_numpy()

def load_image():
    """_summary_        Download, read and resize the originanl image

    Returns
    -------
    _type_              2-dimensional numpy array
        _description_   Resized image in numpy array format
    """
    img = scipy.misc.face(gray=True)
    img = cv2.resize(img, dsize=(249, 185))  
    return img

class PCA:
    def __init__(self, x):
        """
        Parameters
        ----------
        x : _type_        numpy array
            _description_ 2-dimensional numpy array transformed from a set of data points or images
        """
        self.x = x

    def mean(self,x):
        """_summary_      calculate the mean value of each dimension 

        Parameters
        ----------
        x : _type_        numpy array
            _description_ 2-dimensional numpy array of all original data points or images

        Returns
        -------
        _type_            numpy array
            _description_ A 1-dimensional numpy array containing the mean value of each dimension
        """
        return np.mean(x, axis=0)
    
    def center_data(self, x):
        """_summary_      center all datapoints, according to the calculated center point

        Parameters
        ----------
        x : _type_        numpy array
            _description_ 2-dimensional numpy array of all original data points

        Returns
        -------
        _type_            numpy array
            _description_ 2-dimensional numpy array of the newly-centered dataset
        """
        return x - self.mean(x) 

    def svd(self, x_centered):
        """_summary_      perform singular value decomposition to the centered dataset

        Parameters
        ----------
        x_centered : _type_ numpy array
            _description_   the 2-dimensional newly-centered dataset

        Returns
        -------
        u:  _type_          numpy array
            _description_   size （ x.shape[0], x.shape[0] ）
        s:  _type_          numpy array
            _description_   size （ k, k , with k = min(x.shape[0], x.shape[1]), diagonal matrix of all singular values
        vh:  _type_         numpy array
            _description_   size （ x.shape[1], x.shape[1] ）, transposed matrix of all singular vectors 
        """
        u, s, vh = np.linalg.svd(x_centered, full_matrices=True)
        s = np.diag(s)   # is s squared or not.
        return u,s,vh

    def pca_cal(self, x):
        """_summary_        perform both methods center_data() and svd()

        Parameters
        ----------
        x : _type_          numpy array 
            _description_   2-dimensional numpy array of all original data points

        Returns
        -------
        u:  _type_          numpy array
            _description_   size （ x.shape[0], x.shape[0] ）
        s:  _type_          numpy array
            _description_   size （ k, k , with k = min(x.shape[0], x.shape[1]), diagonal matrix of all singular values
        vh:  _type_         numpy array
            _description_   size （ x.shape[1], x.shape[1] ）, transposed matrix of all singular vectors 
        """
        x_centered = self.center_data(x)
        u,s,vh = self.svd(x_centered)
        return u,s,vh
    
    def pca_reconstruction(self,u,s,vh,r):
        """ _summary_      Reconstruct the dataset based on the given number of principal components and u, s, vh

        Parameters
        ----------
        u:  _type_          numpy array
            _description_   size （ x.shape[0], x.shape[0] ）
        s:  _type_          numpy array
            _description_   size （ k, k , with k = min(x.shape[0], x.shape[1]), diagonal matrix of all singular values
        vh:  _type_         numpy array
            _description_   size （ x.shape[1], x.shape[1] ）, transposed matrix of all singular vectors
        r : _type_          int
            _description_   the number of principal components that we want to consider in our pca algorithm

        Returns
        -------        
        _type_              numpy array
            _description_   The constructed dataset with reduced dimensions

        Raises
        ------
        ValueError
            _description_   only if the given number of principal components is too large, i.e. larger than the dimension of dataset
        """
        if r > s.shape[0]:
            raise ValueError ("The input number of principal components is too large.")
        else:
            ur = u[:,:r]
            sr = s[:r,:r]
            vhr = vh[:r,:]
            xr_centered = ur @ sr @ vhr
            x_reconstructed = xr_centered + self.mean(self.x)
        return x_reconstructed

    def energy_calculation(self, s, L, sentence = True):
        """_summary_        calculate the energy of a specific case after performing pca

        Parameters
        ----------
        s:  _type_          numpy array
            _description_   size （ k, k , with k = min(x.shape[0], x.shape[1]), diagonal matrix of all singular values
        L : _type_          int
            _description_   the number of principal components that we want to consider in our pca algorithm
        sentence : bool, optional 
            _description_,  whether use the formulated sentence in output or just return the energy in unit %

        Returns
        -------
        _type_              float
            _description_   The calculated energy of a specific pca case in unit %

        Raises
        ------
        ValueError
            _description_
        """
        if L > s.shape[0]:
            raise ValueError ("The input number of principal components is too large.")
        energy = np.trace(s[:L,:L]**2) / np.trace(s**2)

        if sentence:
            return "The energy resulting from the given number of principal components is " + str(energy*100) + " %"
        else:
            return energy*100
    

def vis(data):
    """_summary_        visualize image or the layout of data points, based on their 2-dimensional transformed numpy array

    Parameters
    ----------
    data : _type_       numpy array
        _description_   2-dimensional numpy array of data points or images
    """
    fig, ax = plt.subplots(1,1, figsize=(8,4) )
    ax.scatter(data[:,0], data[:,1])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-1.5,1.5])
    #fig.savefig("results/fig1.pdf")

def vis_compare(data1,data2):
    """_summary_         visualize and compare the layout of original data points and centered data points

    Parameters
    ----------
    data1 : _type_       numpy array 
        _description_    2-dimensional numpy array of original data points
    data2 : _type_       numpy array 
        _description_    2-dimensional numpy array of centered data points
    """
    fig, ax = plt.subplots(1,1, figsize=(8,4) )
    ax.scatter(data1[:,0], data1[:,1], label = "original" )
    ax.scatter(data2[:,0], data2[:,1], label = "centered")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-1.5,1.5])
    ax.legend()

def vis_pc(data, vh):
    """_summary_         show the array of singular vectors while shoeing the centered datapoints at the same time

    Parameters
    ----------
    data : _type_        numpy array 
        _description_    2-dimensional numpy array of centered data points
    vh:  _type_          numpy array
        _description_    size （ x.shape[1], x.shape[1] ）, transposed matrix of all singular vectors
    """
    fig, ax = plt.subplots(1,1, figsize=(8,4) )
    ax.scatter(data[:,0], data[:,1], color='tab:orange', label = "centered data")
    ax.arrow(0, 0, vh[0, 0], vh[0, 1], width=0.02, color='red', zorder=2)
    ax.text(vh[0, 0]/2, vh[0, 1]/2, 'PC1')
    ax.arrow(0, 0, vh[1, 0], vh[1, 1], width=0.02, color='red', zorder=2)
    ax.text(vh[1, 0]/2, vh[1, 1]/2, 'PC2') 
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-1.5,1.5])
    ax.legend()
    #fig.savefig("results/fig1.pdf")


def vis_image(image):
    """_summary_         visualize the image after running load_image()

    Parameters
    ----------
    image : _type_       numpy array
        _description_    2-dimensional numpy array of an image
    """
    plt.imshow(image, cmap='gray')     

def vis_images(img1, img2, img3, img4):
    """_summary_         visualize and compare the images reconstructed by different number of principal components

    Parameters
    ----------
    img1 : _type_        numpy array 
        _description_    the images reconstructed by all principal components
    img2 : _type_        numpy array 
        _description_    the images reconstructed by 120 principal components
    img3 : _type_        numpy array 
        _description_    the images reconstructed by 50 principal components
    img4 : _type_        numpy array 
        _description_    the images reconstructed by 10 principal components
    """
    fig, ax = plt.subplots(2,2, figsize=(10,8), sharey=True, sharex=True )
    ax[0][0].imshow(img1, cmap='gray')
    ax[0][0].set_title("all principal components")
    ax[0][1].imshow(img2, cmap='gray')
    ax[0][1].set_title("120 principal components")
    ax[1][0].imshow(img3, cmap='gray')
    ax[1][0].set_title("50 principal components")
    ax[1][1].imshow(img4, cmap='gray')
    ax[1][1].set_title("10 principal components")

def vis_path(data):
    """_summary_          visualize the original paths of first two pedestrians

    Parameters
    ----------
    data : _type_         numpy array 
        _description_     paths of pedestrians transformed into 2-dimensional numpy array
    """
    fig, ax = plt.subplots(1,1, figsize=(8,8) )
    ax.plot(data[:,0], data[:,1])
    ax.plot(data[:,2], data[:,3])
    ax.set_xlabel("") 
    ax.set_ylabel("")
    #ax.set_xlim([-2.5,2.5])
    #ax.set_ylim([-1.5,1.5])   

def vis_paths(data1, data2, data3):
    """_summary_          visualize the reconstructed paths of pedestrians based on different number of principal components

    Parameters
    ----------
    data1 : _type_        numpy array 
        _description_     paths of pedestrians reconstructed by 1 principal component
    data2 : _type_        numpy array 
        _description_     paths of pedestrians reconstructed by 2 principal components
    data3 : _type_        numpy array 
        _description_     paths of pedestrians reconstructed by 3 principal components
    """
    fig, ax = plt.subplots(3,1, figsize=(10,10),sharey=True, sharex=True )
    ax[0].plot(data1[:,0], data1[:,1])
    ax[0].plot(data1[:,2], data1[:,3])
    ax[0].set_xlabel("x coordinate") 
    ax[0].set_ylabel("y coordinate")
    ax[1].plot(data2[:,0], data2[:,1])
    ax[1].plot(data2[:,2], data2[:,3])
    ax[2].plot(data3[:,0], data3[:,1])
    ax[2].plot(data3[:,2], data3[:,3])

    

def vis_all_paths(data):
    """_summary_          visualize the original paths of all fifteen pedestrians

    Parameters
    ----------
    data : _type_         numpy array 
        _description_     paths of pedestrians transformed into 2-dimensional numpy array
    """
    fig, ax = plt.subplots(1,1, figsize=(8,8) )
    for i in range(int(data.shape[1]/2)):
        ax.plot(data[:,2*i], data[:,2*i+1])
    ax.set_xlabel("") 
    ax.set_ylabel("")
    #ax.set_xlim([-2.5,2.5])
    #ax.set_ylim([-1.5,1.5])       