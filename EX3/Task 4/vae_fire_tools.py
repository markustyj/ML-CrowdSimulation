#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:39:32 2023

@author: jingzhang

"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def load_reshape_dataset():    
    """
    Function to Load and reshape dataset.
    
    Args:
        train_data, test_data: load the train/test data from files.    
        reshaped_train, reshaped_test: reshape the train/test data to add an extra dimension.
        fire_train, fire_test: convert the reshaped train/test data to a Torch tensor. 
                               add an extra dimension to the train data/test data tensor.

    Returns:
        z : return the original train and test data, as well as the reshaped and processed train and test data.
    
    """
    
    train_data = np.load('FireEvac_train_set.npy')
    test_data = np.load('FireEvac_test_set.npy')
    
    
    reshaped_train = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
    fire_train = torch.Tensor(reshaped_train)
    fire_train = torch.unsqueeze(fire_train, 1)

    reshaped_test = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
    fire_test = torch.Tensor(reshaped_test)
    fire_test = torch.unsqueeze(fire_test, 1)
    
    return train_data, test_data, fire_train, fire_test


def scatterplot_traindata(train_data):    
    """ 
    Scatter plot of the train data.
    
    """
    
    plt.figure(figsize=(8, 6))
    plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', s=5, label='Scatterplot Traindata')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Train Data')
    plt.legend()
    plt.savefig("outputs/scatterplot_traindata.png")  
    plt.show()
    
    
def scatterplot_testdata(test_data):    
    """ 
    Scatter plot of the test data.
    
    """
    
    plt.figure(figsize=(8, 6))
    plt.scatter(test_data[:, 0], test_data[:, 1], color='green', s=5, label='Scatterplot Testdata')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Test Data')
    plt.legend()
    plt.savefig("outputs/scatterplot_testdata.png")  
    plt.show()
    
    
def get_train_loss(model, fire_train, optimizer, epochs=200):
    
    """
    Function to compute the train loss for a given model and dataset.
 
    Args:
        model: VAE Model
        fire_train: the reshaped and processed train dataset.
        optimizer: The optimizer used for training the model.
        epochs: The number of iterations to train the model, default: 200.
     
    Returns:
        The average train loss over all epochs.
        
    """
    
    model.train()         
    loss_sum=0
    total=0   
    for data in fire_train:
        optimizer.zero_grad()
        x_mean, mu, sigma = model(data)
        loss = ((data - x_mean)**2).sum() + (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        loss_sum += loss.item()
        total += data.size(0)      
        loss.backward()
        optimizer.step()
    
    train_loss = loss_sum/total       
    return train_loss
        
        
def get_test_loss(model, fire_test, epochs=200):     
    """
    Function to compute the test loss for a given model and dataset.
 
    Args:
        model: VAE Model.
        fire_test: the reshaped and processed test dataset.
        epochs: The number of iterations to train the model, default: 200.
     
    Returns:
        The average test loss over all epochs.
        
    """
    
    model.eval()       
    loss_sum=0
    total=0        
    with torch.no_grad():
        for data in fire_test:
            x_mean, mu, sigma = model(data)
            loss = ((data - x_mean)**2).sum() + (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            loss_sum += loss.item()
            total += data.size(0)        
        
         
    test_loss = loss_sum/total       
    return test_loss
    
    
    
def plot_loss(train_loss_list, test_loss_list):
    """
    Function to plot curve of train losses and curve of test losses.
 
    Args:
        train_loss_list: the train losses list.
        test_loss_list: the test losses list.
        
    """
    
    plt.figure(2)
    plt.plot(train_loss_list,label = 'train loss')
    plt.plot(test_loss_list, label = 'test loss')
    plt.title('Training & Testing loss in epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig("outputs/plot_loss.png")  
    
    
def plot_reconstructed_testset(model, fire_test):
    """
    Function to plot the reconstructed test set.
 
    Args:
        recon_data: reconstructed data that obtained after passing the test data through the model.
        reshaped_data: reshape the recon_data to match the desired dimensions.
        x, y: get the x and y coordinates from the reshaped_data
        
    """
    
    with torch.no_grad():
        recon_data, mean, std = model(torch.Tensor(fire_test))
    
    reshaped_data = recon_data.view(600, 2) 
    x = reshaped_data[:, 0].numpy()  
    y = reshaped_data[:, 1].numpy()
    plt.scatter(x, y)
    plt.title('Reconstructed Testset Scatterplot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig("outputs/scatterplot_reconstructed_testset.png")  
    plt.show()
    
    
def plot_generated_sample(model, num_smaples, latent_dim):
    """
    Function to plot the generated samples.
 
    Args:
        prior_samples: a tensor of random samples drawn from a standard normal distribution
                       shape is: (num_samples, latent_dim).                   
        torch.no_grad(): ensures that no gradients are computed during this forward pass.    
        generated_digits: the results after passing prior_samples through the decoder
                          then detached from the computation graph and converted to a NumPy array.                        
        generated_np: reshape the array to remove the singleton dimensions.
        x, y: get the x and y coordinates from the generated_np
        
    """

    prior_samples = torch.randn(num_smaples, latent_dim) 
    with torch.no_grad():
        generated_digits = model.decoder(prior_samples).detach().numpy()
        
    
    generated_np = generated_digits.reshape(-1, 2)
    x = generated_np[:, 0]
    y = generated_np[:, 1]

    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Digits Scatterplot')
    plt.savefig("outputs/scatterplot_generated_samples.png")  
    plt.show()
    
    

def generate_data(model, latent_dim):
    """
    Function to generate a sample within the MI building.
 
    Args:
        prior_samples: a tensor of random samples drawn from a standard normal distribution
                       shape is: (1, latent_dim).  
        torch.no_grad(): ensures that no gradients are computed during this forward pass.    
        generated_sample: the results after passing prior_samples through the decoder
                          then detached from the computation graph and converted to a NumPy array.   
        
    """
    
    prior_samples = torch.randn(1, latent_dim)
    with torch.no_grad():
        generated_sample = model.decoder(prior_samples).detach().numpy()
    
    return generated_sample


def estimate_critical_number(model, latent_dim): 
    """
    Function to calculate the number of samples needed to exceed the critical number.
 
    Args:
        max_people: set the maximum number of people to count within the sensitive area.
        x_min, x_max, y_min, y_max: Define the boundaries of the sensitive area.
        num_people: number of people within the sensitive area.
        num_samples: total number of generated samples.
        generated_sample: generate a sample using the VAE model.
        x_pos, y_pos: get the x and y positions from the generated sample.
     
    Returns:
        the total number of generated samples.
        
    """
    
    max_people = 100  
    x_min, x_max = 130, 150
    y_min, y_max = 50, 70
    num_people = 0
    num_samples = 0 
    
    while num_people <= max_people:       
        generated_sample = generate_data(model, latent_dim)
        x_pos = generated_sample[0,0][0,0]
        y_pos = generated_sample[0,0][1,0]
        num_samples += 1

        if x_min <= x_pos <= x_max and y_min <= y_pos <= y_max:
            num_people += 1

    return num_samples
    
    
    
    
    
    
    
    
    
    