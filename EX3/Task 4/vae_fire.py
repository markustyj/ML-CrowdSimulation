#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:37:56 2023

@author: jingzhang
"""

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions


class VariationalEncoder(nn.Module):
    """ 
    This is the encoder part of VAE model
    
    """
    
    def __init__(self, latent_dims):
        """
        Initialize the encoder
        
        Args:
            latent_dims (int): use the latent space to construct encoder
            
        """
        super(VariationalEncoder, self).__init__()
        # minist is 28 *28 dimensinal datasets
        self.linear1 = nn.Linear(2, 64)
        self.linear2 = nn.Linear(64,64)
        self.linearMean = nn.Linear(64, latent_dims)
        self.linearStd = nn.Linear(64, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    
    def forward(self, x):
        """
        The froward process of encoder,espically add the reparameterize process to ensure back propogation.
        
        Args:
            x : the imput data of encoder.

        Returns:
            z : the latent space representation.
            
        """
        x = torch.flatten(x, start_dim=1)
        # first encoder layer activated function is relu
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # calculate the mean in latent space
        mu =  self.linearMean(x)
        # calculate the standard deviation in latent space
        sigma = self.linearStd(x)
        # reparameterizes
        sigma = torch.exp(0.5*sigma)
        z = mu + sigma*self.N.sample(mu.shape)
        # calculate the diffenrences between latent space distribution and standard normal distribution
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu, sigma

    
class VariationalAutoencoder(nn.Module):
    """
    Here is the completed model of VAE,including encoder and decoder.
    
    """
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    
    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        x_mean = self.decoder(z) 
        
        return x_mean, mu, sigma
 
    
class Decoder(nn.Module):
    """
    The decoder part of VAE.
    
    Args:
        self.std_dev: implement the standard deviation for the decoder distribution as one.
    
    """
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linearMean = nn.Linear(64, 2)
        
        self.std_dev = nn.Parameter(torch.tensor(1.0))

        
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.relu(self.linear2(z))
        z = self.linearMean(z)
        return z.reshape((-1, 1, 2, 1))