#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 16:46:49 2023

@author: jingzhang, Mengyue Wang
"""

import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200


class Encoder(nn.Module):
    """the encoder architecture of VAE, using pytorch"""

    def __init__(self, latent_dims, input_size):
        """define the strcture of vae

        Parameters:
            latent_dims: int
                the dimension of latent space
            input_size: int
                the dimension of input data
        """
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 256)
        self.linear5 = nn.Linear(256, 128)
        self.linear_mean = nn.Linear(128, latent_dims)
        self.linear_logvar = nn.Linear(128, latent_dims)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        mean = self.linear_mean(x)
        logvar = self.linear_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    """the decoder architecture of VAE, using pytorch"""

    def __init__(self, latent_dims, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 128)
        self.linear5 = nn.Linear(128, 256)

        self.linear_mean = nn.Linear(256, output_size)
        self.std_dev = nn.Parameter(torch.tensor(1.0))

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = F.relu(self.linear5(z))
        mean = self.linear_mean(z)

        return mean


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, input_size, output_size):
        """combine the encoder and decoder to compete the full archetecture of model

        Args:
            latent_dims: int
                the dimension of latent space
            input_size: int
                the dimension of input data
            output_size: int
                the dimension of output data
        """
        super(VariationalAutoencoder, self).__init__()
        self.latent_dims = latent_dims
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = Encoder(latent_dims, input_size)
        self.decoder = Decoder(latent_dims, output_size)
        self.weights_init()

    def weights_init(self):
        """xavier weight initialization to ensure a more stable output of neural network"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reparameterize(self, mean, log_var):
        """reparamterize the model to ensure the backpropagation of VAE

        Args:
            mean: int
                choose a mean for generating epsilon
            log_var: int
                choose a variance for generating the samples

        Returns:
            sample_z: the generated samples
        """
        epsilon = torch.randn_like(mean)
        sample_z = mean + torch.exp(0.5 * log_var) * epsilon
        return sample_z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_mean = self.decoder(z)
        return x_mean
