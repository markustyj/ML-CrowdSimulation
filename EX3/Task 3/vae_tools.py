import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200


def plot_latent(vae, data, num_batches=100):
    """plot the laten representaation

    Args:
        vae: the trainning model
        data: the output of encoder

    """
    for i, (x, y) in enumerate(data):
        z = vae.encoder(x)
        z = z.detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

            
def plot_samples(ax, samples):
    """ plot the images with axis
    Args:
        ax : axis to hold images
        samples: images need to be print
        
    """
    for index, sample in enumerate(samples):
        ax[index].imshow(np.squeeze(sample), cmap='gray')
        ax[index].axis('off')

            
def plot_original_data(images):
    """print the original data of the test set

    Args:
        images : the original data
        
    """
    fig, ax= plt.subplots(nrows=1, ncols=15)

    ax[0].set_title('Original digits                    ')
    plot_samples(ax[0:], images)

    plt.savefig('outputs/Original Digits')
    plt.show()

    
def plot_generated_data(vae,latent_dims,epoch):
    """plot the generated data which which use sample points in a normal distribution 
    in the latent space as input data. Then,only put the data to decoder of VAE.
    
    Args:
        vae : the vae model which is only use the decoder
        latent_dims: the latent dimension to generate related generated sample dimension 
        epoch : the current epoch
        
    """
   
    fig, ax= plt.subplots(nrows=1, ncols=15)

    prior_samples = torch.randn(15, latent_dims)
    generated_digits = vae.decoder(prior_samples).detach().numpy()
    
    ax[0].set_title('Genetated Digits after Epoch {}    '.format(epoch))
    plot_samples(ax[0:], generated_digits)
    
    plt.savefig('outputs/Generated Digits after Epoch {}.png'.format(epoch))
    plt.show()

    
def plot_reconstructed_data(vae,images,epoch):
    """plot the reconstructed images through both encoder and decoder of VAE

    Args:
        vae : the vae model which use both encoder and decoder part.
        images : images to input the encoder
        epoch : the current epoch
        
    """
    fig, ax= plt.subplots(nrows=1, ncols=15)

    recon_digits = vae(images).detach().numpy()
    
    ax[0].set_title('Reconstructed Digits after Epoch {}'.format(epoch))
    plot_samples(ax[0:], recon_digits)

    plt.savefig('outputs/Reconstructed Digits after Epoch {}.png'.format(epoch))
    plt.show()


def plot_latent_space(vae,epoch,train_loader):
    """The latent representations obtained by encoder are visualized,  using 2D  
    scatter plots for illustration. 

    Args:
        vae : the training VAE model
        epoch : current epoch
        train_loader : the training sets
        
    """
    plt.figure(epoch,figsize=(6,6))
    plot_latent(vae, train_loader)
    plt.title('latent representation of '+ 'Epoch {}'.format(epoch))
    plt.savefig('outputs/Latent representation of '+ 'Epoch {}.png'.format(epoch))
    plt.show()

    
def get_train_loss(train_loss_list,train_loader,opt,vae,loss_sum,total,epoch):
    """train the model and plot the loss curve of each epoch.

    Args:
        train_loss_list : list stores the loss of each epoch
        train_loader : training set
        opt: Adam optimizer, 0.001 learning rate.
        epoch (_type_): current epoch

    Returns:
        loss_list: loss list of whole epochs
        
    """
    # set in train mode
    vae.train()
    for x, y in train_loader:
            # clear the gradient
            opt.zero_grad()
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            loss_sum += loss.item()
            total += y.size(0)
            # back propagation ,calculate the current gradient 
            loss.backward()
            # update the parameters based on the gradient
            opt.step()
            
    print('Epoch', epoch, 'train loss', loss_sum/total)
    train_loss_list.append(loss_sum/total)
    
    return train_loss_list


def get_test_loss(test_loss_list,tests_loader,vae,loss_sum,total,epoch):
    
    """test the model and plot the loss curve of each epoch.This part does not 
       contains the back propagation of model.

    Args:
        test_loss_list : list stores the loss of each epoch
        test_loader : testing set
        epoch (_type_): current epoch

    Returns:
        loss_list: loss list of whole epochs
        
    """
    # set in test mode
    vae.eval()
    with torch.no_grad():
        for x, y in tests_loader:
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            loss_sum += loss.item()
            total += y.size(0)
            
    print('Epoch', epoch, 'test loss', loss_sum/total)
    test_loss_list.append(loss_sum/total)
    
    return test_loss_list


def train(vae, train_loader, test_loader,tests_loader,latent_dims, epochs=100):
    """the most important part of model training.

    Args:
        vae : the pre-defined model.
        latent_dims : the latent dimension of VAE model.
        epochs : the total epoch of training.

    Returns:
        _type_: _description_
        
    """
    opt = torch.optim.Adam(vae.parameters(),lr=0.001)
    train_loss_list=[]
    test_loss_list=[]
    images, number = next(iter(test_loader))
    # plot original digits
    plot_original_data(images)
    
    for epoch in range(epochs):
        loss_sum=0
        total=0
        # get taining loss
        train_loss_list = get_train_loss(train_loss_list,train_loader,opt,vae,loss_sum,total,epoch)
        # get the test loss
        test_loss_list = get_test_loss(test_loss_list,tests_loader,vae,loss_sum,total,epoch)

        if epoch in [0,4,24,49,99]:
            plot_generated_data(vae,latent_dims,epoch)
            plot_reconstructed_data(vae,images,epoch)
            plot_latent_space(vae,epoch,train_loader)

    return vae, train_loss_list , test_loss_list


def spilt_data(dataset):
    """after download the datasets, spilt the datasets into training sets and testing sets.

    Args:
        dataset : Minst datasets.

    Returns:
        train_loader,test_loader: return the loader of training sets and testing sets.
        
    """
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=15, shuffle=False)
    tests_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)
    
    return train_loader,test_loader,tests_loader

    
def plot_loss(train_loss_list, test_loss_list):
    """plot the training loss and testing loss into one figure.

    Args:
        train_loss_list: the loss list of training
        test_loss_list : the loss list of testing
        
    """
    plt.figure(2)
    plt.plot(train_loss_list,label = 'train loss')
    plt.plot(test_loss_list, label = 'test loss')
    plt.title('Training & Testing loss in epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('outputs/Plot loss')
        


