import numpy as np
import pandas as pd
import scipy
import cv2
import matplotlib.pyplot as plt


def read_file(path):
    """_summary_

    Parameters
    ----------
    path : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    data = pd.read_csv(path, delimiter=" ", header = None)
    return data.to_numpy()

def load_image():
    img = scipy.misc.face(gray=True)
    img = cv2.resize(img, dsize=(249, 185))  
    return img

class PCA:
    def __init__(self, x):
        self.x = x

    def mean(self,x):
        return np.mean(x, axis=0)
    
    def center_data(self, x):
        return x - self.mean(x) 

    def svd(self, x_centered):
        u, s, vh = np.linalg.svd(x_centered, full_matrices=True)
        s = np.diag(s)   # is s squared or not.
        return u,s,vh

    def pca_cal(self, x):
        x_centered = self.center_data(x)
        u,s,vh = self.svd(x_centered)
        return u,s,vh
    
    def pca_reconstruction(self,u,s,vh,r):
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
        if L > s.shape[0]:
            raise ValueError ("The input number of principal components is too large.")
        energy = np.trace(s[:L,:L]**2) / np.trace(s**2)

        if sentence:
            return "The energy resulting from the given number of principal components is " + str(energy*100) + " %"
        else:
            return energy*100
    

def vis(data):
    fig, ax = plt.subplots(1,1, figsize=(8,4) )
    ax.scatter(data[:,0], data[:,1])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-1.5,1.5])
    #fig.savefig("results/fig1.pdf")

def vis_compare(data1,data2):
    fig, ax = plt.subplots(1,1, figsize=(8,4) )
    ax.scatter(data1[:,0], data1[:,1], label = "original" )
    ax.scatter(data2[:,0], data2[:,1], label = "centered")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-1.5,1.5])
    ax.legend()

def vis_pc(data, vh):
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
    plt.imshow(image, cmap='gray')     

def vis_images(img1, img2, img3, img4):
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
    fig, ax = plt.subplots(1,1, figsize=(8,8) )
    ax.plot(data[:,0], data[:,1])
    ax.plot(data[:,2], data[:,3])
    ax.set_xlabel("") 
    ax.set_ylabel("")
    #ax.set_xlim([-2.5,2.5])
    #ax.set_ylim([-1.5,1.5])   

def vis_paths(data1, data2, data3):
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
    fig, ax = plt.subplots(1,1, figsize=(8,8) )
    for i in range(int(data.shape[1]/2)):
        ax.plot(data[:,2*i], data[:,2*i+1])
    ax.set_xlabel("") 
    ax.set_ylabel("")
    #ax.set_xlim([-2.5,2.5])
    #ax.set_ylim([-1.5,1.5])       