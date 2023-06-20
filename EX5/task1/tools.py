import matplotlib.pyplot as plt
import numpy as np
import random

def load_data(file_path):
    """load the datasets and split the two columns.

    Args:
        file_path: the path of datasets

    Returns:
        x: the first column of data
        fx: the second column of data
    """
    data = np.loadtxt(file_path)
    x = np.reshape(data[:,0], (data.shape[0], -1))
    fx = np.reshape(data[:,1], (data.shape[0], -1))
    
    return x,fx

def linear_appoximation_function(x, fx):
    """use least-sqaures minimization to minimises |fx-x*A.T| and find matrix A

    Args:
        x : the coordinate x of data
        fx : the coordinate y of data

    Returns:
        A: the transpose of matrics A
    """
    A = np.linalg.lstsq(x, fx, rcond = 0.01)[0].T
    return A


def nonlinear_appoximation_function(x, fx, eps, L):
    """
    1.Randomly selects L centers for the rbf.
    2.computes the transformation matrix C.
    3.use least-sqaures minimization to minimises |fx-phi(x)*C.T| and find matrix C

    Args:
        x : the coordinate x of data
        fx : the coordinate y of data
        eps : the denominator of phi
        L : the numbers of rbf

    Returns:
        C: the matric C which is calculate by the least-suqres minimization
        centers: L random centers
    """
    # randomly choose L data pointsx for the centers of the radial basis functions
    random.seed(6)
    centers = x[random.sample(range(x.shape[0]), L)]
    # transformation of the data matrix x to the new data matrix phi
    phi = rbf(x, centers, eps)
    return np.linalg.lstsq(phi, fx, rcond = 0.01)[0].T, centers

def rbf_single(x, x_L, eps):
    """ compute the rbf for single x

    Args:
         x : the input value of rbf function
        x_L : the center of rbf function
        eps : the bandiwdth of rbf,in her we choose the eps**2

    Returns:
        phi: the part of special functions φ: so-called radial basis functions
    """
    return np.exp(-sum((x_L-x)**2)/(eps**2))

def rbf(x, x_L, eps):
    
    """ compute the rbf for multiple x

    Args:
        x : the input value of rbf function
        x_L : the center of rbf function
        eps : the bandiwdth of rbf

    Returns:
        phi: the special functions φ: so-called radial basis functions
    """
    phi = np.array([[rbf_single(x[i,:], center, eps) for center in x_L] for i in range(x.shape[0])])
    return phi

def plot_approximation_figure(x, fx, matrix, filename, linear):
    """plot the approximation figures based on the boolean condition linear.

    Args:
        x : the input value of rbf function
        x_L : the center of rbf function
        matrix : 
                 1. if linear=true, this matrix is the A
                 2. if linear= false, this matrix is the combination of [C, centers, eps, L]
            
        filename : save the picture in specific name.
        linear : use the linear aprroximation function / nonlinear approximation function
    """
    
    # create figure
    fig, ax = plt.subplots()
    ax.plot(x, fx, 'o', label='Data Original', markersize=2)

    # construct x values for the approximation function between the highest and lowest data point in x
    x_approx = np.linspace(min(x), max(x), 100)

    if linear is True:
        # linearly approximate y 
        y_approx = x_approx @ matrix.T
    else:
        # nonlinear approximate y
        phi_approx = rbf(x_approx, matrix[1], matrix[2])
        y_approx = phi_approx @ matrix[0].T
        
    # plot approximation ffunction of the data
    ax.plot(x_approx, y_approx, label='Data Approximation')
 
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_ylim(-2.0, 2.0)
    plt.legend(loc='upper right')
    plt.show() 
    fig.savefig(filename, dpi=200)
    
    
def plot_experiment_figure(eps,L,x,fx):
    """_summary_

    Args:
        eps : the bandiwdth of rbf
        L: the numbers of rbf
        x : the coordinate x of data
        fx : the coordinate y of data
    """
    # bilid a plot with subplots
    fig, axs = plt.subplots(len(eps), 1, figsize=(8, 6), sharex=True)

    for i in range(len(eps)):

        # current subplots (smae bandwidth)
        ax = axs[i]
        ax.plot(x, fx, 'o', label='original data', markersize=2)
        
        for j in range(len(L)):
            x_approx = np.linspace(min(x), max(x), 100)

            # calculate the nonlinear parameters which is necessary for nonlinear approximation
            C, centers = nonlinear_appoximation_function(x, fx, eps[i], L[j])
            phi_approx = rbf(x_approx, centers, eps[i])
            y_approx = phi_approx @ C.T
            
            ax.plot(x_approx, y_approx, label='L = ' + str(L[j]))
            ax.set_ylim(-2.0, 2.0)
            ax.legend(loc='upper right')
            ax.set_title('bandwidth ε = ' + str(eps[i]))


    plt.xlabel('x')
    plt.tight_layout()
    plt.show()
    fig.savefig('experiments.png', dpi=300)